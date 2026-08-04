"""Qdrant-backed store for exact and optional semantic subagent result lookup.

The collection uses:

- **Named vectors** ``task_embed`` / ``result_embed`` — reserved for the
  opt-in semantic path; exact lookups ignore them.
- **Payload indexes** on ``exact_hash``, ``subagent_type``, ``stale``, etc.
  so exact lookup is an index scan, not an ANN search.

Requires ``qdrant-client>=1.18`` for the ``query_points`` API (the older
``.search()`` method is deprecated and not used here).

See ``docs/ARTICLE.md`` §7 for schema rationale and false-hit mitigations.
"""

from __future__ import annotations

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from recall.types import CacheHit

# FastEmbed / all-MiniLM-L6-v2 default; use 1536 for text-embedding-3-small.
EMBED_DIM = 384

_PAYLOAD_INDEXES: list[tuple[str, PayloadSchemaType]] = [
    ("subagent_type", PayloadSchemaType.KEYWORD),
    ("model", PayloadSchemaType.KEYWORD),
    ("tools_version", PayloadSchemaType.KEYWORD),
    ("exact_hash", PayloadSchemaType.KEYWORD),
    ("source_refs", PayloadSchemaType.KEYWORD),
    ("stale", PayloadSchemaType.BOOL),
    ("created_at", PayloadSchemaType.KEYWORD),
]


async def ensure_collection(
    client: AsyncQdrantClient,
    collection: str = "recall",
    *,
    embed_dim: int = EMBED_DIM,
) -> None:
    """Create the recall collection with named vectors and payload indexes.

    Idempotent: if the collection already exists, only missing payload
    indexes are added. Both named vectors are declared even when running
    ``exact_only=True`` so enabling semantic matching later does not
    require a migration.

    Parameters
    ----------
    client:
        Connected :class:`AsyncQdrantClient` (URL, path, or ``:memory:``).
    collection:
        Collection name (default ``"recall"``).
    embed_dim:
        Dimension for both named vectors; must match the embedder in use.
    """
    existing = {c.name for c in (await client.get_collections()).collections}
    if collection not in existing:
        await client.create_collection(
            collection_name=collection,
            vectors_config={
                "task_embed": VectorParams(size=embed_dim, distance=Distance.COSINE),
                "result_embed": VectorParams(size=embed_dim, distance=Distance.COSINE),
            },
        )
    for field, schema in _PAYLOAD_INDEXES:
        try:
            await client.create_payload_index(
                collection_name=collection,
                field_name=field,
                field_schema=schema,
            )
        except Exception:
            # Index may already exist on a warm collection; safe to continue.
            continue


class SubagentCacheStore:
    """Exact (payload scroll) and optional semantic (``query_points``) cache store.

    Typical payload stored per point::

        {
          "subagent_type": "file-summarizer",
          "model": "anthropic:claude-haiku-4-5",
          "tools_version": "a3f9c2d1",
          "exact_hash": "sha256:...",
          "source_refs": ["docs/api.md"],
          "content_hashes": {"docs/api.md": "sha256:..."},
          "created_at": "2026-08-04T18:00:00+00:00",
          "stale": false,
          "result": "..."
        }
    """

    def __init__(self, client: AsyncQdrantClient, collection: str = "recall") -> None:
        self.client = client
        self.collection = collection

    async def lookup_exact(self, exact_hash: str, subagent_type: str) -> CacheHit | None:
        """Look up a non-stale entry by exact content hash + subagent type.

        Uses ``scroll`` with a payload filter — no ANN traversal. Returns
        ``None`` on miss.
        """
        hits, _ = await self.client.scroll(
            collection_name=self.collection,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="exact_hash", match=MatchValue(value=exact_hash)),
                    FieldCondition(key="subagent_type", match=MatchValue(value=subagent_type)),
                    FieldCondition(key="stale", match=MatchValue(value=False)),
                ]
            ),
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        if hits:
            payload = hits[0].payload or {}
            return CacheHit(
                point_id=hits[0].id,
                payload=payload,
                result=str(payload["result"]),
            )
        return None

    async def lookup_semantic(
        self,
        task_embed: list[float],
        subagent_type: str,
        *,
        threshold: float,
    ) -> str | None:
        """Opt-in ANN path against the named ``task_embed`` vector.

        Filtered by ``subagent_type`` and ``stale=False`` during HNSW
        traversal (payload-aware filtering). Prefer
        ``threshold >= 0.97``; lower thresholds risk false hits (article §11).
        """
        result = await self.client.query_points(
            collection_name=self.collection,
            query=task_embed,
            using="task_embed",
            query_filter=Filter(
                must=[
                    FieldCondition(key="subagent_type", match=MatchValue(value=subagent_type)),
                    FieldCondition(key="stale", match=MatchValue(value=False)),
                ]
            ),
            limit=1,
            score_threshold=threshold,
            with_payload=True,
        )
        if result.points:
            payload = result.points[0].payload or {}
            value = payload.get("result")
            return str(value) if value is not None else None
        return None

    async def insert(
        self,
        *,
        point_id: str,
        exact_hash: str,
        subagent_type: str,
        model: str,
        tools_version: str,
        source_refs: list[str],
        content_hashes: dict[str, str],
        task_embed: list[float],
        result_embed: list[float],
        result: str,
        created_at: str,
    ) -> None:
        """Upsert one cached subagent result into the collection."""
        await self.client.upsert(
            collection_name=self.collection,
            points=[
                PointStruct(
                    id=point_id,
                    vector={"task_embed": task_embed, "result_embed": result_embed},
                    payload={
                        "exact_hash": exact_hash,
                        "subagent_type": subagent_type,
                        "model": model,
                        "tools_version": tools_version,
                        "source_refs": source_refs,
                        "content_hashes": content_hashes,
                        "created_at": created_at,
                        "stale": False,
                        "result": result,
                    },
                )
            ],
        )
