"""Qdrant-backed store for exact and optional semantic subagent result lookup."""

from __future__ import annotations

from typing import Any

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
    """Create the recall collection with named vectors and payload indexes if missing."""
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
    """Exact (payload scroll) and optional semantic (query_points) cache store."""

    def __init__(self, client: AsyncQdrantClient, collection: str = "recall") -> None:
        self.client = client
        self.collection = collection

    async def lookup_exact(self, exact_hash: str, subagent_type: str) -> dict[str, Any] | None:
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
            return {
                "point_id": hits[0].id,
                "payload": payload,
                "result": payload["result"],
            }
        return None

    async def lookup_semantic(
        self,
        task_embed: list[float],
        subagent_type: str,
        *,
        threshold: float,
    ) -> str | None:
        """Opt-in ANN path — uses query_points against the named `task_embed` vector."""
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
            return payload.get("result")
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
