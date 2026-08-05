"""Qdrant-backed content-addressed result store."""
from __future__ import annotations
import uuid
import time
from typing import Any
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue, PayloadSchemaType,
)


EMBED_DIM = 384
_COLLECTION = "recall_cache"


class RecallStore:
    """Content-addressed store: exact hash lookup with optional semantic fallback."""

    def __init__(self, client: QdrantClient, collection: str = _COLLECTION) -> None:
        self._client = client
        self._collection = collection
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        existing = {c.name for c in self._client.get_collections().collections}
        if self._collection not in existing:
            self._client.create_collection(
                self._collection,
                vectors_config={"semantic": VectorParams(size=EMBED_DIM, distance=Distance.COSINE)},
            )
            self._client.create_payload_index(
                self._collection, "cache_key", PayloadSchemaType.KEYWORD
            )

    def lookup_exact(self, cache_key: str) -> dict[str, Any] | None:
        """Return cached result for an exact SHA-256 key match, or None."""
        hits, _ = self._client.scroll(
            self._collection,
            scroll_filter=Filter(must=[FieldCondition(key="cache_key", match=MatchValue(value=cache_key))]),
            limit=1,
            with_payload=True,
        )
        if hits:
            return hits[0].payload  # type: ignore[index]
        return None

    def insert(self, cache_key: str, result: Any, provenance: dict[str, Any], embed_fn=None) -> None:
        """Upsert a cached result with provenance metadata."""
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, cache_key))
        payload = {
            "cache_key": cache_key,
            "result": result,
            "inserted_at": time.time(),
            **provenance,
        }
        if embed_fn:
            vector = {"semantic": embed_fn(str(result))}
        else:
            vector = {"semantic": _hash_embed(str(result))}

        self._client.upsert(
            self._collection,
            points=[PointStruct(id=point_id, vector=vector, payload=payload)],
        )


def _hash_embed(text: str) -> list[float]:
    """Deterministic hash-based embedding (no ML deps) for tests."""
    import hashlib, struct
    digest = hashlib.sha256(text.encode()).digest()
    floats: list[float] = []
    for i in range(0, min(len(digest), EMBED_DIM * 4), 4):
        chunk = digest[i: i + 4].ljust(4, b"\x00")
        floats.append(struct.unpack("<f", chunk)[0])
    while len(floats) < EMBED_DIM:
        floats.append(0.0)
    norm = sum(x ** 2 for x in floats) ** 0.5 or 1.0
    return [x / norm for x in floats[:EMBED_DIM]]
