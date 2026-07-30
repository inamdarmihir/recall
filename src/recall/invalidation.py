"""Provenance-based invalidation for cached subagent results."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue


class _StoreLike(Protocol):
    client: AsyncQdrantClient
    collection: str


async def verify_provenance(
    store: _StoreLike,
    content_hash_fn: Callable[[str], str],
    cached: dict[str, Any],
) -> bool:
    """Re-hash every recorded source_ref; soft-invalidate on mismatch."""
    content_hashes: dict[str, str] = cached["payload"].get("content_hashes") or {}
    if not content_hashes:
        return True  # pure-without-files (e.g. classification)
    for ref, expected in content_hashes.items():
        if content_hash_fn(ref) != expected:
            await store.client.set_payload(
                collection_name=store.collection,
                payload={"stale": True},
                points=[cached["point_id"]],
            )
            return False
    return True


async def invalidate_profile(
    client: AsyncQdrantClient,
    collection: str,
    subagent_type: str,
    old_tools_version: str,
) -> None:
    """Bulk-invalidate on a prompt or tool-schema change via payload filter delete."""
    await client.delete(
        collection_name=collection,
        points_selector=Filter(
            must=[
                FieldCondition(key="subagent_type", match=MatchValue(value=subagent_type)),
                FieldCondition(key="tools_version", match=MatchValue(value=old_tools_version)),
            ]
        ),
    )
