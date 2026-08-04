"""Provenance-based invalidation for cached subagent results.

TTL-based expiry is the wrong model for this workload: a summary of an
unchanged file is still valid weeks later, while a summary of a file that
changed five minutes ago is stale immediately.

An entry stays valid while every recorded ``source_ref`` still hashes to
the value stored at write time. Content hashes folded into the cache key
(see :mod:`recall.keys`) catch most staleness via key mismatch;
:func:`verify_provenance` is a second line of defense (and soft-marks the
entry ``stale=True`` so subsequent lookups skip it).

See ``docs/ARTICLE.md`` §10.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Protocol

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue


class _StoreLike(Protocol):
    """Minimal store surface needed by :func:`verify_provenance`."""

    client: AsyncQdrantClient
    collection: str


async def verify_provenance(
    store: _StoreLike,
    content_hash_fn: Callable[[str], str],
    cached: Mapping[str, Any],
) -> bool:
    """Re-hash every recorded ``source_ref``; soft-invalidate on mismatch.

    Parameters
    ----------
    store:
        Object exposing ``.client`` and ``.collection`` (typically a
        :class:`~recall.store.SubagentCacheStore`).
    content_hash_fn:
        Same hasher used at key-computation time
        (``path -> "sha256:..."``).
    cached:
        Hit dict from :meth:`SubagentCacheStore.lookup_exact` with
        ``point_id`` and ``payload`` keys.

    Returns
    -------
    bool
        ``True`` if still valid (or the entry has no file provenance,
        e.g. a pure classifier). ``False`` after marking ``stale=True``.
    """
    content_hashes: dict[str, str] = cached["payload"].get("content_hashes") or {}
    if not content_hashes:
        return True  # pure-without-files (e.g. closed-label classification)
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
    """Bulk-delete entries for a prompt / tool-schema change.

    Uses a payload-index filter — O(matching entries), not an ANN search.
    Call this when you bump ``tools_version`` or rewrite a subagent's
    system prompt and want to drop the previous generation of results.

    Example::

        await invalidate_profile(client, "recall", "doc-summarizer", "a3f9c2d1")
    """
    await client.delete(
        collection_name=collection,
        points_selector=Filter(
            must=[
                FieldCondition(key="subagent_type", match=MatchValue(value=subagent_type)),
                FieldCondition(key="tools_version", match=MatchValue(value=old_tools_version)),
            ]
        ),
    )
