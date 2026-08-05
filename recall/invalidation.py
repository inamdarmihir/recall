"""Provenance verification and staleness detection for cached results."""
from __future__ import annotations
import time
from typing import Any
from recall.store import RecallStore


class ProvenanceVerifier:
    """Checks whether a cached result is still valid against staleness rules."""

    def __init__(self, store: RecallStore, max_age_seconds: float = 86400.0) -> None:
        self._store = store
        self._max_age = max_age_seconds

    def is_stale(self, cache_key: str) -> bool:
        """Return True if the cached entry is older than max_age_seconds."""
        entry = self._store.lookup_exact(cache_key)
        if not entry:
            return True
        inserted_at = entry.get("inserted_at", 0)
        return (time.time() - inserted_at) > self._max_age

    def invalidate_by_tools_version(self, store: RecallStore, old_version: str) -> int:
        """
        Scroll through cache and delete entries from an outdated tools_version.
        Returns the count of deleted entries.
        """
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        count = 0
        offset = None
        while True:
            hits, next_offset = store._client.scroll(
                store._collection,
                scroll_filter=Filter(must=[
                    FieldCondition(key="tools_version", match=MatchValue(value=old_version))
                ]),
                limit=100,
                offset=offset,
                with_payload=False,
            )
            if not hits:
                break
            store._client.delete(
                store._collection,
                points_selector=[h.id for h in hits],
            )
            count += len(hits)
            if next_offset is None:
                break
            offset = next_offset
        return count
