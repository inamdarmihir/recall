"""Tests for provenance verification and profile invalidation."""

from __future__ import annotations

import pytest
from qdrant_client import AsyncQdrantClient

from recall.invalidation import invalidate_profile, verify_provenance
from recall.store import SubagentCacheStore, ensure_collection


@pytest.fixture
async def store() -> SubagentCacheStore:
    client = AsyncQdrantClient(location=":memory:")
    await ensure_collection(client, "recall", embed_dim=4)
    return SubagentCacheStore(client, "recall")


@pytest.mark.asyncio
async def test_verify_provenance_ok_and_stale(store: SubagentCacheStore) -> None:
    zero = [0.0] * 4
    await store.insert(
        point_id="44444444-4444-4444-4444-444444444444",
        exact_hash="sha256:p",
        subagent_type="doc-summarizer",
        model="anthropic:claude-haiku-4-5",
        tools_version="v1",
        source_refs=["a.md"],
        content_hashes={"a.md": "sha256:deadbeef"},
        task_embed=zero,
        result_embed=zero,
        result="ok",
        created_at="2026-07-30T00:00:00+00:00",
    )
    cached = await store.lookup_exact("sha256:p", "doc-summarizer")
    assert cached is not None
    assert await verify_provenance(store, lambda _: "sha256:deadbeef", cached) is True

    cached = await store.lookup_exact("sha256:p", "doc-summarizer")
    assert cached is not None
    assert await verify_provenance(store, lambda _: "sha256:changed", cached) is False
    assert await store.lookup_exact("sha256:p", "doc-summarizer") is None


@pytest.mark.asyncio
async def test_invalidate_profile_deletes_matching(store: SubagentCacheStore) -> None:
    zero = [0.0] * 4
    await store.insert(
        point_id="55555555-5555-5555-5555-555555555555",
        exact_hash="sha256:x",
        subagent_type="doc-summarizer",
        model="anthropic:claude-haiku-4-5",
        tools_version="old",
        source_refs=[],
        content_hashes={},
        task_embed=zero,
        result_embed=zero,
        result="x",
        created_at="2026-07-30T00:00:00+00:00",
    )
    await invalidate_profile(store.client, store.collection, "doc-summarizer", "old")
    assert await store.lookup_exact("sha256:x", "doc-summarizer") is None
