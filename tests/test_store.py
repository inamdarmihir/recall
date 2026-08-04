"""Tests for Qdrant-backed SubagentCacheStore."""

from __future__ import annotations

import pytest
from qdrant_client import AsyncQdrantClient

from recall.store import SubagentCacheStore, ensure_collection


@pytest.fixture
async def store() -> SubagentCacheStore:
    client = AsyncQdrantClient(location=":memory:")
    await ensure_collection(client, "recall", embed_dim=8)
    return SubagentCacheStore(client, "recall")


@pytest.mark.asyncio
async def test_exact_lookup_roundtrip(store: SubagentCacheStore) -> None:
    zero = [0.0] * 8
    await store.insert(
        point_id="11111111-1111-1111-1111-111111111111",
        exact_hash="sha256:abc",
        subagent_type="doc-summarizer",
        model="anthropic:claude-haiku-4-5",
        tools_version="a3f9c2d1",
        source_refs=["docs/a.md"],
        content_hashes={"docs/a.md": "sha256:1"},
        task_embed=zero,
        result_embed=zero,
        result="summary-a",
        created_at="2026-08-04T00:00:00+00:00",
    )
    hit = await store.lookup_exact("sha256:abc", "doc-summarizer")
    assert hit is not None
    assert hit["result"] == "summary-a"
    miss = await store.lookup_exact("sha256:missing", "doc-summarizer")
    assert miss is None


@pytest.mark.asyncio
async def test_stale_entries_are_excluded(store: SubagentCacheStore) -> None:
    zero = [0.0] * 8
    await store.insert(
        point_id="22222222-2222-2222-2222-222222222222",
        exact_hash="sha256:stale",
        subagent_type="doc-summarizer",
        model="anthropic:claude-haiku-4-5",
        tools_version="a3f9c2d1",
        source_refs=[],
        content_hashes={},
        task_embed=zero,
        result_embed=zero,
        result="old",
        created_at="2026-08-04T00:00:00+00:00",
    )
    await store.client.set_payload(
        collection_name=store.collection,
        payload={"stale": True},
        points=["22222222-2222-2222-2222-222222222222"],
    )
    assert await store.lookup_exact("sha256:stale", "doc-summarizer") is None


@pytest.mark.asyncio
async def test_semantic_lookup_threshold(store: SubagentCacheStore) -> None:
    vec = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    await store.insert(
        point_id="33333333-3333-3333-3333-333333333333",
        exact_hash="sha256:sem",
        subagent_type="classifier",
        model="anthropic:claude-haiku-4-5",
        tools_version="a3f9c2d1",
        source_refs=[],
        content_hashes={},
        task_embed=vec,
        result_embed=vec,
        result="label:bug",
        created_at="2026-08-04T00:00:00+00:00",
    )
    hit = await store.lookup_semantic(vec, "classifier", threshold=0.99)
    assert hit == "label:bug"
    miss = await store.lookup_semantic(
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "classifier",
        threshold=0.99,
    )
    assert miss is None
