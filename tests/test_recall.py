"""Basic tests for recall cache (offline, in-memory Qdrant)."""
from __future__ import annotations
import pytest
from qdrant_client import QdrantClient
from recall.keys import compute_key, canonicalize, SubagentPurity, classify_purity
from recall.store import RecallStore
from recall.middleware import SubagentResultCacheMiddleware


@pytest.fixture
def store():
    client = QdrantClient(":memory:")
    return RecallStore(client)


def test_compute_key_stable():
    k1 = compute_key("researcher", "You are helpful", "gpt-4o", "1.0", {"task": "summarise"})
    k2 = compute_key("researcher", "You are helpful", "gpt-4o", "1.0", {"task": "summarise"})
    assert k1 == k2


def test_compute_key_differs_on_input():
    k1 = compute_key("researcher", "sys", "gpt-4o", "1.0", {"task": "A"})
    k2 = compute_key("researcher", "sys", "gpt-4o", "1.0", {"task": "B"})
    assert k1 != k2


def test_canonicalize_stable():
    assert canonicalize({"b": 2, "a": 1}) == canonicalize({"a": 1, "b": 2})


def test_store_insert_and_exact_lookup(store):
    key = compute_key("coder", "sys", "gpt-4o", "1.0", "write a hello world")
    store.insert(key, "print('hello')", {"subagent_type": "coder", "model": "gpt-4o"})
    result = store.lookup_exact(key)
    assert result is not None
    assert result["result"] == "print('hello')"


def test_store_miss(store):
    assert store.lookup_exact("nonexistent-key") is None


def test_purity_classification():
    profiles = {"coder": {"purity": "pure"}, "browser": {"purity": "impure"}}
    assert classify_purity("coder", profiles) == SubagentPurity.PURE
    assert classify_purity("browser", profiles) == SubagentPurity.IMPURE
    assert classify_purity("unknown", profiles) == SubagentPurity.IMPURE


def test_middleware_caches_pure_subagent(store):
    profiles = {"summariser": {"purity": "pure"}}
    middleware = SubagentResultCacheMiddleware(store, profiles)
    call_count = 0

    def invoke():
        nonlocal call_count
        call_count += 1
        return "summary result"

    result1 = middleware.wrap_tool_call("summariser", "sys", "gpt-4o", "1.0", "text to summarise", invoke)
    result2 = middleware.wrap_tool_call("summariser", "sys", "gpt-4o", "1.0", "text to summarise", invoke)
    assert result1 == "summary result"
    assert result2 == "summary result"
    assert call_count == 1  # second call served from cache


def test_middleware_skips_impure_subagent(store):
    profiles = {"browser": {"purity": "impure"}}
    middleware = SubagentResultCacheMiddleware(store, profiles)
    call_count = 0

    def invoke():
        nonlocal call_count
        call_count += 1
        return "browsed result"

    middleware.wrap_tool_call("browser", "sys", "gpt-4o", "1.0", "query", invoke)
    middleware.wrap_tool_call("browser", "sys", "gpt-4o", "1.0", "query", invoke)
    assert call_count == 2  # always invoked, never cached
