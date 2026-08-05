"""Persistent cache analytics memory backed by mem0 + Qdrant."""
from __future__ import annotations
from typing import Any


def build_memory(qdrant_url: str = "http://localhost:6333", collection_name: str = "recall_memory"):
    from mem0 import Memory
    return Memory.from_config({
        "vector_store": {
            "provider": "qdrant",
            "config": {"url": qdrant_url, "collection_name": collection_name},
        }
    })


def record_cache_event(memory, subagent_type: str, hit: bool, model: str, agent_id: str = "recall") -> None:
    """Persist cache hit/miss events for cost analytics."""
    status = "HIT" if hit else "MISS"
    memory.add(
        f"Cache {status} for subagent '{subagent_type}' model '{model}'",
        user_id=agent_id,
        metadata={"subagent_type": subagent_type, "hit": hit, "model": model},
    )


def query_cache_stats(memory, subagent_type: str, agent_id: str = "recall") -> list[dict[str, Any]]:
    """Retrieve cache event history for a subagent type."""
    results = memory.search(f"cache events for {subagent_type}", user_id=agent_id)
    return results.get("results", [])
