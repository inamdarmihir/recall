# API Reference

## compute_key

```python
from recall.keys import compute_key

key = compute_key(
    subagent_type: str,
    system_prompt: str,
    model: str,
    tools_version: str,
    task_input: Any,
) -> str   # SHA-256 hex digest
```

## RecallStore

```python
store.lookup_exact(cache_key: str) -> dict | None
store.insert(cache_key: str, result: Any, provenance: dict, embed_fn=None)
```

## SubagentResultCacheMiddleware

```python
# Sync
result = middleware.wrap_tool_call(subagent_type, system_prompt, model, tools_version, task_input, invoke)

# Async
result = await middleware.awrap_tool_call(subagent_type, system_prompt, model, tools_version, task_input, invoke)
```

## ProvenanceVerifier

```python
from recall.invalidation import ProvenanceVerifier

verifier = ProvenanceVerifier(store, max_age_seconds=86400)
verifier.is_stale(cache_key: str) -> bool
verifier.invalidate_by_tools_version(store, old_version: str) -> int  # count deleted
```

## Memory

```python
from recall.memory import build_memory, record_cache_event, query_cache_stats

memory = build_memory(qdrant_url, collection_name)
record_cache_event(memory, subagent_type, hit: bool, model)
query_cache_stats(memory, subagent_type) -> list[dict]
```

## Agno + LangGraph

```python
from recall.agent import build_agno_cache_agent, build_langgraph_cache_pipeline

agent = build_agno_cache_agent(store, memory=None, model="gpt-4o")
pipeline = build_langgraph_cache_pipeline(store, middleware, memory=None)
```
