# Quick Start

## 1. Install

```bash
pip install recall-cache
docker run -p 6333:6333 qdrant/qdrant
```

## 2. Define purity profiles

```python
from recall.keys import SubagentPurity

profiles = {
    "researcher":  {"purity": SubagentPurity.PURE},
    "summariser":  {"purity": SubagentPurity.PURE},
    "browser":     {"purity": SubagentPurity.IMPURE},
    "code_runner": {"purity": SubagentPurity.IMPURE},
}
```

## 3. Wrap tool calls with middleware

```python
from qdrant_client import QdrantClient
from recall import RecallStore, SubagentResultCacheMiddleware
from recall.memory import build_memory

store = RecallStore(QdrantClient("http://localhost:6333"))
memory = build_memory()
middleware = SubagentResultCacheMiddleware(store, profiles)

def dispatch_researcher(task: str) -> str:
    print("  [LLM called]")    # only printed on first call
    return f"Research result for: {task}"

# First call — LLM invoked, result stored in Qdrant
r1 = middleware.wrap_tool_call(
    "researcher", "You are a researcher", "gpt-4o", "1.0", "task A",
    lambda: dispatch_researcher("task A"),
)
# Second call — served from Qdrant cache
r2 = middleware.wrap_tool_call(
    "researcher", "You are a researcher", "gpt-4o", "1.0", "task A",
    lambda: dispatch_researcher("task A"),
)
assert r1 == r2
```

## 4. Async variant

```python
async def run():
    result = await middleware.awrap_tool_call(
        "researcher", "sys", "gpt-4o", "1.0", "task B",
        invoke=async_dispatch,
    )
```

## 5. Inspect the cache

```bash
recall lookup --subagent-type researcher --input "task A"
# HIT: {'result': 'Research result for: task A', 'inserted_at': 1234567890.0}

recall stats
# Collection: recall_cache — 42 entries
```

## 6. Invalidate on tools upgrade

```python
from recall.invalidation import ProvenanceVerifier

verifier = ProvenanceVerifier(store)
count = verifier.invalidate_by_tools_version(store, old_version="1.0.0")
print(f"Invalidated {count} stale entries")
```

## 7. Track hit rate with mem0

```python
from recall.agent import build_agno_cache_agent

agent = build_agno_cache_agent(store, memory=memory)
agent.print_response("What is the cache hit rate for the researcher subagent this week?")
```
