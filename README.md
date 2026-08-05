<div align="center">

# recall

**Content-addressed result cache for Deep Agents — eliminate redundant LLM API calls across subagent dispatches.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Qdrant](https://img.shields.io/badge/vector--db-Qdrant-red.svg)](https://qdrant.tech)
[![Agno](https://img.shields.io/badge/agent-agno%20v2.8.6-blueviolet.svg)](https://github.com/agno-agi/agno)
[![mem0](https://img.shields.io/badge/memory-mem0%20v3.0.0-green.svg)](https://mem0.ai)
[![LangGraph](https://img.shields.io/badge/orchestration-LangGraph%20v1.2.10-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![deepagents](https://img.shields.io/badge/harness-deepagents-lightgrey.svg)](https://pypi.org/project/deepagents/)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-informational.svg)](https://inamdarmihir.github.io/recall/)

</div>

---

## The Problem

In a multi-agent system, the same research task, code analysis, or summarisation job is dispatched to a subagent dozens of times across different sessions. Each dispatch burns LLM tokens even though the result is deterministic given the same inputs.

## The Solution

**recall** intercepts subagent tool calls via middleware, computes a SHA-256 content-addressed key over `(subagent_type, system_prompt, model, tools_version, task_input)`, and returns a cached Qdrant result for **pure** subagents. Impure subagents always pass through.

## Cost Model

| Scenario | Without recall | With recall (90% hit rate) |
|---|---|---|
| 1,000 dispatches × $0.01 | $10.00 | **$1.00** |
| 10,000 dispatches × $0.01 | $100.00 | **$10.00** |
| 100,000 dispatches × $0.01 | $1,000.00 | **$100.00** |

## Subagent Purity

| Purity | Description | Cached? |
|---|---|---|
| `PURE` | Deterministic given inputs (researcher, summariser) | ✅ Yes |
| `IMPURE` | Has side effects or randomness (browser, executor) | ❌ No |
| `CONDITIONAL` | Opt-in caching with caller acknowledgment | ⚠️ Explicit only |

## How It Works

```
Agent dispatches subagent
  │
  ▼
SubagentResultCacheMiddleware.wrap_tool_call()
  ├─ classify_purity() — IMPURE → pass through immediately
  ├─ compute_key() — SHA-256(type+prompt+model+tools+input)
  ├─ RecallStore.lookup_exact() — Qdrant payload scan
  │    HIT  → return cached result (0 LLM calls)
  │    MISS → invoke() → store in Qdrant → return result
  └─ mem0 — cache analytics (hit/miss rates)
```

## Quick Start

```bash
pip install recall-cache
docker run -p 6333:6333 qdrant/qdrant
```

```python
from qdrant_client import QdrantClient
from recall import RecallStore, SubagentResultCacheMiddleware
from recall.keys import SubagentPurity
from recall.memory import build_memory

store = RecallStore(QdrantClient(":memory:"))
memory = build_memory()

profiles = {
    "researcher": {"purity": SubagentPurity.PURE},
    "browser":    {"purity": SubagentPurity.IMPURE},
}
middleware = SubagentResultCacheMiddleware(store, profiles)

def dispatch_researcher(task: str) -> str:
    return f"Research result for: {task}"   # expensive LLM call

# First call — hits LLM and stores result
r1 = middleware.wrap_tool_call("researcher", "You are a researcher", "gpt-4o", "1.0", "task A", lambda: dispatch_researcher("task A"))
# Second call — served from Qdrant cache (0 extra API calls)
r2 = middleware.wrap_tool_call("researcher", "You are a researcher", "gpt-4o", "1.0", "task A", lambda: dispatch_researcher("task A"))
assert r1 == r2
```

```bash
recall lookup --subagent-type researcher --input "task A"
recall stats
```

## Agno + LangGraph

```python
from recall.agent import build_agno_cache_agent, build_langgraph_cache_pipeline

agent = build_agno_cache_agent(store, memory=memory)
agent.print_response("What is the cache hit rate for the researcher subagent?")

pipeline = build_langgraph_cache_pipeline(store, middleware, memory=memory)
```

## Configuration

| Variable | Default | Description |
|---|---|---|
| `QDRANT_URL` | `http://localhost:6333` | Qdrant instance URL |
| `RECALL_TOOLS_VERSION` | `1.0.0` | Invalidates cache when tools change |
| `RECALL_MAX_AGE_SECONDS` | `86400` | TTL for cached results (1 day) |

## Tech Stack

| Component | Purpose |
|---|---|
| [Qdrant](https://qdrant.tech) `>=1.18.0` | Content-addressed result store |
| [Agno](https://github.com/agno-agi/agno) `>=2.8.6` | Cache management agent |
| [mem0](https://mem0.ai) `>=3.0.0` | Hit/miss analytics memory |
| [LangGraph](https://langchain-ai.github.io/langgraph/) `>=1.2.10` | classify→cache→invoke pipeline |
| [deepagents](https://pypi.org/project/deepagents/) | Sub-agent harness middleware |

## License

MIT — see [LICENSE](LICENSE).
