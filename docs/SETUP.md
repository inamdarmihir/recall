# Setup guide

This guide gets **Recall** running against [Deep Agents](https://docs.langchain.com/oss/python/deepagents) end to end: install, wire the middleware, run tests, and try the map-reduce demo.

## Prerequisites

| Requirement | Notes |
|-------------|--------|
| Python **3.11+** | Developed on 3.12 |
| [Deep Agents](https://pypi.org/project/deepagents/) **≥ 0.7** | Pulls in LangChain / LangGraph |
| [Qdrant](https://qdrant.tech) | Local Docker is enough for demos |
| Anthropic API key | Only for the live demo, not for unit tests |

## 1. Clone and install

```bash
git clone https://github.com/inamdarmihir/recall.git
cd recall

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -e ".[dev]"
```

Optional extras:

```bash
# Local embeddings for semantic-match mode (off by default)
pip install -e ".[fastembed]"

# Live Anthropic demo
pip install -e ".[anthropic]"
```

Copy the env template if you plan to run the demo:

```bash
cp .env.example .env
# edit .env — set ANTHROPIC_API_KEY
```

## 2. Start Qdrant

```bash
docker run --rm -p 6333:6333 qdrant/qdrant
```

Dashboard (optional): [http://localhost:6333/dashboard](http://localhost:6333/dashboard).

For unit tests you do **not** need Docker — tests use `AsyncQdrantClient(location=":memory:")`.

## 3. Wire Recall into a Deep Agent

Keep three things in sync for each cacheable subagent:

1. The entry in `create_deep_agent(..., subagents=[...])` (`name`, `system_prompt`, `model`)
2. The matching Recall `profiles[name]` (`system_prompt`, `model`, `tools_version`, `pure=True`)
3. File paths the orchestrator names in `task` descriptions (backticks) or via a `files` arg — these are content-hashed into the cache key

```python
from deepagents import create_deep_agent
from recall import SubagentResultCacheMiddleware

DOC_SUMMARIZER_PROMPT = "Summarize structural changes and TODOs."

cache = SubagentResultCacheMiddleware(
    qdrant_url="http://localhost:6333",
    profiles={
        "doc-summarizer": {
            "system_prompt": DOC_SUMMARIZER_PROMPT,
            "model": "anthropic:claude-haiku-4-5",
            "tools_version": "a3f9c2d1",  # bump when tools/prompt change
            "pure": True,                 # opt-in: only pure profiles are cached
        },
        "web-researcher": {
            "system_prompt": "Live web research.",
            "model": "anthropic:claude-sonnet-4-6",
            "tools_version": "a3f9c2d1",
            "pure": False,                # always bypasses the cache
        },
    },
    exact_only=True,   # recommended default — see ARTICLE.md §11
    vfs_root="./docs", # root for resolving/hashing source files
)

agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-6",
    system_prompt="Delegate page reviews to doc-summarizer via the task tool.",
    subagents=[
        {
            "name": "doc-summarizer",
            "description": "Summarize a documentation page.",
            "system_prompt": DOC_SUMMARIZER_PROMPT,
            "model": "anthropic:claude-haiku-4-5",
        }
    ],
    middleware=[cache],
)

result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": "Summarize `api.md` with doc-summarizer.",
    }]
})
print(result["messages"][-1].content)
print(cache.stats)  # {"hits": ..., "misses": ..., "bypassed": ...}
```

### In-memory Qdrant (no Docker)

```python
cache = SubagentResultCacheMiddleware(
    qdrant_location=":memory:",
    profiles={...},
)
```

### On-disk embedded Qdrant

```python
cache = SubagentResultCacheMiddleware(
    qdrant_path=".qdrant",
    profiles={...},
)
```

## 4. Run the unit tests

No API key and no Docker required:

```bash
pytest -q
ruff check src tests examples
mypy src/recall
```

## 5. Run the map-reduce demo

```bash
# Terminal A
docker run --rm -p 6333:6333 qdrant/qdrant

# Terminal B
export ANTHROPIC_API_KEY=sk-...
pip install -e ".[dev,anthropic]"
python examples/map_reduce_docs.py
```

The demo fans out over `examples/sample_docs/*.md`, then runs a second pass. Unchanged pages should show up as cache hits in `cache.stats`.

## How the pieces fit together

```
Orchestrator
   │ task(subagent_type, description, files=[...])
   ▼
SubagentResultCacheMiddleware.awrap_tool_call
   1. is_pure(subagent_type)?           ── no ──▶ handler(req)
   2. compute_key(args)                          (keys.py)
   3. lookup_exact(key)                          (store.py)
   4. verify_provenance(hit)                     (invalidation.py)
   5. hit + valid  ──▶ return cached ToolMessage
      miss/invalid ──▶ handler(req) ──▶ store.insert(...)
```

| Module | Responsibility |
|--------|----------------|
| `keys.py` | Canonicalize inputs; SHA-256 key with file content hashes |
| `store.py` | Qdrant collection, exact scroll, optional `query_points` |
| `middleware.py` | Deep Agents `AgentMiddleware` hook (`awrap_tool_call`) |
| `invalidation.py` | Provenance check + bulk profile invalidation |
| `embeddings.py` | Pluggable embedders (`ZeroEmbedder` default) |

## Safety checklist

- Only mark a profile `pure=True` when the subagent is deterministic for the hashed inputs (no live web, no side-effecting tools).
- Keep `exact_only=True` in production until you have measured semantic false-hit risk (article §11).
- Bump `tools_version` (and call `invalidate_profile` if you need an immediate purge) whenever the subagent's tools or system prompt change.
- A false cache hit is worse than a miss — prefer misses when unsure.

## Verified stack

| Component | Version / ID |
|-----------|--------------|
| Python | ≥ 3.11 |
| `deepagents` | ≥ 0.7 (tested against 0.7.4) |
| `langchain` | ≥ 1.3 |
| `qdrant-client` | ≥ 1.18 (`query_points`) |
| Orchestrator model | `anthropic:claude-sonnet-4-6` |
| Worker model | `anthropic:claude-haiku-4-5` |

## Next reading

- Design article / implementation spec: [`ARTICLE.md`](ARTICLE.md)
- Package overview and API surface: [`../README.md`](../README.md)
- Deep Agents docs: [overview](https://docs.langchain.com/oss/python/deepagents/overview), [subagents](https://docs.langchain.com/oss/python/deepagents/subagents), [customization](https://docs.langchain.com/oss/python/deepagents/customization)
