# Recall

**Content-addressed result cache for [Deep Agents](https://docs.langchain.com/oss/python/deepagents) subagent dispatch.**

Recall is an `AgentMiddleware` that memoizes pure `task` subagent calls in [Qdrant](https://qdrant.tech). Identical inputs return the cached result instead of re-running the worker — cutting redundant API spend on map-reduce and recurring review workloads.

This repository implements the design in [`docs/ARTICLE.md`](docs/ARTICLE.md).

---

## Why it exists

Deep Agents isolate work into subagents to avoid context rot. That isolation multiplies cost: every worker re-derives results even when inputs have not changed. Provider prompt caching discounts system-prompt prefixes; it does **not** skip output tokens for repeated subagent results.

Recall closes that gap with result-level memoization above the API:

1. Hash the subagent profile + canonical task input (including file content hashes).
2. Look up an exact match in Qdrant.
3. Verify provenance; on hit, return a `ToolMessage` without invoking the worker.
4. On miss, run the subagent and store the result.

Semantic matching exists as an opt-in; the default is **exact-only** (`exact_only=True`).

## Features

- Drop-in middleware for `create_deep_agent(..., middleware=[...])`
- Exact content-addressed keys (`SHA-256`) with file content folded into the key
- Qdrant store using payload indexes for exact lookup and `query_points` for optional ANN
- Provenance soft-invalidation (`stale=True`) plus bulk profile invalidation
- Pluggable embedders (`ZeroEmbedder` for exact-only, optional FastEmbed)
- Works with static `task` dispatch and dynamic REPL fan-out (same hook)

## Verified stack & models

| Component | Version / ID | Notes |
|-----------|--------------|--------|
| Python | ≥ 3.11 | Developed on 3.12 |
| `deepagents` | **0.7.0** | Latest at implementation time |
| `langchain` | ≥ 1.3 | `AgentMiddleware` / `ToolCallRequest` API |
| `qdrant-client` | ≥ **1.18** | Uses `query_points` (not deprecated `search`) |
| Orchestrator model | `anthropic:claude-sonnet-5` | Latest Sonnet (Anthropic API, mid-2026) |
| Worker model | `anthropic:claude-haiku-4-5` | Fast / cheap subagent tier |

Deep Agents model docs also list `anthropic:claude-opus-4-8`, `openai:gpt-5.5`, and `google_genai:gemini-3.6-flash` among suggested eval models. Override defaults with env vars in the example.

Middleware API notes (adapted from the article to match live LangChain types):

- Import: `from langchain.agents.middleware import AgentMiddleware`
- Tool identity: `request.tool_call["name"]` / `request.tool_call["args"]`
- Cache hits return `ToolMessage`, not a bare string
- Collection bootstrap: `abefore_agent`

## Install

```bash
# From source (editable)
pip install -e ".[dev]"

# Optional semantic embeddings
pip install -e ".[fastembed]"

# Live Anthropic demo
pip install -e ".[anthropic]"
```

Published name (when released): `pip install recall-agents`.

## Quick start

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
            "tools_version": "a3f9c2d1",
            "pure": True,
        },
        "web-researcher": {
            "system_prompt": "Live web research.",
            "model": "anthropic:claude-sonnet-5",
            "tools_version": "a3f9c2d1",
            "pure": False,  # never cached
        },
    },
    exact_only=True,
    vfs_root="./workspace",
)

agent = create_deep_agent(
    model="anthropic:claude-sonnet-5",
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
```

In-memory Qdrant (tests / local smoke):

```python
cache = SubagentResultCacheMiddleware(
    qdrant_location=":memory:",
    profiles={...},
)
```

## Package layout

```
src/recall/
  __init__.py       # public re-exports
  keys.py           # canonicalize(), compute_key(), purity helpers
  store.py          # Qdrant collection, lookup_exact / lookup_semantic, insert
  middleware.py     # SubagentResultCacheMiddleware (awrap_tool_call)
  invalidation.py   # verify_provenance(), invalidate_profile()
  embeddings.py     # ZeroEmbedder, FastEmbedEmbedder, HashEmbedder
examples/
  map_reduce_docs.py
  sample_docs/
docs/
  ARTICLE.md        # original design article
tests/
```

## End-to-end example

```bash
# Terminal A — Qdrant
docker run --rm -p 6333:6333 qdrant/qdrant

# Terminal B — demo (needs ANTHROPIC_API_KEY)
export ANTHROPIC_API_KEY=sk-...
pip install -e ".[dev,anthropic]"
python examples/map_reduce_docs.py
```

The demo fans out over `examples/sample_docs/*.md`, then runs a second pass so unchanged pages should become cache hits (`cache.stats`).

## Development

```bash
pip install -e ".[dev]"
ruff check src tests examples
ruff format --check src tests examples
pytest -q
```

Tests use an in-memory Qdrant client and do **not** call LLM providers.

## Safety notes

- Only profiles with `pure: True` are cached.
- Prefer `exact_only=True` in production. Semantic matching can return false hits when task strings are near-duplicates with different intent (see article §11).
- A false cache hit is worse than a miss — treat semantic mode as experimental and per-subagent.

## License

MIT
