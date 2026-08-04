<div align="center">

# Recall

**A working content-addressed cache for Deep Agents subagent dispatch.**

![Python](https://img.shields.io/badge/python-3.11%2B-3572A5?style=flat-square)
![Status](https://img.shields.io/badge/status-implemented-2ea44f?style=flat-square)
[![Stars](https://img.shields.io/github/stars/inamdarmihir/recall?style=flat-square&color=FB6A76)](https://github.com/inamdarmihir/recall/stargazers)

**[Setup guide](docs/SETUP.md)** · **[Design article](docs/ARTICLE.md)** · **[License](#license)**

</div>

---

Multi-agent frameworks solve context rot by isolating work into subagents — each with its own context window, its own tool access, its own billing. The isolation is the point. But it comes with a structural cost nobody addresses by default: every subagent re-derives results from scratch, even when the underlying inputs haven't changed. **Recall** is a content-addressed result cache for Deep Agents dispatch, packaged as an `AgentMiddleware` backed by Qdrant.

It targets the **static-dispatch** path (the `task` tool in `SubAgentMiddleware`) and the **dynamic-dispatch** path (programmatic fan-out via `CodeInterpreterMiddleware`). It does not cache the model itself — provider-side prompt caching already handles prefix deduplication at the API layer. What's missing is result-level memoization above the API, which is what this library provides.

## Why Recall

A representative fan-out session — an orchestrator planning and synthesizing, three workers summarizing modules — pays roughly four API call budgets for what could be one sequential pass. Provider-side prompt caching discounts the shared system-prompt prefix, but does nothing for the output tokens of a redundant call, and it's stateless across sessions.

Recall closes that gap with a **content-addressed cache**: it stores a subagent invocation's output keyed by a deterministic hash of `(subagent type, shared state, task input)`. For subagents an operator flags `pure`, identical inputs return the cached result instead of re-executing — with provenance tracked so results invalidate when what they read changes.

## Package layout

```
src/recall/
  __init__.py          # public re-exports
  keys.py              # canonicalize(), compute_key(), purity helpers
  store.py             # Qdrant collection, lookup_exact / lookup_semantic, insert
  middleware.py        # SubagentResultCacheMiddleware (awrap_tool_call)
  invalidation.py      # provenance verification, staleness propagation
  embeddings.py        # ZeroEmbedder, FastEmbedEmbedder, HashEmbedder
  types.py             # CacheProfile / CacheHit TypedDicts
docs/
  SETUP.md             # end-user install & wiring guide
  ARTICLE.md           # original design article / implementation spec
examples/
  map_reduce_docs.py   # live Deep Agents demo
  sample_docs/         # small markdown corpus for the demo
tests/                 # in-memory Qdrant tests (no API key)
```

```
Orchestrator
   │ task(subagent_type, description, files=[...])
   ▼
SubagentResultCacheMiddleware.awrap_tool_call  (middleware.py)
   1. is_pure(subagent_type)?           ── no ──▶ handler(req)
   2. compute_key(args)                          (keys.py)
   3. lookup_exact(key)                          (store.py)
   4. verify_provenance(hit)                     (invalidation)
   5. hit + valid  ──▶ return cached ToolMessage
      miss/invalid ──▶ handler(req) ──▶ store.insert(...)
```

## Requirements

- Python 3.11+
- [`deepagents>=0.7`](https://pypi.org/project/deepagents/) (LangChain `AgentMiddleware` protocol)
- [`qdrant-client>=1.18`](https://pypi.org/project/qdrant-client/) (uses `query_points`, not deprecated `.search()`)
- Optional: local embeddings (`fastembed`) for semantic-match mode

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

Step-by-step walkthrough (venv, Qdrant, wiring, demo): **[docs/SETUP.md](docs/SETUP.md)**.

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
            "model": "anthropic:claude-sonnet-4-6",
            "tools_version": "a3f9c2d1",
            "pure": False,  # reads live web state; never cached
        },
    },
    exact_only=True,   # start conservative; see ARTICLE.md §11 before semantic mode
    vfs_root="./workspace",
)

agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-6",
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

## Verified stack

| Component | Version / ID | Notes |
|-----------|--------------|--------|
| Python | ≥ 3.11 | Developed on 3.12 |
| `deepagents` | ≥ **0.7** | Tested against 0.7.4 |
| `langchain` | ≥ 1.3 | `AgentMiddleware` / `ToolCallRequest` |
| `qdrant-client` | ≥ **1.18** | `query_points` (not deprecated `search`) |
| Orchestrator | `anthropic:claude-sonnet-4-6` | Deep Agents harness profile |
| Worker | `anthropic:claude-haiku-4-5` | Fast / cheap subagent tier |

Middleware API notes (adapted from the article to match live LangChain / Deep Agents types):

- Import: `from langchain.agents.middleware import AgentMiddleware`
- Tool identity: `request.tool_call["name"]` / `request.tool_call["args"]`
- Cache hits return `ToolMessage`; live `task` often returns `Command` (we extract + store the text)
- Collection bootstrap: `abefore_agent`

## End-to-end example

```bash
# Terminal A — Qdrant
docker run --rm -p 6333:6333 qdrant/qdrant

# Terminal B — demo (needs ANTHROPIC_API_KEY)
export ANTHROPIC_API_KEY=sk-...
pip install -e ".[dev,anthropic]"
python examples/map_reduce_docs.py
```

## Design notes

- **Purity is opt-in.** Only subagents an operator explicitly flags `pure` are eligible for caching; everything else always executes.
- **Provenance-based invalidation.** Cached results carry what they read, so changed inputs invalidate the entry instead of silently going stale.
- **Exact match first.** `exact_only=True` is the safe default; semantic (similarity-based) cache hits are a deliberate, riskier opt-in — see the false-hit-risk discussion in the article.
- **Command-aware caching.** Deep Agents' `task` tool returns a `Command` with state updates; Recall extracts the result text for storage and still returns the original `Command` on miss.

Full write-up — the cost model, the Qdrant collection schema, the false-hit-risk analysis, and open problems: [`docs/ARTICLE.md`](docs/ARTICLE.md).

## Development

```bash
pip install -e ".[dev]"
ruff check src tests examples
pytest -q
mypy src/recall
```

Tests use an in-memory Qdrant client and do **not** call LLM providers.

## License

MIT — see [`LICENSE`](LICENSE).

---

<div align="center"><sub>Part of the <a href="https://aihive.hashnode.dev">AIHive</a> series — <a href="https://aihive.hashnode.dev/recall-a-working-content-addressed-cache-for-deep-agents-subagent-dispatch">read the design article</a></sub></div>
