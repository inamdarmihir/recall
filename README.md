<div align="center">

# Recall

**A working content-addressed cache for Deep Agents subagent dispatch.**

![Python](https://img.shields.io/badge/python-3.11%2B-3572A5?style=flat-square)
![Status](https://img.shields.io/badge/status-design%20spec-orange?style=flat-square)
[![Stars](https://img.shields.io/github/stars/inamdarmihir/recall?style=flat-square&color=FB6A76)](https://github.com/inamdarmihir/recall/stargazers)

**[Design article](https://aihive.hashnode.dev/recall-a-working-content-addressed-cache-for-deep-agents-subagent-dispatch)** · **[License](#license)**

</div>

---

> **Status: design specification.** This repository currently contains the full design write-up (`docs/ARTICLE.md`) that this README summarizes. The package layout, APIs, and install commands below describe the proposed implementation and have not shipped as a published package yet. Treat code snippets here as the spec to build against, not as an installed dependency.

Multi-agent frameworks solve context rot by isolating work into subagents — each with its own context window, its own tool access, its own billing. The isolation is the point. But it comes with a structural cost nobody addresses by default: every subagent re-derives results from scratch, even when the underlying inputs haven't changed. Recall is a content-addressed result cache for Deep Agents dispatch, packaged as an `AgentMiddleware` backed by Qdrant.

It targets the **static-dispatch** path (the `task` tool in `SubAgentMiddleware`) and the **dynamic-dispatch** path (programmatic fan-out via `CodeInterpreterMiddleware`). It does not cache the model itself — provider-side prompt caching already handles prefix deduplication at the API layer. What's missing is result-level memoization above the API, which is what this library provides.

---

## Why Recall

A representative fan-out session — an orchestrator planning and synthesizing, three workers summarizing modules — pays roughly four API call budgets for what could be one sequential pass. Provider-side prompt caching discounts the shared system-prompt prefix, but does nothing for the output tokens of a redundant call, and it's stateless across sessions: an invocation on Monday and the same invocation on Thursday both pay full cost.

Recall closes that gap with a **content-addressed cache**: it stores a subagent invocation's output keyed by a deterministic hash of `(subagent type, shared state, task input)`. For subagents an operator flags `pure`, identical inputs return the cached result instead of re-executing — with provenance tracked so results invalidate when what they read changes.

## Package layout

```
recall/
  __init__.py          # public re-exports
  keys.py              # canonicalize(), compute_key(), purity classification
  store.py             # Qdrant collection setup, lookup_exact/lookup_semantic, insert
  middleware.py        # SubagentResultCacheMiddleware (wrap_tool_call hook)
  invalidation.py       # provenance verification, staleness propagation
docs/ARTICLE.md         # original design article / implementation spec
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
   5. hit + valid  ──▶ return cached result
      miss/invalid ──▶ handler(req) ──▶ store.insert(...)
```

## Requirements

- Python 3.11+
- [`qdrant-client>=1.18`](https://pypi.org/project/qdrant-client/) (uses `query_points`, not deprecated `.search()`)
- [`deepagents`](https://github.com/langchain-ai/deepagents) middleware protocol
- Optional: local embeddings (`fastembed`) for semantic-match mode

## Install

```bash
pip install recall-agents
# optional: bundled local embeddings for semantic-match mode
pip install "recall-agents[fastembed]"
```

## Quick start

```python
from deepagents import create_deep_agent
from recall import SubagentResultCacheMiddleware

cache_middleware = SubagentResultCacheMiddleware(
    qdrant_url="http://localhost:6333",
    profiles={
        "doc-summarizer": {
            "system_prompt": DOC_SUMMARIZER_PROMPT,
            "model": "anthropic:claude-haiku-4-5",
            "tools_version": "a3f9c2d1",
            "pure": True,
        },
        "web-researcher": {
            "system_prompt": WEB_RESEARCHER_PROMPT,
            "model": "anthropic:claude-sonnet-4-6",
            "tools_version": "a3f9c2d1",
            "pure": False,  # reads live web state; never cached
        },
    },
    exact_only=True,   # start conservative; see the article before enabling semantic mode
    vfs_root="./workspace",
)

agent = create_deep_agent(
    tools=[...],
    subagents=[...],
    middleware=[cache_middleware],
)
```

## Design notes

- **Purity is opt-in.** Only subagents an operator explicitly flags `pure` are eligible for caching; everything else always executes.
- **Provenance-based invalidation.** Cached results carry what they read, so changed inputs invalidate the entry instead of silently going stale.
- **Exact match first.** `exact_only=True` is the safe default; semantic (similarity-based) cache hits are a deliberate, riskier opt-in — see the false-hit-risk discussion in the article.

Full write-up — the cost model, the Qdrant collection schema, the false-hit-risk analysis, and open problems: [`docs/ARTICLE.md`](docs/ARTICLE.md).

## License

Not yet specified. A `LICENSE` file will be added before this is recommended for external use.

---

<div align="center"><sub>Part of the <a href="https://aihive.hashnode.dev">AIHive</a> series — <a href="https://aihive.hashnode.dev/recall-a-working-content-addressed-cache-for-deep-agents-subagent-dispatch">read the design article</a></sub></div>
