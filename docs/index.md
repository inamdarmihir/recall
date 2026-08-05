# recall

> Content-addressed result cache for Deep Agents — eliminate redundant LLM API calls across subagent dispatches.

__omp_shell("!! success "Break-even point"")
    recall pays for its Qdrant overhead after the **first cache hit** on a $0.02 task. At 100 hits/day it saves ~$60/month. At 1,000 hits/day: ~$600/month.

## What it does

- SHA-256 content-addressed cache keyed on `(subagent_type, system_prompt, model, tools_version, task_input)`
- Classifies subagents as PURE (safe to cache) or IMPURE (always pass through)
- Sync and async middleware variants for any agent framework
- Provenance verification and TTL-based staleness detection
- mem0 cache analytics (hit/miss rates, cost savings)
- Agno agent for cache management and reporting
- LangGraph pipeline: classify → check → invoke → store

## Purity classification

PURE subagents produce identical output for identical inputs. Examples: document summariser, code analyser, data extractor. These are safe to cache — recall can serve their results without calling the LLM.

IMPURE subagents depend on external state: web browser, code executor, time-sensitive data fetcher. These always pass through — caching them would return stale results.

See [Quick Start](quickstart.md) to add caching to your agent in under 10 lines.
