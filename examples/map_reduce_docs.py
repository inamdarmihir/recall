"""End-to-end demo: map-reduce doc summarization with Recall cache.

Models verified against Deep Agents / Anthropic catalogs (July 2026):
- Orchestrator: anthropic:claude-sonnet-5  (latest Sonnet; Anthropic API id `claude-sonnet-5`)
- Worker:       anthropic:claude-haiku-4-5 (fast subagent tier)

Deep Agents docs also list `anthropic:claude-opus-4-8` among suggested eval models.
Override via RECALL_ORCHESTRATOR_MODEL / RECALL_WORKER_MODEL.

Requires:
  export ANTHROPIC_API_KEY=...
  # optional: docker run -p 6333:6333 qdrant/qdrant
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from deepagents import create_deep_agent

from recall import SubagentResultCacheMiddleware

ROOT = Path(__file__).resolve().parent
DOCS = ROOT / "sample_docs"

DOC_SUMMARIZER_PROMPT = """You are a documentation summarizer.
Read the provided page and return:
1) a 2-3 sentence structural summary
2) any TODO / FIXME markers
Be concise. Do not invent content that is not in the page.
"""

ORCHESTRATOR_PROMPT = """You review a documentation corpus by dispatching the
doc-summarizer subagent once per markdown page, then synthesizing a short brief.
Prefer the task tool with subagent_type='doc-summarizer'. Include the page path
in both `files` and backtick-quoted in the description.
"""


def build_agent(qdrant_url: str, vfs_root: str):
    orchestrator_model = os.getenv("RECALL_ORCHESTRATOR_MODEL", "anthropic:claude-sonnet-5")
    worker_model = os.getenv("RECALL_WORKER_MODEL", "anthropic:claude-haiku-4-5")

    cache_middleware = SubagentResultCacheMiddleware(
        qdrant_url=qdrant_url,
        profiles={
            "doc-summarizer": {
                "system_prompt": DOC_SUMMARIZER_PROMPT,
                "model": worker_model,
                "tools_version": "a3f9c2d1",
                "pure": True,
            },
            "web-researcher": {
                "system_prompt": "Live web research — never cache.",
                "model": orchestrator_model,
                "tools_version": "a3f9c2d1",
                "pure": False,
            },
        },
        exact_only=True,
        vfs_root=vfs_root,
    )

    agent = create_deep_agent(
        model=orchestrator_model,
        system_prompt=ORCHESTRATOR_PROMPT,
        subagents=[
            {
                "name": "doc-summarizer",
                "description": "Summarize a single documentation page for structural changes and TODOs.",
                "system_prompt": DOC_SUMMARIZER_PROMPT,
                "model": worker_model,
            }
        ],
        middleware=[cache_middleware],
    )
    return agent, cache_middleware


def main() -> None:
    parser = argparse.ArgumentParser(description="Recall map-reduce docs demo")
    parser.add_argument(
        "--qdrant-url",
        default=os.getenv("QDRANT_URL", "http://localhost:6333"),
        help="Qdrant HTTP URL (default: http://localhost:6333)",
    )
    parser.add_argument(
        "--docs",
        default=str(DOCS),
        help="Directory of markdown pages to review",
    )
    args = parser.parse_args()

    if not os.getenv("ANTHROPIC_API_KEY"):
        raise SystemExit(
            "ANTHROPIC_API_KEY is required to run the live demo. Unit tests do not need an API key."
        )

    docs_dir = Path(args.docs)
    pages = sorted(docs_dir.glob("**/*.md"))
    if not pages:
        raise SystemExit(f"No markdown pages found under {docs_dir}")

    agent, cache = build_agent(args.qdrant_url, str(docs_dir))
    page_list = "\n".join(f"- {p.relative_to(docs_dir)}" for p in pages)
    prompt = (
        "Review these documentation pages with the doc-summarizer subagent "
        f"(one task call per page), then synthesize a brief:\n{page_list}"
    )

    print(
        f"Orchestrator model: {os.getenv('RECALL_ORCHESTRATOR_MODEL', 'anthropic:claude-sonnet-5')}"
    )
    print(f"Worker model:       {os.getenv('RECALL_WORKER_MODEL', 'anthropic:claude-haiku-4-5')}")
    print(f"Pages: {len(pages)} under {docs_dir}")
    print("--- first pass ---")
    result = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
    final = result["messages"][-1].content
    print(final)
    print("cache stats:", cache.stats)

    print("--- second pass (expect cache hits on unchanged pages) ---")
    result2 = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
    final2 = result2["messages"][-1].content
    print(final2)
    print("cache stats:", cache.stats)


if __name__ == "__main__":
    main()
