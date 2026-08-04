"""Recall — content-addressed cache for Deep Agents subagent dispatch.

Attach :class:`SubagentResultCacheMiddleware` to ``create_deep_agent`` to
memoize pure ``task`` subagent results in Qdrant. Identical inputs return
the cached result instead of re-running the worker.

Quick start::

    from deepagents import create_deep_agent
    from recall import SubagentResultCacheMiddleware

    cache = SubagentResultCacheMiddleware(
        qdrant_url="http://localhost:6333",
        profiles={
            "doc-summarizer": {
                "system_prompt": "Summarize the page.",
                "model": "anthropic:claude-haiku-4-5",
                "tools_version": "a3f9c2d1",
                "pure": True,
            },
        },
        exact_only=True,
        vfs_root="./docs",
    )

    agent = create_deep_agent(
        model="anthropic:claude-sonnet-4-6",
        subagents=[{
            "name": "doc-summarizer",
            "description": "Summarize a documentation page.",
            "system_prompt": "Summarize the page.",
            "model": "anthropic:claude-haiku-4-5",
        }],
        middleware=[cache],
    )

Full setup guide: ``docs/SETUP.md``. Design article: ``docs/ARTICLE.md``.
"""

from recall.embeddings import Embedder, FastEmbedEmbedder, HashEmbedder, ZeroEmbedder
from recall.invalidation import invalidate_profile, verify_provenance
from recall.keys import (
    canonicalize,
    compute_key,
    extract_source_refs,
    normalize_path,
    profile_prefix,
)
from recall.middleware import SubagentResultCacheMiddleware
from recall.store import EMBED_DIM, SubagentCacheStore, ensure_collection
from recall.types import CacheHit, CacheProfile

__all__ = [
    "EMBED_DIM",
    "CacheHit",
    "CacheProfile",
    "Embedder",
    "FastEmbedEmbedder",
    "HashEmbedder",
    "SubagentCacheStore",
    "SubagentResultCacheMiddleware",
    "ZeroEmbedder",
    "canonicalize",
    "compute_key",
    "ensure_collection",
    "extract_source_refs",
    "invalidate_profile",
    "normalize_path",
    "profile_prefix",
    "verify_provenance",
]

__version__ = "0.1.0"
