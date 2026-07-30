"""Public API for recall — content-addressed cache for Deep Agents subagent dispatch."""

from recall.embeddings import Embedder, FastEmbedEmbedder, ZeroEmbedder
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

__all__ = [
    "EMBED_DIM",
    "Embedder",
    "FastEmbedEmbedder",
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
