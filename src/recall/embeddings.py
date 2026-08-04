"""Pluggable embedding backends for optional semantic cache matching.

Exact-only deployments (the recommended default) do not need a real
embedder — :class:`ZeroEmbedder` satisfies the named-vector schema in
Qdrant without running an ANN model. Enable semantic matching only after
reading ``docs/ARTICLE.md`` §11 on false-hit risk, then wire
:class:`FastEmbedEmbedder` (or your own :class:`Embedder` subclass).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from recall.store import EMBED_DIM


class Embedder(ABC):
    """Minimal embedding interface used by :class:`SubagentResultCacheMiddleware`.

    Implementations must return a dense vector of length ``self.dim``.
    The dimension must match the Qdrant collection's named-vector size
    (see :data:`recall.store.EMBED_DIM`).
    """

    dim: int = EMBED_DIM

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Return a dense embedding for ``text``."""


class ZeroEmbedder(Embedder):
    """Deterministic zero vectors for exact-only deployments.

    Exact lookups ignore vectors (``with_vectors=False``). Declaring named
    vectors at collection create-time still requires *some* vector on
    insert; zeros keep the schema valid without loading an embedding model.
    """

    def __init__(self, dim: int = EMBED_DIM) -> None:
        self.dim = dim

    async def embed(self, text: str) -> list[float]:
        _ = text
        return [0.0] * self.dim


class HashEmbedder(Embedder):
    """Lightweight local embedder for tests — not for semantic production use.

    Spreads character ordinals into a fixed-dim vector. Stable across runs,
    cheap, and good enough to exercise the ANN code path in unit tests.
    Do **not** use this for real semantic cache hits.
    """

    def __init__(self, dim: int = EMBED_DIM) -> None:
        self.dim = dim

    async def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        if not text:
            return vec
        for i, ch in enumerate(text.encode("utf-8")):
            vec[i % self.dim] += (ch / 255.0) * 0.01
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]


class FastEmbedEmbedder(Embedder):
    """Optional FastEmbed backend (``pip install 'recall-agents[fastembed]'``).

    Defaults to ``sentence-transformers/all-MiniLM-L6-v2`` (384-dim), which
    matches :data:`recall.store.EMBED_DIM`. If you switch models, pass a
    matching ``embed_dim`` into the middleware / ``ensure_collection``.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "fastembed is required for FastEmbedEmbedder. "
                "Install with: pip install 'recall-agents[fastembed]'"
            ) from exc
        self._model = TextEmbedding(model_name=model_name)
        self.dim = EMBED_DIM

    async def embed(self, text: str) -> list[float]:
        vectors: Sequence[Sequence[float]] = list(self._model.embed([text]))
        return list(vectors[0])
