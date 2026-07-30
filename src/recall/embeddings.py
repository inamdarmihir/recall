"""Pluggable embedding backends for optional semantic cache matching."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from recall.store import EMBED_DIM


class Embedder(ABC):
    """Minimal embedding interface used by SubagentResultCacheMiddleware."""

    dim: int = EMBED_DIM

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Return a dense embedding for ``text``."""


class ZeroEmbedder(Embedder):
    """Deterministic zero vectors for exact-only deployments (no ANN needed)."""

    def __init__(self, dim: int = EMBED_DIM) -> None:
        self.dim = dim

    async def embed(self, text: str) -> list[float]:
        _ = text
        return [0.0] * self.dim


class HashEmbedder(Embedder):
    """Lightweight local embedder for tests — not suitable for semantic production use."""

    def __init__(self, dim: int = EMBED_DIM) -> None:
        self.dim = dim

    async def embed(self, text: str) -> list[float]:
        # Spread character ordinals into a fixed-dim vector; stable across runs.
        vec = [0.0] * self.dim
        if not text:
            return vec
        for i, ch in enumerate(text.encode("utf-8")):
            vec[i % self.dim] += (ch / 255.0) * 0.01
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]


class FastEmbedEmbedder(Embedder):
    """Optional FastEmbed backend (install with ``pip install 'recall-agents[fastembed]'``)."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:  # pragma: no cover - exercised only when optional dep missing
            raise ImportError(
                "fastembed is required for FastEmbedEmbedder. "
                "Install with: pip install 'recall-agents[fastembed]'"
            ) from exc
        self._model = TextEmbedding(model_name=model_name)
        self.dim = EMBED_DIM

    async def embed(self, text: str) -> list[float]:
        vectors: Sequence[Sequence[float]] = list(self._model.embed([text]))
        return list(vectors[0])
