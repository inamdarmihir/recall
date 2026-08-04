"""Deep Agents / LangChain ``AgentMiddleware`` for content-addressed caching.

Interception is deliberately narrow: only ``task`` tool calls for profiles
marked ``pure=True`` are eligible. Everything else passes through with a
single dictionary lookup of overhead.

Compatible with Deep Agents ``create_deep_agent`` / LangChain
``AgentMiddleware`` (verified against ``deepagents>=0.7`` and
``langchain>=1.3``):

- Tool identity: ``request.tool_call["name"]`` / ``["args"]``
- Cache hits return a :class:`~langchain.messages.ToolMessage`
- Live ``task`` handlers often return a :class:`~langgraph.types.Command`
  (Deep Agents ``SubAgentMiddleware``); we extract the result text for
  storage and still return the original ``Command`` so state updates apply
- Collection bootstrap: ``abefore_agent``

Static ``task`` dispatch and dynamic REPL fan-out
(:class:`~deepagents.middleware` + QuickJS) share this same hook — no
extra integration is required for map-reduce style workflows.

See ``docs/ARTICLE.md`` §8 and ``docs/SETUP.md`` for end-user wiring.
"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from langchain.agents.middleware import AgentMiddleware
from langchain.messages import ToolMessage
from langchain.tools.tool_node import ToolCallRequest
from langgraph.types import Command
from qdrant_client import AsyncQdrantClient

from recall.embeddings import Embedder, ZeroEmbedder
from recall.invalidation import verify_provenance
from recall.keys import compute_key, extract_source_refs
from recall.store import EMBED_DIM, SubagentCacheStore, ensure_collection


def _message_content_to_str(content: Any) -> str:
    """Flatten ToolMessage / multimodal content blocks to a plain string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and "text" in block:
                parts.append(str(block["text"]))
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(content)


def _extract_result_text(result: ToolMessage | Command[Any]) -> str | None:
    """Pull the cacheable result string from a handler return value.

    Deep Agents' ``task`` tool returns a ``Command`` whose ``update``
    contains a ``ToolMessage`` with the subagent's final text. Plain
    ``ToolMessage`` returns (tests / custom tools) are also supported.
    Returns ``None`` when nothing cacheable can be extracted.
    """
    if isinstance(result, ToolMessage):
        return _message_content_to_str(result.content)

    update = result.update
    if not isinstance(update, Mapping):
        return None
    messages = update.get("messages")
    if not isinstance(messages, list):
        return None
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            return _message_content_to_str(msg.content)
        content = getattr(msg, "content", None)
        if content is not None:
            return _message_content_to_str(content)
    return None


class SubagentResultCacheMiddleware(AgentMiddleware):
    """Memoize pure Deep Agents ``task`` subagent results in Qdrant.

    Parameters
    ----------
    qdrant_url:
        HTTP URL of a running Qdrant instance (e.g. ``http://localhost:6333``).
        Ignored when ``qdrant_location`` or ``qdrant_path`` is set.
    profiles:
        Mapping of ``subagent_type`` → profile dict. Each profile should
        include ``system_prompt``, ``model``, ``tools_version``, and
        ``pure``. Only ``pure=True`` profiles are cached.
    collection:
        Qdrant collection name (default ``"recall"``).
    exact_only:
        When ``True`` (default), skip semantic ANN lookup. Keep this on
        until you have measured false-hit risk for your workload.
    similarity_threshold:
        Cosine threshold for opt-in semantic hits (default ``0.97``).
    vfs_root:
        Local directory used to resolve source file paths when hashing
        content into the cache key. Defaults to the process cwd.
    qdrant_path:
        On-disk Qdrant path (local embedded mode).
    qdrant_location:
        Pass ``":memory:"`` for tests / ephemeral smoke runs.
    embedder:
        :class:`~recall.embeddings.Embedder` instance. Defaults to
        :class:`~recall.embeddings.ZeroEmbedder` (exact-only safe).
    embed_dim:
        Named-vector dimension; must match the embedder and collection.

    Example
    -------
    >>> from deepagents import create_deep_agent
    >>> from recall import SubagentResultCacheMiddleware
    >>> cache = SubagentResultCacheMiddleware(
    ...     qdrant_url="http://localhost:6333",
    ...     profiles={
    ...         "doc-summarizer": {
    ...             "system_prompt": "Summarize the page.",
    ...             "model": "anthropic:claude-haiku-4-5",
    ...             "tools_version": "a3f9c2d1",
    ...             "pure": True,
    ...         },
    ...     },
    ...     vfs_root="./docs",
    ... )
    >>> agent = create_deep_agent(
    ...     model="anthropic:claude-sonnet-4-6",
    ...     subagents=[{
    ...         "name": "doc-summarizer",
    ...         "description": "Summarize a documentation page.",
    ...         "system_prompt": "Summarize the page.",
    ...         "model": "anthropic:claude-haiku-4-5",
    ...     }],
    ...     middleware=[cache],
    ... )
    """

    # Distinct from Deep Agents built-in middleware names so v0.7's
    # "same .name replaces default" merge leaves defaults intact.
    name = "SubagentResultCacheMiddleware"

    def __init__(
        self,
        qdrant_url: str | None = None,
        profiles: dict[str, dict[str, Any]] | None = None,
        collection: str = "recall",
        exact_only: bool = True,
        similarity_threshold: float = 0.97,
        vfs_root: str | None = None,
        *,
        qdrant_path: str | None = None,
        qdrant_location: str | None = None,
        embedder: Embedder | None = None,
        embed_dim: int = EMBED_DIM,
    ) -> None:
        super().__init__()
        if profiles is None:
            raise ValueError(
                "profiles is required — pass a dict of subagent_type → "
                "{system_prompt, model, tools_version, pure}."
            )
        self._url = qdrant_url
        self._path = qdrant_path
        self._location = qdrant_location
        self._profiles = profiles
        self._collection_name = collection
        self._exact_only = exact_only
        self._threshold = similarity_threshold
        self._vfs_root = vfs_root
        self._embed_dim = embed_dim
        self._embedder: Embedder = embedder or ZeroEmbedder(dim=embed_dim)
        self._store: SubagentCacheStore | None = None
        self._stats: dict[str, int] = {"hits": 0, "misses": 0, "bypassed": 0}

    @property
    def stats(self) -> dict[str, int]:
        """Return a copy of hit / miss / bypass counters for instrumentation."""
        return dict(self._stats)

    def _make_client(self) -> AsyncQdrantClient:
        if self._location is not None:
            return AsyncQdrantClient(location=self._location)
        if self._path is not None:
            return AsyncQdrantClient(path=self._path)
        if self._url is not None:
            return AsyncQdrantClient(url=self._url)
        return AsyncQdrantClient(location=":memory:")

    async def _ensure_store(self) -> SubagentCacheStore:
        if self._store is None:
            client = self._make_client()
            await ensure_collection(
                client,
                self._collection_name,
                embed_dim=self._embed_dim,
            )
            self._store = SubagentCacheStore(client, self._collection_name)
        return self._store

    async def abefore_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        """Open the Qdrant client and ensure the collection exists (async path)."""
        _ = state, runtime
        await self._ensure_store()
        return None

    def before_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        """Sync bootstrap — prefer ``abefore_agent`` on the async Deep Agents path."""
        _ = state, runtime
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self._ensure_store())
        # When a loop is already running, store init is deferred to the first
        # awrap_tool_call via _ensure_store() lazy init.
        return None

    def _is_pure(self, subagent_type: str) -> bool:
        profile = self._profiles.get(subagent_type)
        return bool(profile and profile.get("pure", False))

    def _content_hash(self, path: str) -> str:
        root = Path(self._vfs_root) if self._vfs_root else Path(".")
        target = root / path.lstrip("/")
        if not target.is_file():
            return "sha256:missing"
        return "sha256:" + hashlib.sha256(target.read_bytes()).hexdigest()

    async def _embed(self, text: str) -> list[float]:
        return await self._embedder.embed(text)

    def _cached_tool_message(self, request: ToolCallRequest, result: str) -> ToolMessage:
        return ToolMessage(
            content=result,
            tool_call_id=request.tool_call["id"],
            name=request.tool_call.get("name", "task"),
            additional_kwargs={"recall_cache_hit": True},
        )

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """Intercept ``task`` calls: serve cache hits or run + store on miss."""
        tool_name = request.tool_call.get("name")
        if tool_name != "task":
            self._stats["bypassed"] += 1
            return await handler(request)

        args = dict(request.tool_call.get("args") or {})
        subagent_type = args.get("subagent_type", "default")
        if not self._is_pure(subagent_type):
            self._stats["bypassed"] += 1
            return await handler(request)

        store = await self._ensure_store()
        profile = self._profiles[subagent_type]
        key = compute_key(subagent_type, profile, args, self._content_hash)
        cached = await store.lookup_exact(key, subagent_type)
        if cached is not None and await verify_provenance(store, self._content_hash, cached):
            self._stats["hits"] += 1
            return self._cached_tool_message(request, cached["result"])

        if not self._exact_only:
            embed_vec = await self._embed(str(args.get("description", "")))
            semantic = await store.lookup_semantic(
                embed_vec,
                subagent_type,
                threshold=self._threshold,
            )
            if semantic is not None:
                self._stats["hits"] += 1
                return self._cached_tool_message(request, semantic)

        result = await handler(request)
        self._stats["misses"] += 1

        result_text = _extract_result_text(result)
        if result_text is None:
            # Nothing we can safely memoize (unexpected return shape).
            return result

        source_refs = extract_source_refs(args)
        await store.insert(
            point_id=str(uuid4()),
            exact_hash=key,
            subagent_type=subagent_type,
            model=profile.get("model", "default"),
            tools_version=profile.get("tools_version", "unset"),
            source_refs=source_refs,
            content_hashes={ref: self._content_hash(ref) for ref in source_refs},
            task_embed=await self._embed(str(args.get("description", ""))),
            result_embed=await self._embed(result_text[:512]),
            result=result_text,
            created_at=datetime.now(UTC).isoformat(),
        )
        return result

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """Sync wrapper — bridges to :meth:`awrap_tool_call`."""

        async def _async_handler(req: ToolCallRequest) -> ToolMessage | Command[Any]:
            return handler(req)

        def _run() -> ToolMessage | Command[Any]:
            return asyncio.run(self.awrap_tool_call(request, _async_handler))

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return _run()

        # Sync tool path while an event loop is already running (uncommon for Deep Agents).
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(_run).result()
