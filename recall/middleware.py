"""SubagentResultCacheMiddleware — intercepts deepagents tool calls for cache lookup/store."""
from __future__ import annotations
from typing import Any, Callable, Awaitable
from recall.keys import compute_key, classify_purity, SubagentPurity
from recall.store import RecallStore


class SubagentResultCacheMiddleware:
    """
    Middleware for deepagents / LangGraph agents.
    Intercepts subagent tool calls; returns cached results for pure subagents.
    """

    def __init__(
        self,
        store: RecallStore,
        profiles: dict[str, dict],
        tools_version: str = "1.0.0",
    ) -> None:
        self._store = store
        self._profiles = profiles
        self._tools_version = tools_version

    async def awrap_tool_call(
        self,
        subagent_type: str,
        system_prompt: str,
        model: str,
        task_input: Any,
        invoke: Callable[[], Awaitable[Any]],
    ) -> Any:
        """Async middleware entry point. Pure subagents check cache first."""
        purity = classify_purity(subagent_type, self._profiles)
        if purity is not SubagentPurity.PURE:
            return await invoke()

        key = compute_key(subagent_type, system_prompt, model, self._tools_version, task_input)
        cached = self._store.lookup_exact(key)
        if cached:
            return cached["result"]

        result = await invoke()
        self._store.insert(
            cache_key=key,
            result=result,
            provenance={
                "subagent_type": subagent_type,
                "model": model,
                "tools_version": self._tools_version,
            },
        )
        return result

    def wrap_tool_call(
        self,
        subagent_type: str,
        system_prompt: str,
        model: str,
        task_input: Any,
        invoke: Callable[[], Any],
    ) -> Any:
        """Sync variant for non-async callers."""
        purity = classify_purity(subagent_type, self._profiles)
        if purity is not SubagentPurity.PURE:
            return invoke()

        key = compute_key(subagent_type, system_prompt, model, self._tools_version, task_input)
        cached = self._store.lookup_exact(key)
        if cached:
            return cached["result"]

        result = invoke()
        self._store.insert(
            cache_key=key,
            result=result,
            provenance={"subagent_type": subagent_type, "model": model},
        )
        return result
