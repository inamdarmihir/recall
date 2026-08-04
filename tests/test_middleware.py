"""Middleware integration tests (no live LLM required)."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest
from langchain.messages import ToolMessage
from langchain.tools.tool_node import ToolCallRequest
from langgraph.types import Command

from recall.embeddings import ZeroEmbedder
from recall.middleware import SubagentResultCacheMiddleware

DOC_PROMPT = "Summarize structural changes and TODOs."


def _request(
    subagent_type: str, description: str, files: list[str] | None = None
) -> ToolCallRequest:
    args: dict[str, Any] = {
        "subagent_type": subagent_type,
        "description": description,
    }
    if files is not None:
        args["files"] = files
    return ToolCallRequest(
        tool_call={
            "name": "task",
            "args": args,
            "id": f"call_{uuid4().hex[:8]}",
            "type": "tool_call",
        },
        tool=None,
        state={},
        runtime=None,  # type: ignore[arg-type]
    )


@pytest.fixture
def middleware(tmp_path) -> SubagentResultCacheMiddleware:
    page = tmp_path / "page.md"
    page.write_text("# Stable page\n", encoding="utf-8")
    return SubagentResultCacheMiddleware(
        qdrant_location=":memory:",
        profiles={
            "doc-summarizer": {
                "system_prompt": DOC_PROMPT,
                "model": "anthropic:claude-haiku-4-5",
                "tools_version": "a3f9c2d1",
                "pure": True,
            },
            "web-researcher": {
                "system_prompt": "Search the web.",
                "model": "anthropic:claude-sonnet-4-6",
                "tools_version": "a3f9c2d1",
                "pure": False,
            },
        },
        exact_only=True,
        vfs_root=str(tmp_path),
        embedder=ZeroEmbedder(dim=384),
    )


@pytest.mark.asyncio
async def test_cache_hit_on_second_identical_task(
    middleware: SubagentResultCacheMiddleware,
) -> None:
    calls = {"n": 0}

    async def handler(request: ToolCallRequest) -> ToolMessage:
        calls["n"] += 1
        return ToolMessage(
            content=f"summary-{calls['n']}",
            tool_call_id=request.tool_call["id"],
            name="task",
        )

    req1 = _request("doc-summarizer", "Summarize `page.md`", ["page.md"])
    out1 = await middleware.awrap_tool_call(req1, handler)
    assert isinstance(out1, ToolMessage)
    assert out1.content == "summary-1"
    assert calls["n"] == 1

    req2 = _request("doc-summarizer", "Summarize `page.md`", ["page.md"])
    out2 = await middleware.awrap_tool_call(req2, handler)
    assert isinstance(out2, ToolMessage)
    assert out2.content == "summary-1"
    assert out2.additional_kwargs.get("recall_cache_hit") is True
    assert calls["n"] == 1
    assert middleware.stats["hits"] == 1
    assert middleware.stats["misses"] == 1


@pytest.mark.asyncio
async def test_caches_command_return_from_deep_agents_task(
    middleware: SubagentResultCacheMiddleware,
) -> None:
    """Deep Agents SubAgentMiddleware returns Command, not bare ToolMessage."""
    calls = {"n": 0}

    async def handler(request: ToolCallRequest) -> Command:
        calls["n"] += 1
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"command-summary-{calls['n']}",
                        tool_call_id=request.tool_call["id"],
                        name="task",
                    )
                ]
            }
        )

    req1 = _request("doc-summarizer", "Summarize `page.md`", ["page.md"])
    out1 = await middleware.awrap_tool_call(req1, handler)
    assert isinstance(out1, Command)
    assert calls["n"] == 1

    req2 = _request("doc-summarizer", "Summarize `page.md`", ["page.md"])
    out2 = await middleware.awrap_tool_call(req2, handler)
    assert isinstance(out2, ToolMessage)
    assert out2.content == "command-summary-1"
    assert out2.additional_kwargs.get("recall_cache_hit") is True
    assert calls["n"] == 1
    assert middleware.stats["hits"] == 1


@pytest.mark.asyncio
async def test_non_pure_bypasses_cache(middleware: SubagentResultCacheMiddleware) -> None:
    calls = {"n": 0}

    async def handler(request: ToolCallRequest) -> ToolMessage:
        calls["n"] += 1
        return ToolMessage(
            content=f"live-{calls['n']}",
            tool_call_id=request.tool_call["id"],
            name="task",
        )

    for _ in range(2):
        req = _request("web-researcher", "Find latest news")
        out = await middleware.awrap_tool_call(req, handler)
        assert isinstance(out, ToolMessage)

    assert calls["n"] == 2
    assert middleware.stats["bypassed"] == 2


@pytest.mark.asyncio
async def test_file_edit_causes_miss(
    middleware: SubagentResultCacheMiddleware,
    tmp_path,
) -> None:
    calls = {"n": 0}

    async def handler(request: ToolCallRequest) -> ToolMessage:
        calls["n"] += 1
        return ToolMessage(
            content=f"summary-{calls['n']}",
            tool_call_id=request.tool_call["id"],
            name="task",
        )

    req = _request("doc-summarizer", "Summarize `page.md`", ["page.md"])
    await middleware.awrap_tool_call(req, handler)
    (tmp_path / "page.md").write_text("# Edited page\n", encoding="utf-8")
    out = await middleware.awrap_tool_call(req, handler)
    assert isinstance(out, ToolMessage)
    assert out.content == "summary-2"
    assert calls["n"] == 2
