"""Agno + LangGraph + deepagents agent integration for recall cache management."""
from __future__ import annotations
from typing import Any


def build_agno_cache_agent(store, memory=None, model: str = "gpt-4o"):
    """Agno Agent that manages the recall cache and provides cost analytics."""
    from agno.agent import Agent
    from agno.models.openai import OpenAIChat

    def lookup(subagent_type: str, system_prompt: str, model_name: str, tools_version: str, task_input: str) -> dict[str, Any]:
        """Look up a cached result for a subagent dispatch."""
        from recall.keys import compute_key
        import json
        try:
            parsed_input = json.loads(task_input)
        except Exception:
            parsed_input = task_input
        key = compute_key(subagent_type, system_prompt, model_name, tools_version, parsed_input)
        result = store.lookup_exact(key)
        hit = result is not None
        if memory:
            from recall.memory import record_cache_event
            record_cache_event(memory, subagent_type, hit, model_name)
        return {"hit": hit, "result": result}

    def get_hit_rate(subagent_type: str) -> dict[str, Any]:
        """Get cache hit rate stats for a subagent type from mem0."""
        if not memory:
            return {"subagent_type": subagent_type, "stats": []}
        from recall.memory import query_cache_stats
        events = query_cache_stats(memory, subagent_type)
        hits = sum(1 for e in events if e.get("metadata", {}).get("hit"))
        return {"subagent_type": subagent_type, "total": len(events), "hits": hits, "rate": hits / max(len(events), 1)}

    agent = Agent(
        model=OpenAIChat(id=model),
        name="RecallCacheAgent",
        description="Manages a content-addressed result cache for Deep Agents subagent dispatch.",
        instructions=[
            "You manage the Recall cache for subagent result memoization.",
            "Use lookup() to check for cached results before dispatching subagents.",
            "Use get_hit_rate() to report on cache efficiency and cost savings.",
            "Only cache PURE subagents — those with deterministic outputs given the same inputs.",
        ],
        tools=[lookup, get_hit_rate],
        show_tool_calls=True,
        markdown=True,
    )
    return agent


def build_langgraph_cache_pipeline(store, middleware, memory=None):
    """LangGraph pipeline: classify purity → check cache → invoke → store."""
    from typing import TypedDict
    from langgraph.graph import StateGraph, END

    class CacheState(TypedDict):
        subagent_type: str
        system_prompt: str
        model: str
        tools_version: str
        task_input: Any
        purity: str
        cache_key: str
        cache_hit: bool
        result: Any

    def classify_node(state: CacheState) -> CacheState:
        from recall.keys import classify_purity, SubagentPurity
        purity = classify_purity(state["subagent_type"], {})
        state["purity"] = purity.value
        return state

    def check_cache_node(state: CacheState) -> CacheState:
        from recall.keys import compute_key
        key = compute_key(state["subagent_type"], state["system_prompt"],
                          state["model"], state["tools_version"], state["task_input"])
        state["cache_key"] = key
        cached = store.lookup_exact(key)
        state["cache_hit"] = cached is not None
        if cached:
            state["result"] = cached["result"]
        if memory:
            from recall.memory import record_cache_event
            record_cache_event(memory, state["subagent_type"], state["cache_hit"], state["model"])
        return state

    def should_use_cache(state: CacheState) -> str:
        from recall.keys import SubagentPurity
        if state["purity"] != SubagentPurity.PURE.value:
            return "invoke"
        return "cached" if state["cache_hit"] else "invoke"

    def cached_node(state: CacheState) -> CacheState:
        return state  # result already populated

    def invoke_node(state: CacheState) -> CacheState:
        # Placeholder: caller fills in actual subagent invocation
        state["result"] = None
        return state

    def store_node(state: CacheState) -> CacheState:
        if not state["cache_hit"] and state["purity"] == "pure" and state["result"] is not None:
            store.insert(state["cache_key"], state["result"],
                         {"subagent_type": state["subagent_type"], "model": state["model"],
                          "tools_version": state["tools_version"]})
        return state

    graph = StateGraph(CacheState)
    graph.add_node("classify", classify_node)
    graph.add_node("check_cache", check_cache_node)
    graph.add_node("cached", cached_node)
    graph.add_node("invoke", invoke_node)
    graph.add_node("store", store_node)
    graph.set_entry_point("classify")
    graph.add_edge("classify", "check_cache")
    graph.add_conditional_edges("check_cache", should_use_cache, {"cached": "cached", "invoke": "invoke"})
    graph.add_edge("cached", END)
    graph.add_edge("invoke", "store")
    graph.add_edge("store", END)
    return graph.compile()
