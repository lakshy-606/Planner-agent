"""LangGraph wiring: clarify branch, parallel data fetch, draft → estimate → validate retry loop, fallback."""
from __future__ import annotations

from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from . import nodes, tools
from .models import PlannerState


def route_after_clarity(state: PlannerState) -> str:
    return "ask_clarification" if state.get("clarifying_questions") else "resolve_city"


def route_after_draft(state: PlannerState) -> str:
    return "fallback_plan" if state.get("llm_failed") else "estimate_travel_and_cost"


def route_after_validate(state: PlannerState) -> str:
    if not tools.errors(state.get("issues", [])):
        return "finalize_plan"
    if state.get("retries", 0) >= nodes.MAX_RETRIES:
        return "fallback_plan"
    return "draft_plan"


@lru_cache(maxsize=1)
def build_graph():
    g = StateGraph(PlannerState)
    g.add_node("parse_preferences", nodes.parse_preferences)
    g.add_node("check_clarity", nodes.check_clarity)
    g.add_node("ask_clarification", nodes.ask_clarification)
    g.add_node("resolve_city", nodes.resolve_city)
    g.add_node("get_activity_options", nodes.get_activity_options)
    g.add_node("get_food_options", nodes.get_food_options)
    g.add_node("enrich_and_filter", nodes.enrich_and_filter)
    g.add_node("draft_plan", nodes.draft_plan)
    g.add_node("estimate_travel_and_cost", nodes.estimate_travel_and_cost)
    g.add_node("validate_plan", nodes.validate_plan)
    g.add_node("fallback_plan", nodes.fallback_plan)
    g.add_node("finalize_plan", nodes.finalize_plan)

    g.add_edge(START, "parse_preferences")
    g.add_edge("parse_preferences", "check_clarity")
    g.add_conditional_edges("check_clarity", route_after_clarity, ["ask_clarification", "resolve_city"])
    g.add_edge("ask_clarification", END)
    # parallel fan-out / fan-in
    g.add_edge("resolve_city", "get_activity_options")
    g.add_edge("resolve_city", "get_food_options")
    g.add_edge(["get_activity_options", "get_food_options"], "enrich_and_filter")
    g.add_edge("enrich_and_filter", "draft_plan")
    g.add_conditional_edges("draft_plan", route_after_draft, ["fallback_plan", "estimate_travel_and_cost"])
    g.add_edge("estimate_travel_and_cost", "validate_plan")
    g.add_conditional_edges("validate_plan", route_after_validate, ["finalize_plan", "fallback_plan", "draft_plan"])
    g.add_edge("fallback_plan", "finalize_plan")
    g.add_edge("finalize_plan", END)
    return g.compile()


def run(raw_input, is_followup: bool = False) -> dict:
    """Convenience wrapper for tests / CLI."""
    return build_graph().invoke({"raw_input": raw_input, "is_followup": is_followup, "trace": []})
