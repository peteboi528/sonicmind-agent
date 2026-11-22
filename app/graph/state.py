from __future__ import annotations

from typing import Any, TypedDict

from app.models import AgentAnswer, AgentPlan, StreamEvent


class AgentState(TypedDict, total=False):
    user_id: str
    run_id: str
    thread_id: str
    dialogue_thread_id: str
    asset_id: str | None
    query: str
    turn_started_at: float
    turn_budget_limit: float | None
    image_turn: bool
    image_observation: dict[str, Any] | None
    image_session: dict[str, Any] | None
    image_vision_calls: int
    top_k: int
    history: list[dict[str, Any]]
    plan: AgentPlan
    context: dict[str, Any]
    results: list[dict[str, Any]]
    trace: list[str]
    answer: AgentAnswer
    events: list[StreamEvent]
    tool_outcomes: list[dict[str, Any]]
    error: str
    _need_web_fallback: bool
    _need_refine: bool
    _refine_count: int
    _interrupt_enabled: bool
