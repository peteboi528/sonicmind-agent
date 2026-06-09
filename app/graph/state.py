from __future__ import annotations

from typing import Any, TypedDict

from app.models import AgentAnswer, AgentPlan, StreamEvent


class AgentState(TypedDict, total=False):
    user_id: str
    asset_id: str | None
    query: str
    top_k: int
    history: list[dict[str, Any]]
    plan: AgentPlan
    context: dict[str, Any]
    results: list[dict[str, Any]]
    trace: list[str]
    answer: AgentAnswer
    events: list[StreamEvent]
    error: str
    _need_web_fallback: bool
