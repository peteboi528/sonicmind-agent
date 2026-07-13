"""graph 层单轮墙钟预算与渐进降级逻辑。"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from app.config import settings
from app.graph._shared import _is_knowledge_intent
from app.models import StreamEvent

if TYPE_CHECKING:
    from app.graph.state import AgentState

logger = logging.getLogger(__name__)


def _turn_budget_exceeded(state: AgentState) -> bool:
    """通用路径单轮墙钟预算是否耗尽。

    knowledge intent 走自己的 tool 层 deadline_at，不在此卡；context 无 turn_deadline_at
    （旧 checkpoint state）时返回 False，向后兼容。
    """
    deadline = (state.get("context") or {}).get("turn_deadline_at")
    if not deadline or _is_knowledge_intent(state["plan"].intent):
        return False
    return time.monotonic() >= float(deadline)


def _remaining_turn_budget_seconds(state: AgentState) -> float | None:
    deadline = (state.get("context") or {}).get("turn_deadline_at")
    if not deadline or _is_knowledge_intent(state["plan"].intent):
        return None
    return max(0.0, float(deadline) - time.monotonic())


def _turn_budget_degrade_level(state: AgentState) -> str | None:
    remaining = _remaining_turn_budget_seconds(state)
    if remaining is None:
        return None
    if remaining <= settings.turn_budget_hard_degrade_seconds:
        return "hard"
    if remaining <= settings.turn_budget_soft_degrade_seconds:
        return "soft"
    return None


def _apply_turn_budget_degradation(state: AgentState) -> AgentState:
    """在真正耗尽 budget 前先收缩高成本扩展路径。"""
    level = _turn_budget_degrade_level(state)
    if not level:
        return state
    context = dict(state.get("context") or {})
    if context.get("budget_degrade_level") == level:
        return state
    plan = state["plan"]
    retrieval = plan.retrieval_plan
    trace = list(state.get("trace", []))
    events = list(state.get("events", []))

    updates: dict[str, Any] = {}
    if retrieval.search_variants:
        updates["search_variants"] = []
    revised = plan.model_copy(
        update={
            "retrieval_plan": retrieval.model_copy(update=updates) if updates else retrieval,
        }
    )
    context["budget_degrade_level"] = level

    if level == "soft":
        note = "[budget] 剩余预算偏紧，关闭 search_variants 与 LLM recovery，优先走确定性主路径。"
    else:
        note = "[budget] 剩余预算很紧，recovery 将直接切本地，避免继续在线扩展。"
    trace.append(note)
    events.append(StreamEvent(type="eval", content=note, payload={"budget_degrade_level": level}))
    return {**state, "plan": revised, "context": context, "trace": trace, "events": events}


def _finalize_due_to_budget(state: AgentState) -> AgentState:
    """超预算：停止一切 refine/recovery，由 finalize 的 shortfall 兜底诚实说明。"""
    return {
        **state,
        "_need_refine": False,
        "trace": [*state.get("trace", []), "[reflect] 单轮墙钟预算耗尽，停止补量，由 finalize 诚实说明 shortfall。"],
        "events": [
            *state.get("events", []),
            StreamEvent(
                type="eval",
                content="单轮时间预算耗尽，已停止补量。",
                payload={"budget_exceeded": True},
            ),
        ],
    }


def _latency_budget_summary(
    context: dict[str, Any],
    outcomes: list[dict[str, Any]],
    results: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not _is_knowledge_intent(str(context.get("intent") or "")) and not context.get("latency_budget"):
        # context may not carry intent; infer from knowledge result payloads.
        if not any(r.get("type") in {"music_dossier", "music_compare", "sample_dossier"} for r in results):
            return None
    started = context.get("started_at_monotonic")
    elapsed = round(max(0.0, time.monotonic() - float(started)), 3) if started else 0
    timed_out = [str(item.get("tool") or "") for item in outcomes if (item.get("error") or {}).get("kind") == "timeout"]
    skipped = []
    for item in outcomes:
        metrics = item.get("metrics") or {}
        if metrics.get("deadline_skipped"):
            skipped.append(str(item.get("tool") or ""))
    partial = any(
        (r.get("dossier") or {}).get("partial") for r in results if r.get("type") in {"music_dossier", "music_compare"}
    )
    partial = partial or any(
        (r.get("sample_dossier") or {}).get("partial") for r in results if r.get("type") == "sample_dossier"
    )
    budget = context.get("latency_budget") or {}
    return {
        "budget_seconds": budget.get("budget_seconds", settings.knowledge_turn_budget_seconds),
        "elapsed_seconds": elapsed,
        "timed_out_tools": [t for t in timed_out if t],
        "skipped_due_to_deadline": [s for s in skipped if s],
        "partial": partial,
    }
