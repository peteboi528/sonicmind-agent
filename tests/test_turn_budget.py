"""P4：全局 turn 预算 + reflect 阶梯。

验证四件事：
1. 通用路径（recommend/search/playlist）单轮墙钟超预算 → reflect 停止 refine/recovery，
   由 finalize shortfall 兜底（治"超时卡死"）。
2. knowledge intent 走自己的 tool 层 deadline_at，不受 turn_deadline_at 影响。
3. 旧 checkpoint state 无 turn_deadline_at → 向后兼容，不卡。
4. max_attempts=2 允许第二 attempt 恢复（reflect 阶梯），不被 max_attempts=1 卡死。
"""

from __future__ import annotations

import asyncio
import time

from app.agent import AudioVisualAgent
from app.config import settings
from app.graph import nodes
from app.graph.nodes import reflect_async, route_after_reflect
from app.models import AgentPlan, RetrievalPlan
from app.storage import JsonStore


def _agent(tmp_path) -> AudioVisualAgent:
    return AudioVisualAgent(JsonStore(tmp_path / "store"))


def _outcome(tool: str, status: str, *, query: str = "q", kind: str = "", message: str = "", attempt: int = 0) -> dict:
    error = {"kind": kind, "message": message, "retryable": False} if kind else None
    return {
        "call_id": f"{tool}-1",
        "tool": tool,
        "status": status,
        "arguments": {"query": query},
        "summary": "",
        "error": error,
        "card_count": 0,
        "provenance": [],
        "metrics": {},
        "attempt": attempt,
    }


def _state(
    plan: AgentPlan,
    outcomes: list[dict] | None = None,
    *,
    turn_deadline: float | None = None,
    refine_count: int = 0,
    query: str = "来点深夜放松的歌曲",
) -> dict:
    ctx: dict = {"dialogue_state": {}}
    if turn_deadline is not None:
        ctx["turn_deadline_at"] = turn_deadline
    return {
        "user_id": "u",
        "query": query,
        "top_k": 5,
        "plan": plan,
        "results": [],
        "tool_outcomes": outcomes or [],
        "trace": [],
        "events": [],
        "context": ctx,
        "_refine_count": refine_count,
    }


def test_reflect_stops_refine_when_turn_budget_exceeded(tmp_path, monkeypatch):
    """通用路径 turn_deadline_at 过期 → reflect 直接 _need_refine=False，不走 recovery。"""
    monkeypatch.setattr(settings, "enable_empty_result_recovery", True)
    plan = AgentPlan(
        intent="recommend",
        tools_needed=["recommend"],
        target_count=5,
        retrieval_plan=RetrievalPlan(search_query="放松"),
    )
    state = _state(plan, [], turn_deadline=time.monotonic() - 1.0)
    out = asyncio.run(reflect_async(_agent(tmp_path), state))
    assert out["_need_refine"] is False
    assert route_after_reflect(out) == "finalize"
    assert any("预算耗尽" in t for t in out["trace"])


def test_reflect_ignores_turn_budget_for_knowledge(tmp_path, monkeypatch):
    """knowledge intent 即使 turn_deadline_at 过期也不卡（走自己的 deadline_at）。"""
    monkeypatch.setattr(settings, "enable_empty_result_recovery", True)
    plan = AgentPlan(
        intent="album_deep_dive",
        tools_needed=["music_knowledge"],
        retrieval_plan=RetrievalPlan(search_query="专辑"),
    )
    state = _state(plan, [], turn_deadline=time.monotonic() - 1.0)
    out = asyncio.run(reflect_async(_agent(tmp_path), state))
    # knowledge 不被 turn 预算卡（_turn_budget_exceeded 对 knowledge 返回 False）
    assert not any("预算耗尽" in t for t in out["trace"])


def test_turn_budget_not_set_does_not_block(tmp_path, monkeypatch):
    """旧 state 无 turn_deadline_at → 向后兼容，不卡，正常走 recovery/reflect。"""
    monkeypatch.setattr(settings, "enable_empty_result_recovery", True)
    plan = AgentPlan(
        intent="recommend",
        tools_needed=["recommend"],
        target_count=5,
        retrieval_plan=RetrievalPlan(search_query="放松"),
    )
    state = _state(plan, [])  # 无 turn_deadline_at
    out = asyncio.run(reflect_async(_agent(tmp_path), state))
    assert not any("预算耗尽" in t for t in out["trace"])


def test_recovery_blocked_by_turn_budget(tmp_path, monkeypatch):
    """max_attempts=2 但 turn_deadline_at 过期 → recovery 不触发，直接 finalize shortfall。"""
    monkeypatch.setattr(settings, "enable_empty_result_recovery", True)
    monkeypatch.setattr(settings, "empty_result_recovery_max_attempts", 2)
    plan = AgentPlan(
        intent="recommend",
        tools_needed=["web_music_search", "recommend"],
        target_count=5,
        retrieval_plan=RetrievalPlan(search_query="深夜放松", entities=["The Weeknd"]),
    )
    state = _state(
        plan,
        [_outcome("web_music_search", "empty", query="深夜放松")],
        turn_deadline=time.monotonic() - 1.0,
    )
    out = asyncio.run(reflect_async(_agent(tmp_path), state))
    assert out["_need_refine"] is False  # 预算挡住，不恢复
    assert any("预算耗尽" in t for t in out["trace"])


def test_recovery_two_step_ladder_allows_second_attempt(tmp_path, monkeypatch):
    """max_attempts=2：_refine_count=1（已恢复过一次）时仍允许第二次恢复，证明阶梯生效。"""
    monkeypatch.setattr(settings, "enable_empty_result_recovery", True)
    monkeypatch.setattr(settings, "empty_result_recovery_max_attempts", 2)
    plan = AgentPlan(
        intent="recommend",
        tools_needed=["web_music_search", "recommend"],
        target_count=5,
        retrieval_plan=RetrievalPlan(
            search_query="深夜放松",
            entities=["The Weeknd"],
            mood_filter=["放松"],
            search_variants=["late night rnb"],
        ),
    )
    state = _state(
        plan,
        [_outcome("web_music_search", "empty", query="深夜放松", attempt=1)],
        refine_count=1,
        query="深夜放松",
    )
    state["context"]["dialogue_state"] = {"entities": ["The Weeknd"], "mood_tags": ["chill"]}
    out = asyncio.run(reflect_async(_agent(tmp_path), state))
    # max_attempts=2 → attempt=1 < 2 → recovery 进入决策；有未试变体 → 恢复
    assert out["_need_refine"] is True


def test_soft_budget_degrades_variants_and_skips_llm_recovery(tmp_path, monkeypatch):
    """soft budget：先关 search_variants，并禁止走 LLM recovery。"""
    monkeypatch.setattr(settings, "enable_empty_result_recovery", True)
    monkeypatch.setattr(settings, "llm_api_key", "fake-key")
    monkeypatch.setattr(settings, "turn_budget_soft_degrade_seconds", 6.0)
    monkeypatch.setattr(settings, "turn_budget_hard_degrade_seconds", 3.0)
    calls = {"count": 0}

    class FakeLLM:
        async def agenerate(self, *_args, **_kwargs):
            calls["count"] += 1
            return '{"action":"retry","reason":"llm","search_query":"x","calls":["recommend"]}'

    monkeypatch.setattr(nodes, "select_llm", lambda *_args: FakeLLM())
    plan = AgentPlan(
        intent="recommend",
        tools_needed=["recommend"],
        retrieval_plan=RetrievalPlan(search_query="放松", search_variants=["late night chill"]),
    )
    state = _state(
        plan,
        [_outcome("recommend", "error", kind="ValueError", message="bad result")],
        turn_deadline=time.monotonic() + 5.0,
    )

    out = asyncio.run(reflect_async(_agent(tmp_path), state))

    assert out["context"]["budget_degrade_level"] == "soft"
    assert out["plan"].retrieval_plan.search_variants == []
    assert calls["count"] == 0
    assert any("关闭 search_variants 与 LLM recovery" in t for t in out["trace"])


def test_hard_budget_recovery_switches_directly_to_local(tmp_path, monkeypatch):
    """hard budget：不再在线重搜，recovery 直接切本地。"""
    monkeypatch.setattr(settings, "enable_empty_result_recovery", True)
    monkeypatch.setattr(settings, "empty_result_recovery_max_attempts", 2)
    monkeypatch.setattr(settings, "turn_budget_soft_degrade_seconds", 6.0)
    monkeypatch.setattr(settings, "turn_budget_hard_degrade_seconds", 3.0)
    plan = AgentPlan(
        intent="recommend",
        tools_needed=["web_music_search", "recommend"],
        target_count=5,
        retrieval_plan=RetrievalPlan(
            search_query="深夜放松",
            entities=["The Weeknd"],
            search_variants=["late night rnb"],
        ),
    )
    state = _state(
        plan,
        [_outcome("web_music_search", "empty", query="深夜放松")],
        turn_deadline=time.monotonic() + 2.0,
    )

    out = asyncio.run(reflect_async(_agent(tmp_path), state))

    assert out["context"]["budget_degrade_level"] == "hard"
    assert out["_need_refine"] is True
    assert out["plan"].tools_needed == ["recommend"]
    assert any("直接切到可追溯的本地检索" in t for t in out["trace"])


def test_planning_deadline_uses_deterministic_fallback(tmp_path, monkeypatch):
    """规划 LLM 卡住时不应穿透普通请求 deadline。"""
    from app.graph import nodes
    from app.graph.planning import plan_intent_async

    async def slow_plan(*_args, **_kwargs):
        await asyncio.sleep(0.2)
        return None, {}, {}

    monkeypatch.setattr(nodes, "plan_with_llm_with_meta_async", slow_plan)
    plan = AgentPlan(intent="recommend", retrieval_plan=RetrievalPlan(search_query="放松"))
    state = _state(plan, turn_deadline=time.monotonic() + 1)
    state["context"]["deadline_at"] = time.monotonic() + 0.02
    started = time.monotonic()
    out = asyncio.run(plan_intent_async(_agent(tmp_path), state))
    assert time.monotonic() - started < 0.12
    assert out["plan"].intent == "recommend"
    assert any("规划超时" in line for line in out["trace"])
