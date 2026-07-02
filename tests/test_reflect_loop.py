from __future__ import annotations

import asyncio

from app.agent import AudioVisualAgent
from app.config import settings
from app.graph.nodes import reflect_async, route_after_reflect
from app.models import AgentPlan, ExternalTrack, RetrievalPlan
from app.storage import JsonStore


def _track(title: str, eid: str) -> ExternalTrack:
    return ExternalTrack(
        external_id=eid, title=title, artist="x", source="netease",
        genre=["流行"], mood=["欢快"],
    )


def test_async_reflect_marks_refine_and_excluded_tracks(tmp_path, monkeypatch):
    agent = AudioVisualAgent(JsonStore(tmp_path / "store"))
    monkeypatch.setattr(settings, "llm_api_key", "fake-key")
    monkeypatch.setattr(settings, "enable_reflect_refine", True)
    agent.memory.add_exclusion("u", "抖音热歌")

    async def generate(*_args, **_kwargs):
        return '{"drop": [0], "reason": "bad"}'

    monkeypatch.setattr(agent.llm, "agenerate", generate)
    drop, keep = _track("Douyin Hit", "drop"), _track("Real Song", "keep")
    plan = AgentPlan(intent="recommend", target_count=2)
    state = {
        "user_id": "u", "query": "推荐两首，不要抖音热歌", "plan": plan,
        "results": [{"type": "web_music_search", "tracks": [drop, keep]}],
        "trace": [], "events": [], "context": {}, "_refine_count": 0,
    }

    out = asyncio.run(reflect_async(agent, state))

    assert out["_need_refine"] is True
    assert route_after_reflect(out) == "refine"
    assert out["results"][0]["tracks"] == [keep]
    assert plan._excluded_tracks[0]["title"] == "Douyin Hit"


def test_async_reflect_stops_after_limit(tmp_path, monkeypatch):
    agent = AudioVisualAgent(JsonStore(tmp_path / "store"))
    monkeypatch.setattr(settings, "llm_api_key", "fake-key")
    agent.memory.add_exclusion("u", "抖音热歌")

    async def generate(*_args, **_kwargs):
        return '{"drop": [0]}'

    monkeypatch.setattr(agent.llm, "agenerate", generate)
    state = {
        "user_id": "u", "query": "推荐两首，不要抖音热歌",
        "plan": AgentPlan(intent="recommend", target_count=2),
        "results": [{"type": "web_music_search", "tracks": [_track("Douyin Hit", "drop")]}],
        "trace": [], "events": [], "context": {}, "_refine_count": 1,
    }

    out = asyncio.run(reflect_async(agent, state))
    assert out["_need_refine"] is False
    assert route_after_reflect(out) == "finalize"


# ── 知识意图自省（Reflexion）：确定性核对 + 可选重试 ──


def _knowledge_state(*, resolve_status: str, dossier: dict, entities=None, refine_count: int = 0) -> dict:
    plan = AgentPlan(
        intent="artist_deep_dive",
        retrieval_plan=RetrievalPlan(entities=entities or [], search_query="The Weeknd 的音乐路线"),
    )
    return {
        "user_id": "u", "query": "The Weeknd 的音乐路线", "plan": plan,
        "results": [{"type": "music_dossier", "dossier": dossier}],
        "tool_outcomes": [{"tool": "resolve_music_entity", "status": resolve_status, "attempt": refine_count}],
        "trace": [], "events": [], "context": {}, "_refine_count": refine_count,
    }


def test_knowledge_reflect_sufficient(tmp_path, monkeypatch):
    """档案正常：判定「结果充分」，不重试。"""
    agent = AudioVisualAgent(JsonStore(tmp_path / "store"))
    monkeypatch.setattr(settings, "llm_api_key", "fake-key")
    state = _knowledge_state(
        resolve_status="ok",
        dossier={"partial": False, "is_parametric": False, "summary": "## 正文\nThe Weeknd 是暗色流行代言人。"},
    )
    out = asyncio.run(reflect_async(agent, state))
    assert out["_need_refine"] is False
    assert route_after_reflect(out) == "finalize"
    assert any("结果充分" in t for t in out["trace"])


def test_knowledge_reflect_resolve_empty_parametric_rescued(tmp_path, monkeypatch):
    """resolve 空、但 parametric 已兜底出完整直答：不算降级，不重试（即使开关开）。"""
    agent = AudioVisualAgent(JsonStore(tmp_path / "store"))
    monkeypatch.setattr(settings, "llm_api_key", "fake-key")
    monkeypatch.setattr(settings, "enable_knowledge_refine", True)
    state = _knowledge_state(
        resolve_status="empty",
        dossier={"partial": False, "is_parametric": True, "summary": "## 正文\nThe Weeknd 暗夜流行灵魂。"},
    )
    out = asyncio.run(reflect_async(agent, state))
    assert out["_need_refine"] is False
    assert any("已由 parametric 直答兜底" in t for t in out["trace"])


def test_knowledge_reflect_degraded_flag_off_logs_only(tmp_path, monkeypatch):
    """档案降级、开关关：只出可观测判定，不触发重试（默认行为，零延迟）。"""
    agent = AudioVisualAgent(JsonStore(tmp_path / "store"))
    monkeypatch.setattr(settings, "llm_api_key", "fake-key")
    monkeypatch.setattr(settings, "enable_knowledge_refine", False)
    state = _knowledge_state(
        resolve_status="empty",
        dossier={"partial": True, "is_parametric": False, "summary": "本轮未能合成完整中文介绍。"},
    )
    out = asyncio.run(reflect_async(agent, state))
    assert out["_need_refine"] is False
    assert any("知识结果不足" in t for t in out["trace"])


def test_knowledge_reflect_degraded_flag_on_refines(tmp_path, monkeypatch):
    """档案降级、开关开：用清洗后的实体名回环重试一次 resolve。"""
    agent = AudioVisualAgent(JsonStore(tmp_path / "store"))
    monkeypatch.setattr(settings, "llm_api_key", "fake-key")
    monkeypatch.setattr(settings, "enable_knowledge_refine", True)
    state = _knowledge_state(
        resolve_status="empty",
        dossier={"partial": True, "is_parametric": False, "summary": ""},
        entities=["The Weeknd"],
    )
    out = asyncio.run(reflect_async(agent, state))
    assert out["_need_refine"] is True
    assert route_after_reflect(out) == "refine"
    # 用清洗后的实体名（去掉「的音乐路线」）重试
    assert out["plan"].retrieval_plan.search_query == "The Weeknd"
    assert any("回环重试 resolve" in t and "The Weeknd" in t for t in out["trace"])
