"""Phase 0 编排升级回归测试：LLM 结构化意图、确定性标签、web_fallback 条件路由。"""

from __future__ import annotations

import pytest

from app.agent import AudioVisualAgent
from app.graph import nodes
from app.graph.tag_rules import extract_genre, extract_mood, extract_scenario, extract_tags
from app.models import AgentPlan
from app.storage import JsonStore


@pytest.fixture
def agent(tmp_path):
    return AudioVisualAgent(JsonStore(tmp_path / "store"))


# ---- 确定性标签规则 ----

def test_extract_genre_maps_keywords():
    assert "摇滚" in extract_genre("来点摇滚")
    assert "电子" in extract_genre("想听 electronic / EDM")
    assert extract_genre("随便") == []


def test_extract_mood_and_scenario():
    assert "放松" in extract_mood("chill 一点的")
    assert "运动" in extract_scenario("适合跑步健身的歌")
    assert "学习" in extract_scenario("写代码时专注用")


def test_extract_tags_bundles_three_dimensions():
    tags = extract_tags("适合跑步的激昂电子乐")
    assert tags["genre"] == ["电子"]
    assert "激昂" in tags["mood"]
    assert "运动" in tags["scenario"]


# ---- LLM 结构化意图（MockLLM 走 query_plan 路径）----

def test_plan_with_llm_recommend(agent):
    plan = nodes.plan_with_llm(agent, "推荐几首适合跑步的歌")
    assert plan is not None
    assert plan.intent == "recommend"
    assert plan.online_required is True
    # 标签由确定性规则填充，不靠 LLM
    assert "运动" in plan.retrieval_plan.scenario_filter


def test_plan_with_llm_taste_is_memory_only(agent):
    plan = nodes.plan_with_llm(agent, "分析我的音乐品味")
    assert plan is not None
    assert plan.intent == "taste"
    assert plan.strategy == "memory_only"
    assert plan.online_required is False


def test_plan_with_llm_playlist_target_count(agent):
    plan = nodes.plan_with_llm(agent, "帮我做 20 首 chill 歌单")
    assert plan is not None
    assert plan.intent == "playlist"
    assert plan.target_count == 20


def test_plan_falls_back_to_keyword_on_bad_json(agent, monkeypatch):
    monkeypatch.setattr(agent.llm, "generate", lambda *a, **k: "这不是 JSON")
    plan = nodes.plan_with_llm(agent, "推荐几首歌")
    assert plan is None  # 调用方会降级到 build_agent_plan


# ---- web_fallback 条件路由 ----

def test_needs_web_fallback_when_candidates_insufficient():
    plan = AgentPlan(intent="search", tools_needed=["search"], online_required=True, target_count=3)
    assert nodes._needs_web_fallback(plan, [], {"search"}) is True
    assert nodes.route_after_execute({"_need_web_fallback": True}) == "web_fallback"


def test_no_web_fallback_when_already_searched():
    plan = AgentPlan(intent="recommend", tools_needed=["web_music_search", "recommend"], online_required=True)
    assert nodes._needs_web_fallback(plan, [], {"web_music_search", "recommend"}) is False


def test_no_web_fallback_for_taste_intent():
    plan = AgentPlan(intent="taste", tools_needed=["taste"], online_required=False)
    assert nodes._needs_web_fallback(plan, [], {"taste"}) is False
    assert nodes.route_after_execute({"_need_web_fallback": False}) == "evaluate"


# ---- 端到端：graph 主路径产出可追溯答案 + trace ----

def test_chat_recommend_end_to_end(agent):
    answer = agent.chat("u-p0", "推荐几首适合跑步的歌")
    assert answer.answer
    assert any("[plan]" in t for t in answer.agent_trace)
    assert any("web_music_search" in t or "recommend" in t for t in answer.agent_trace)
