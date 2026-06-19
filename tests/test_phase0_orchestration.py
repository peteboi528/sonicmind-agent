"""Phase 0 编排升级回归测试：LLM 结构化意图、确定性标签、web_fallback 条件路由。"""

from __future__ import annotations

import time

import pytest

from app.agent import AudioVisualAgent
from app.config import settings
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
    assert "running workout" in plan.retrieval_plan.search_variants
    # 标签由确定性规则填充，不靠 LLM
    assert "运动" in plan.retrieval_plan.scenario_filter


def test_plan_with_llm_adds_typo_search_variant(agent):
    plan = nodes.plan_with_llm(agent, "推荐几首 Emenem 的歌")
    assert plan is not None
    assert plan.intent == "recommend"
    assert "Eminem" in plan.retrieval_plan.search_variants


def test_plan_with_llm_taste_is_memory_only(agent):
    plan = nodes.plan_with_llm(agent, "分析我的音乐品味")
    assert plan is not None
    assert plan.intent == "taste"
    assert plan.strategy == "memory_only"
    assert plan.online_required is False


def test_chat_skips_semantic_recall(agent, monkeypatch):
    calls = {"n": 0}

    def fake_recall(*args, **kwargs):
        calls["n"] += 1
        return ["不该召回"]

    monkeypatch.setattr(agent.memory, "recall_episodes", fake_recall)
    state = nodes.load_context(agent, {
        "user_id": "u-chat-recall",
        "asset_id": None,
        "query": "你好",
        "history": [],
        "top_k": 3,
    })
    out = nodes.plan_intent(agent, state)

    assert out["plan"].intent == "chat"
    assert calls["n"] == 0
    assert any("跳过跨会语义召回" in line for line in out["trace"])


def test_non_chat_attaches_deferred_semantic_recall(agent, monkeypatch):
    monkeypatch.setattr(agent.memory, "recall_episodes", lambda *_args, **_kwargs: ["三周前偏好：慵懒爵士"])
    state = nodes.load_context(agent, {
        "user_id": "u-recommend-recall",
        "asset_id": None,
        "query": "推荐几首适合深夜的歌",
        "history": [],
        "top_k": 3,
    })
    assert not any("语义召回" in line for line in state["trace"])

    out = nodes.plan_intent(agent, state)

    assert "三周前偏好：慵懒爵士" in out["context"]["memory_query"]
    assert any("语义召回 1 条" in line for line in out["trace"])


def test_plan_with_llm_playlist_target_count(agent):
    plan = nodes.plan_with_llm(agent, "帮我做 20 首 chill 歌单")
    assert plan is not None
    assert plan.intent == "playlist"
    assert plan.target_count == 20


def test_explicit_journey_keyword_overrides_llm_recommend_misclassification(agent, monkeypatch):
    wrong = AgentPlan(intent="recommend", tools_needed=["daily_recommend"])
    monkeypatch.setattr(nodes, "plan_with_llm_with_meta", lambda *_args, **_kwargs: (wrong, {}, {}))
    state = nodes.load_context(agent, {
        "user_id": "u-journey-override",
        "asset_id": None,
        "query": "做一个从清晨到深夜的学习旅程",
        "history": [],
        "top_k": 3,
    })

    out = nodes.plan_intent(agent, state)

    assert out["plan"].intent == "journey"
    assert out["plan"].tools_needed == ["journey"]


def test_mock_plan_uses_current_turn_not_journey_history(agent):
    history = "user: 做一个清晨到深夜的音乐旅程，热身到冲刺再放松\nassistant: 已生成音乐旅程"
    plan = nodes.plan_with_llm(agent, "分析我的音乐品味", history_text=history)
    assert plan is not None
    assert plan.intent == "taste"
    assert plan.strategy == "memory_only"
    assert plan.online_required is False


@pytest.mark.parametrize("query,expected_intent,expected_tool", [
    ("推荐 Taylor Swift 的专辑", "artist_albums", "artist_albums"),
    ("推荐点不一样的，做个品味实验", "taste_experiment", "taste_experiment"),
    ("找 NewJeans 的 MV 视频", "video", "video_search"),
    ("介绍 NewJeans 的背景", "artist_info", "web_info_search"),
    ("帮我导入这个网易云歌单 playlist?id=123456", "import", "import"),
])
def test_mock_plan_routes_new_intents(agent, query, expected_intent, expected_tool):
    plan = nodes.plan_with_llm(agent, query)
    assert plan is not None
    assert plan.intent == expected_intent
    assert expected_tool in plan.tools_needed


def test_keyword_plan_artist_albums_uses_album_tool():
    plan = nodes.build_agent_plan("推荐 The Weeknd 的专辑")
    assert plan.intent == "artist_albums"
    assert plan.tools_needed == ["artist_albums"]
    assert plan.strategy == "online_first"


def test_plan_falls_back_to_keyword_on_bad_json(agent, monkeypatch):
    monkeypatch.setattr(agent.llm, "generate", lambda *a, **k: "这不是 JSON")
    plan = nodes.plan_with_llm(agent, "推荐几首歌")
    assert plan is None  # 调用方会降级到 build_agent_plan


def test_plan_repairs_invalid_payload_once(agent, monkeypatch):
    calls = {"n": 0}

    def fake_generate(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return '{"intent":"recommend","entities":"周杰伦","use_local":true,"use_vector":true,"use_web":true}'
        return '{"intent":"recommend","entities":["周杰伦"],"use_local":true,"use_vector":true,"use_web":true,"search_query":"周杰伦","language":"","target_count":5,"reasoning":"repair ok"}'

    monkeypatch.setattr(agent.llm, "generate", fake_generate)
    plan = nodes.plan_with_llm(agent, "推荐5首周杰伦的歌")

    assert plan is not None
    assert plan.intent == "recommend"
    assert plan.target_count == 5
    assert plan.retrieval_plan.entities == ["周杰伦"]
    assert calls["n"] == 2


def test_plan_returns_none_when_repair_still_invalid(agent, monkeypatch):
    monkeypatch.setattr(agent.llm, "generate", lambda *a, **k: "not valid json at all")
    plan = nodes.plan_with_llm(agent, "推荐几首歌")
    assert plan is None


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
    assert nodes.route_after_execute({"_need_web_fallback": False}) == "reflect"


def test_execute_tools_isolates_single_tool_failure(agent, monkeypatch):
    plan = AgentPlan(
        intent="recommend",
        tools_needed=["web_music_search", "recommend"],
        online_required=True,
    )

    def boom(*args, **kwargs):
        raise RuntimeError("recommend down")

    monkeypatch.setattr(agent, "recommend_for_query", boom)
    state = nodes.execute_tools(agent, {
        "user_id": "u-tool-fail",
        "asset_id": None,
        "query": "推荐几首歌",
        "history": [],
        "top_k": 3,
        "plan": plan,
        "results": [],
        "trace": [],
        "events": [],
    })

    assert any(r.get("type") == "web_music_search" for r in state["results"])
    assert any("[tool_error] recommend" in line for line in state["trace"])
    assert any(event.type == "error" and event.payload.get("tool") == "recommend" for event in state["events"])
    assert state["_need_web_fallback"] is False


def test_web_fallback_isolates_web_search_failure(agent, monkeypatch):
    plan = AgentPlan(intent="recommend", tools_needed=["recommend"], online_required=True)

    def boom(*args, **kwargs):
        raise RuntimeError("web down")

    monkeypatch.setattr(agent, "search_web_music", boom)
    state = nodes.web_fallback(agent, {
        "user_id": "u-web-fail",
        "asset_id": None,
        "query": "推荐几首歌",
        "history": [],
        "top_k": 3,
        "plan": plan,
        "results": [],
        "trace": [],
        "events": [],
    })

    assert any("[tool_error] web_music_search" in line for line in state["trace"])
    assert any(event.type == "error" and event.payload.get("tool") == "web_music_search" for event in state["events"])
    assert state["_need_web_fallback"] is False


def test_reflect_node_failure_isolated(agent, monkeypatch):
    monkeypatch.setattr(nodes, "_reflect_impl", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("reflect down")))
    out = nodes.reflect(agent, {
        "user_id": "u-reflect-fail",
        "asset_id": None,
        "query": "推荐几首歌",
        "history": [],
        "top_k": 3,
        "plan": AgentPlan(intent="recommend", tools_needed=["recommend"]),
        "results": [],
        "trace": [],
        "events": [],
    })

    assert out["_need_refine"] is False
    assert any("[reflect_error]" in line for line in out["trace"])
    assert any(event.type == "eval" for event in out["events"])


def test_reflect_adds_eval_when_evaluate_node_skipped(agent):
    out = nodes.reflect(agent, {
        "user_id": "u-reflect-eval",
        "asset_id": None,
        "query": "推荐几首歌",
        "history": [],
        "top_k": 3,
        "plan": AgentPlan(intent="recommend", tools_needed=["recommend"]),
        "results": [],
        "trace": [],
        "events": [],
    })

    assert out["_evaluated"] is True
    assert any(line.startswith("[eval]") for line in out["trace"])
    assert any(event.type == "eval" for event in out["events"])


def test_finalize_node_failure_still_emits_final(agent, monkeypatch):
    monkeypatch.setattr(nodes, "_finalize_impl", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("final down")))
    out = nodes.finalize(agent, {
        "user_id": "u-final-fail",
        "asset_id": None,
        "query": "推荐几首歌",
        "history": [],
        "top_k": 3,
        "plan": AgentPlan(intent="recommend", tools_needed=["recommend"]),
        "results": [],
        "trace": [],
        "events": [],
        "context": {},
    })

    assert out["answer"].fallback_reason.startswith("finalize_error")
    assert out["events"][-1].type == "final"
    assert "没有编造额外歌曲" in out["events"][-1].content


def test_execute_tool_chain_runs_web_before_recommend(agent, monkeypatch):
    monkeypatch.setattr(settings, "enable_parallel_tools", True, raising=False)

    def fake_safe(agent, tool, plan, query, user_id, top_k, results, trace, events):
        time.sleep(0.1)
        results.append({"type": tool})
        trace.append(tool)
        return True

    monkeypatch.setattr(nodes, "_run_tool_safely", fake_safe)
    plan = AgentPlan(intent="recommend", tools_needed=["web_music_search", "recommend"], online_required=True)
    results: list[dict] = []
    trace: list[str] = []
    events = []

    start = time.monotonic()
    executed = nodes._execute_tool_chain(agent, plan.tools_needed, plan, "q", "u", 5, results, trace, events)
    elapsed = time.monotonic() - start

    assert executed == {"web_music_search", "recommend"}
    assert [r["type"] for r in results] == ["web_music_search", "recommend"]
    assert elapsed >= 0.18


def test_execute_tool_chain_flushes_before_playlist(agent, monkeypatch):
    monkeypatch.setattr(settings, "enable_parallel_tools", True, raising=False)
    seen_by_playlist = {"has_web": False}

    def fake_safe(agent, tool, plan, query, user_id, top_k, results, trace, events):
        if tool == "web_music_search":
            results.append({"type": "web_music_search", "tracks": ["seed"]})
        if tool == "playlist":
            seen_by_playlist["has_web"] = any(r.get("type") == "web_music_search" for r in results)
            results.append({"type": "playlist", "playlist": "ok"})
        trace.append(tool)
        return True

    monkeypatch.setattr(nodes, "_run_tool_safely", fake_safe)
    plan = AgentPlan(intent="playlist", tools_needed=["web_music_search", "playlist"], online_required=True)
    results: list[dict] = []

    nodes._execute_tool_chain(agent, plan.tools_needed, plan, "q", "u", 5, results, [], [])

    assert seen_by_playlist["has_web"] is True
    assert [r["type"] for r in results] == ["web_music_search", "playlist"]


def test_execute_tool_chain_import_finishes_before_recommend(agent, monkeypatch):
    monkeypatch.setattr(settings, "enable_parallel_tools", True, raising=False)
    seen_by_recommend = {"has_import": False}

    def fake_safe(agent, tool, plan, query, user_id, top_k, results, trace, events):
        if tool == "import":
            results.append({"type": "import_netease_playlist", "result": {"tracks": []}})
        if tool == "recommend":
            seen_by_recommend["has_import"] = any(
                result.get("type") == "import_netease_playlist" for result in results
            )
            results.append({"type": "daily_recommend", "recommendation": "ok"})
        return True

    monkeypatch.setattr(nodes, "_run_tool_safely", fake_safe)
    plan = AgentPlan(intent="import", tools_needed=["import", "recommend"], online_required=True)
    results: list[dict] = []

    nodes._execute_tool_chain(agent, plan.tools_needed, plan, "q", "u", 5, results, [], [])

    assert seen_by_recommend["has_import"] is True
    assert [result["type"] for result in results] == ["import_netease_playlist", "daily_recommend"]


# ---- 端到端：graph 主路径产出可追溯答案 + trace ----

def test_chat_recommend_end_to_end(agent):
    answer = agent.chat("u-p0", "推荐几首适合跑步的歌")
    assert answer.answer
    assert any("[plan]" in t for t in answer.agent_trace)
    assert any("web_music_search" in t or "recommend" in t for t in answer.agent_trace)
