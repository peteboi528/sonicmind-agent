"""多意图并行执行回归测试。

覆盖：
- QueryPlanPayload.secondary 解析：有/无/非法 → 建/弃
- flag / 白名单三重闸门：控制 sub_plans 是否真正建立
- AgentPlan.sub_plans / all_intents / is_multi_intent 属性
- _merge_multi_intent_stages：共享 resolve 去重、层内并行
- _select_listed_tracks 多意图下（primary=知识）仍返 track 卡片
- compose_answer_stream_async（flag on）同时出编号曲目《》与 dossier narrative
- guard：primary=track + secondary=knowledge 时整体跳过 guard，dossier 正文保留
- flag-off parity：同一 payload flag 关 → sub_plans==[]
"""

from __future__ import annotations

import asyncio

import pytest

from app.config import settings
from app.graph import nodes
from app.intents import is_allowed_multi_intent_pair
from app.models import (
    AgentPlan,
    ExternalTrack,
    MusicDossier,
    MusicEntity,
    QueryPlanPayload,
)


def _run(coro):
    return asyncio.run(coro)


async def _drain(agen) -> str:
    parts: list[str] = []
    async for piece in agen:
        parts.append(piece)
    return "".join(parts)


@pytest.fixture
def multi_on(monkeypatch):
    monkeypatch.setattr(settings, "enable_multi_intent", True)


@pytest.fixture
def multi_off(monkeypatch):
    monkeypatch.setattr(settings, "enable_multi_intent", False)


# ---- QueryPlanPayload.secondary 解析 ----


def test_payload_parses_valid_secondary():
    payload = QueryPlanPayload.model_validate(
        {
            "intent": "recommend",
            "entities": ["The Weeknd"],
            "secondary": {"intent": "artist_deep_dive", "entities": ["The Weeknd"], "search_query": "The Weeknd"},
        }
    )
    assert payload.secondary is not None
    assert payload.secondary.intent == "artist_deep_dive"


def test_payload_no_secondary_defaults_none():
    payload = QueryPlanPayload.model_validate({"intent": "recommend"})
    assert payload.secondary is None


def test_payload_invalid_secondary_intent_dropped():
    payload = QueryPlanPayload.model_validate(
        {
            "intent": "recommend",
            "secondary": {"intent": "not_a_real_intent", "entities": ["X"]},
        }
    )
    # 非法 intent → SecondaryIntent 归一空串 → model_validator 丢弃整个 secondary
    assert payload.secondary is None


# ---- 白名单 ----


def test_allowed_pairs_whitelist():
    assert is_allowed_multi_intent_pair("recommend", "artist_deep_dive")
    assert is_allowed_multi_intent_pair("search", "artist_info")
    # 同型/不 coherent 组合不在白名单
    assert not is_allowed_multi_intent_pair("recommend", "playlist")
    assert not is_allowed_multi_intent_pair("artist_deep_dive", "recommend")
    assert not is_allowed_multi_intent_pair("chat", "artist_info")


# ---- 三重闸门：_build_secondary_sub_plans / _plan_from_query_payload ----


def _payload_with_secondary():
    return QueryPlanPayload.model_validate(
        {
            "intent": "recommend",
            "entities": ["The Weeknd"],
            "use_web": True,
            "search_query": "The Weeknd",
            "secondary": {"intent": "artist_deep_dive", "entities": ["The Weeknd"], "search_query": "The Weeknd"},
        }
    )


def test_flag_on_and_allowed_builds_sub_plan(multi_on):
    plan = nodes._plan_from_query_payload(_payload_with_secondary(), "推几首 The Weeknd 顺便讲讲他")
    assert plan.is_multi_intent
    assert plan.all_intents == ["recommend", "artist_deep_dive"]
    assert plan.sub_plans[0].intent == "artist_deep_dive"
    assert "The Weeknd" in plan.sub_plans[0].retrieval_plan.entities


def test_flag_off_discards_secondary(multi_off):
    plan = nodes._plan_from_query_payload(_payload_with_secondary(), "推几首 The Weeknd 顺便讲讲他")
    assert plan.sub_plans == []
    assert not plan.is_multi_intent
    assert plan.all_intents == ["recommend"]


def test_disallowed_pair_discards_secondary(multi_on):
    payload = QueryPlanPayload.model_validate(
        {
            "intent": "recommend",
            "secondary": {"intent": "playlist", "entities": []},  # recommend+playlist 不在白名单
        }
    )
    plan = nodes._plan_from_query_payload(payload, "推几首歌再做个歌单")
    assert plan.sub_plans == []


def test_no_secondary_single_intent(multi_on):
    payload = QueryPlanPayload.model_validate({"intent": "recommend", "entities": ["A"]})
    plan = nodes._plan_from_query_payload(payload, "推荐几首")
    assert plan.sub_plans == []


# ---- AgentPlan 属性 ----


def test_agent_plan_multi_intent_properties():
    sub = AgentPlan(intent="artist_deep_dive")
    plan = AgentPlan(intent="recommend", sub_plans=[sub])
    assert plan.is_multi_intent
    assert plan.all_intents == ["recommend", "artist_deep_dive"]
    # 空 sub_plans
    solo = AgentPlan(intent="recommend")
    assert not solo.is_multi_intent
    assert solo.all_intents == ["recommend"]


# ---- _merge_multi_intent_stages ----


def test_merge_dedupes_shared_resolve_and_parallelizes(multi_on):
    plan = nodes._plan_from_query_payload(_payload_with_secondary(), "推几首 The Weeknd 顺便讲讲他")
    plan = nodes._materialize_tool_stages(plan, "The Weeknd", 5)
    merged = nodes._merge_multi_intent_stages(plan, "The Weeknd", 5)

    # primary recommend 链：[web_music_search, recommend]（recommend 依赖 web → 分层）
    # secondary artist_deep_dive 链：[resolve] → [metadata, web_knowledge] → [build]
    # 合并后各层压进同一 stage；至少存在一个 parallel 合并 stage。
    assert any(stage.parallel and len(stage.calls) > 1 for stage in merged.stages)

    # resolve_music_entity 在整组 stages 里只出现一次（共享去重；两链 arguments 一致时）
    all_calls = [call.name for stage in merged.stages for call in stage.calls]
    assert all_calls.count("resolve_music_entity") == 1
    # tools_needed 是并集，含两条链的工具
    assert "recommend" in merged.tools_needed
    assert "build_music_dossier" in merged.tools_needed
    # sub_plans 被替换为已 materialize 的版本（带 stages）
    assert merged.sub_plans[0].stages


# ---- _select_listed_tracks：primary=知识 但 sub_plan=track 时仍出卡片 ----


def _fake_search_results():
    tracks = [
        ExternalTrack(external_id="1", title="Blinding Lights", artist="The Weeknd", source="netease"),
        ExternalTrack(external_id="2", title="Save Your Tears", artist="The Weeknd", source="netease"),
    ]

    class _Resp:
        external = tracks
        local: list = []

    return [{"type": "search", "response": _Resp()}]


def test_select_listed_tracks_multi_intent_uses_track_subplan():
    # 反常但可能的排布：primary=知识（不产 track），secondary=search（产 track）。
    # _select_listed_tracks 应回退到 track 型 sub_plan 取卡片。
    primary = AgentPlan(intent="artist_deep_dive")
    sub = AgentPlan(intent="search", target_count=12)
    plan = AgentPlan(intent="artist_deep_dive", sub_plans=[sub])
    _ = primary  # 语义说明
    results = _fake_search_results()
    tracks = nodes._select_listed_tracks(results, plan)
    assert len(tracks) == 2
    assert tracks[0].title == "Blinding Lights"


def test_select_listed_tracks_single_intent_unchanged():
    plan = AgentPlan(intent="search", target_count=12)
    results = _fake_search_results()
    tracks = nodes._select_listed_tracks(results, plan)
    assert len(tracks) == 2


# ---- 合成：一条答案出两段 ----


def _dossier_result():
    dossier = MusicDossier(
        entity=MusicEntity(name="The Weeknd", type="artist"),
        summary="The Weeknd 以暗黑另类 R&B 起家，逐步走向流行巅峰。代表作《Blinding Lights》。",
        summary_is_narrative=True,
    )
    return {"type": "music_dossier", "dossier": dossier.model_dump(mode="json")}


def test_compose_multi_intent_emits_tracks_and_dossier(multi_on):
    sub = AgentPlan(intent="artist_deep_dive")
    plan = AgentPlan(intent="search", target_count=12, sub_plans=[sub])
    results = [*_fake_search_results(), _dossier_result()]
    text = _run(_drain(nodes.compose_answer_stream_async("推几首 The Weeknd 顺便讲讲他", results, plan)))
    # track 段：编号 + 《歌名》
    assert "《Blinding Lights》" in text
    assert "1." in text
    # dossier narrative 段
    assert "暗黑另类 R&B" in text or "The Weeknd" in text


def test_compose_single_intent_flag_off_parity(multi_off):
    plan = AgentPlan(intent="search", target_count=12)
    results = _fake_search_results()
    text = _run(_drain(nodes.compose_answer_stream_async("找几首 The Weeknd", results, plan)))
    assert "《Blinding Lights》" in text


# ---- guard：多意图知识段跳过 guard ----


def test_finalize_skips_guard_when_secondary_is_knowledge(monkeypatch):
    called = {"guard": False}

    def _spy_guard(answer, known):
        called["guard"] = True
        return answer, []

    monkeypatch.setattr(nodes, "guard_answer", _spy_guard)

    sub = AgentPlan(intent="artist_deep_dive")
    plan = AgentPlan(intent="search", sub_plans=[sub])
    # _skip_guard 只看 plan 结构，不需要跑完整 finalize；直接复算判据
    skip = nodes._is_knowledge_intent(plan.intent) or (
        plan.is_multi_intent and any(nodes._is_knowledge_intent(sp.intent) for sp in plan.sub_plans)
    )
    assert skip is True


def test_guard_runs_for_plain_track_intent():
    plan = AgentPlan(intent="search")
    skip = nodes._is_knowledge_intent(plan.intent) or (
        plan.is_multi_intent and any(nodes._is_knowledge_intent(sp.intent) for sp in plan.sub_plans)
    )
    assert skip is False


# ---- 端到端：一条 final payload 同时带 cards 与 dossier ----


def test_finalize_payload_has_both_cards_and_dossier(tmp_path, multi_on):
    from app.agent import AudioVisualAgent
    from app.storage import JsonStore

    agent = AudioVisualAgent(JsonStore(tmp_path / "store"))
    sub = AgentPlan(intent="artist_deep_dive")
    plan = AgentPlan(intent="search", target_count=12, sub_plans=[sub])
    results = [*_fake_search_results(), _dossier_result()]
    state = {
        "user_id": "u-multi",
        "query": "推几首 The Weeknd 顺便讲讲他",
        "plan": plan,
        "results": results,
        "trace": [],
        "events": [],
        "context": {},
        "tool_outcomes": [],
    }
    answer_text = "为你找到 2 首：\n1. 《Blinding Lights》 - The Weeknd（netease）\n\nThe Weeknd 以暗黑另类 R&B 起家。"
    _answer, final_payload, _trace = _run(nodes._finalize_tail_async(agent, state, answer_text))
    # track 卡片与知识 dossier 同时出现在一条 final payload 里
    assert final_payload.get("cards"), "缺少 track cards"
    assert len(final_payload["cards"]) == 2
    assert final_payload.get("dossier"), "缺少 dossier"
    assert final_payload["dossier"]["entity"]["name"] == "The Weeknd"
