from __future__ import annotations

import asyncio
import time

from app.config import settings
from app.graph import nodes
from app.graph.builder import AgentGraphRunner
from app.models import AgentPlan, StreamEvent
from app.tools.contracts import ToolCall, ToolContext, ToolResult, ToolStatus
from app.tools.runtime import ToolRuntime


import pytest


@pytest.fixture
def agent(tmp_path):
    from app.agent import AudioVisualAgent
    from app.storage import JsonStore

    return AudioVisualAgent(JsonStore(tmp_path / "store"))


def _run(coro):
    return asyncio.run(coro)


def _events(agent, query: str) -> list[StreamEvent]:
    return _run(_collect(agent, query))


async def _collect(agent, query: str) -> list[StreamEvent]:
    return [event async for event in AgentGraphRunner(agent).astream("u-knowledge", None, query, thread_id="t-knowledge")]


def test_album_deep_dive_keyword_routes_to_fixed_knowledge_stages():
    plan = nodes.build_agent_plan("讲讲 Blonde 这张专辑，乐评怎么说？")
    assert plan.intent in {"album_deep_dive", "review_summary"}
    plan = nodes._materialize_tool_stages(plan, "讲讲 Blonde 这张专辑，乐评怎么说？", 5)
    assert [[call.name for call in stage.calls] for stage in plan.stages] == [
        ["resolve_music_entity"],
        ["music_metadata_lookup", "review_search"],
        ["build_music_dossier"],
    ]
    assert plan.stages[1].parallel is True


def test_album_listening_note_routes_to_knowledge_not_discuss():
    query = (
        "这张专辑，我最想让你先听《Self Control》和《White Ferrari》。"
        "前者那种断断续续的假声，像在深夜对着自己说话；"
        "后者则是一段公路上的沉默，钢琴和氛围音把揉得很轻。"
        "如果你喜欢那种的感觉，这两首最对味。"
    )

    plan = nodes.build_agent_plan(query)
    assert plan.intent == "album_deep_dive"
    plan = nodes._materialize_tool_stages(plan, query, 5)
    assert [[call.name for call in stage.calls] for stage in plan.stages] == [
        ["resolve_music_entity"],
        ["music_metadata_lookup", "review_search"],
        ["build_music_dossier"],
    ]
    assert "web_music_search" not in plan.tools_needed


def test_album_keyword_overrides_llm_discuss_plan(agent):
    from app.models import RetrievalPlan

    query = "这张专辑先听哪几首？我想知道哪些歌最能进入它的状态。"
    llm_plan = AgentPlan(
        intent="discuss",
        tools_needed=["web_music_search"],
        online_required=True,
        retrieval_plan=RetrievalPlan(use_web=True, search_query="Self Control White Ferrari"),
        reasoning_summary="误判为普通音乐讨论",
    )
    state = {
        "query": query,
        "user_id": "u-knowledge",
        "top_k": 5,
        "context": {"semantic_recall_pending": False},
        "trace": [],
        "events": [],
    }

    out = nodes._finish_plan_intent(agent, state, llm_plan, {}, {})
    plan = out["plan"]
    assert plan.intent == "album_deep_dive"
    assert plan.tools_needed == [
        "resolve_music_entity",
        "music_metadata_lookup",
        "review_search",
        "build_music_dossier",
    ]


def test_music_compare_cleans_common_album_aliases():
    from app.knowledge import resolve_music_entities

    entities = resolve_music_entities("Blonde 和 orange channel的区别", "music_compare", {"intent": "music_compare"})
    assert [entity.name for entity in entities] == ["Blonde", "Channel Orange"]


def test_album_query_binds_explicit_artist_before_canonicalization():
    from app.knowledge import resolve_music_entities

    entities = resolve_music_entities("讲讲 Frank Ocean 的 Blonde 这张专辑，乐评怎么说？", "review_summary", {"intent": "review_summary"})

    assert len(entities) == 1
    assert entities[0].type == "album"
    assert entities[0].name == "Blonde"
    assert entities[0].artist == "Frank Ocean"


def test_field_style_album_input_preserves_title_and_artist():
    from app.knowledge import resolve_music_entities

    query = "album\nBlonde\nFrank Ocean"
    entities = resolve_music_entities(query, "review_summary", {"intent": "review_summary"})

    assert len(entities) == 1
    assert entities[0].type == "album"
    assert entities[0].name == "Blonde"
    assert entities[0].artist == "Frank Ocean"


def test_two_line_album_input_preserves_title_and_artist():
    from app.knowledge import resolve_music_entities

    query = "Blonde\nFrank Ocean"
    entities = resolve_music_entities(query, "review_summary", {"intent": "review_summary"})

    assert len(entities) == 1
    assert entities[0].type == "album"
    assert entities[0].name == "Blonde"
    assert entities[0].artist == "Frank Ocean"


def test_review_search_passes_bounded_timeout(monkeypatch):
    from app.knowledge import search_reviews
    from app.models import MusicEntity

    seen: list[float] = []

    def fake_search(_query, max_results=5, api_key="", timeout=None):
        seen.append(timeout)
        return [{
            "title": "Blonde / Endless Album Review - Frank Ocean - Pitchfork",
            "url": "https://pitchfork.com/reviews/albums/22295-blonde-endless/",
            "content": "Frank Ocean returns with richly emotional songwriting.",
        }]

    monkeypatch.setattr("app.knowledge.web_search_source.search_web_info", fake_search)

    payload = search_reviews([MusicEntity(type="album", name="Blonde", artist="Frank Ocean")])

    assert payload["citations"]
    assert seen
    assert all(value is not None and value <= settings.knowledge_review_timeout_seconds for value in seen)


def test_kid_a_ok_computer_compare_uses_professional_profile():
    from app.knowledge import build_dossier, dossier_answer
    from app.models import MusicEntity

    dossier = build_dossier(
        None,
        "Kid A 和 OK Computer 的区别",
        "music_compare",
        [MusicEntity(type="album", name="Kid A"), MusicEntity(type="album", name="OK Computer")],
        [], [], [], [], [],
    )
    text = dossier_answer(dossier)
    assert "Kid A" in text
    assert "OK Computer" in text
    assert "声音/制作" in text
    assert "主题" in text or "情绪" in text
    assert "Everything In Its Right Place" in text
    assert "前者" not in text
    assert "一个可能" not in text


def test_knowledge_planned_arguments_keep_original_compare_query():
    plan = AgentPlan(
        intent="music_compare",
        tools_needed=["resolve_music_entity", "music_metadata_lookup", "review_search", "build_music_dossier"],
    )
    plan.retrieval_plan.search_query = "Blonde Orange Channel Frank Ocean"
    args = nodes._planned_arguments("resolve_music_entity", "Blonde 和 orange channel的区别", plan, 5)
    assert args["query"] == "Blonde 和 orange channel的区别"


def test_dossier_synthesizes_chinese_prose_from_evidence():
    """有真实证据 + LLM 返回 JSON 时，summary/critical_consensus 走合成而非原始摘录直出。"""
    from app.knowledge import build_dossier
    from app.models import MusicCitation, MusicEntity, ReviewOpinion

    class _StubLLM:
        def generate(self, prompt, system=None, temperature=0.7, thinking=None):
            return (
                '{"summary": "Blonde 是 Frank Ocean 2016 年的另类 R&B 专辑，以氛围化制作著称。",'
                ' "critical_consensus": "乐评普遍称赞其情绪深度与制作，少数认为结构松散。"}'
            )

    class _Agent:
        llm = _StubLLM()

    entity = MusicEntity(type="album", name="Blonde", artist="Frank Ocean")
    metadata = [{"entity": entity.model_dump(mode="json"), "summary": "alternative R&B record from 2016", "tags": ["r&b"]}]
    reviews = [MusicCitation(source="pitchfork", title="Blonde review", url="https://pitchfork.com/x",
                             kind="review", excerpt="Four years after Channel Orange, Frank Ocean returns", confidence=0.9)]
    opinions = [ReviewOpinion(source="pitchfork", sentiment="positive", summary="praises the production", citation_id=0)]

    dossier = build_dossier(
        _Agent(), "讲讲 Blonde", "album_deep_dive", [entity],
        metadata, [], reviews, opinions, [],
    )
    assert "Frank Ocean" in dossier.summary
    assert "另类 R&B" in dossier.summary or "氛围" in dossier.summary
    # 不再是原始英文摘录直出
    assert "Four years after" not in dossier.critical_consensus
    assert "乐评" in dossier.critical_consensus


def test_dossier_falls_back_to_mechanical_when_llm_returns_non_json():
    """LLM 返回非 JSON（如 MockLLM 散文）时，安全回落机械摘要，不抛错。"""
    from app.knowledge import build_dossier
    from app.models import MusicEntity

    class _ProseLLM:
        def generate(self, prompt, system=None, temperature=0.7, thinking=None):
            return "这是一段没有 JSON 结构的散文回复。"

    class _Agent:
        llm = _ProseLLM()

    entity = MusicEntity(type="album", name="Blonde", artist="Frank Ocean")
    metadata = [{"entity": entity.model_dump(mode="json"), "summary": "alternative R&B record", "tags": ["r&b"]}]
    dossier = build_dossier(
        _Agent(), "讲讲 Blonde", "album_deep_dive", [entity], metadata, [], [], [], [],
    )
    # LLM 非 JSON：安全回落到中文机械 summary（不再直出英文 meta_text，避免半句英文），并标记降级。
    assert dossier.summary.startswith("我整理了《Blonde》")
    assert dossier.partial is True


def test_sample_lookup_routes_to_sample_tool_chain():
    plan = nodes.build_agent_plan("Bound 2 采样了什么，源曲给我调出来")
    assert plan.intent == "sample_lookup"
    plan = nodes._materialize_tool_stages(plan, "Bound 2 采样了什么，源曲给我调出来", 5)
    assert [[call.name for call in stage.calls] for stage in plan.stages] == [
        ["resolve_music_entity"],
        ["sample_relation_search"],
        ["locate_sample_sources"],
        ["build_sample_dossier"],
    ]


def test_guard_whitelists_compare_related_entity():
    from app.answer import guard_answer, collect_known_titles

    results = [{
        "type": "music_dossier",
        "dossier": {
            "entity": {"type": "album", "name": "Blonde"},
            "related_entities": [{"type": "album", "name": "Channel Orange"}],
            "key_tracks": [],
        },
    }]
    known = collect_known_titles(results)
    cleaned, removed = guard_answer("《Blonde》和《Channel Orange》", known)
    assert cleaned == "《Blonde》和《Channel Orange》"
    assert removed == []


def test_knowledge_intent_does_not_recover_empty_results(agent):
    plan = AgentPlan(intent="album_deep_dive", tools_needed=["resolve_music_entity"])
    out = _run(nodes._prepare_empty_result_recovery_async(agent, {
        "plan": plan,
        "_refine_count": 0,
        "tool_outcomes": [{"tool": "review_search", "status": "empty", "attempt": 0}],
    }))
    assert out is None


def test_runtime_skips_knowledge_tool_when_deadline_expired():
    from app.tools.handlers import install_default_handlers

    install_default_handlers()
    result = _run(ToolRuntime().execute(
        ToolCall(name="review_search", arguments={"query": "Blonde review"}),
        ToolContext(thread_id="t", user_id="u", query="Blonde review", deadline_at=time.monotonic() + 0.05),
    ))
    assert result.status == ToolStatus.EMPTY
    assert result.metrics["deadline_skipped"] is True
    assert result.data["skipped_due_to_deadline"] == ["review_search"]


def test_runtime_degrades_knowledge_timeout_without_error(monkeypatch):
    from app.tools.registry import TOOL_REGISTRY

    spec = TOOL_REGISTRY["review_search"]
    original = spec.handler
    original_timeout = spec.timeout_seconds

    def slow_handler(_args, _ctx):
        time.sleep(0.1)
        return ToolResult(tool="review_search", status=ToolStatus.OK)

    spec.handler = slow_handler
    spec.timeout_seconds = 0.02
    try:
        result = _run(ToolRuntime().execute(
            ToolCall(name="review_search", arguments={"query": "Blonde review"}),
            ToolContext(thread_id="t", user_id="u", query="Blonde review"),
        ))
    finally:
        spec.handler = original
        spec.timeout_seconds = original_timeout
    assert result.status == ToolStatus.EMPTY
    assert result.error is None
    assert result.metrics["timeout_as_degraded"] is True
    assert result.metrics["deadline_skipped"] is False
    assert result.data["timed_out_tools"] == ["review_search"]


def test_sample_source_ranking_and_relation_extraction():
    from app.knowledge import search_sample_relations, locate_sample_sources
    from app.models import MusicEntity, SampleEvidence, TrackRef

    payload = search_sample_relations([MusicEntity(type="track", name="Bound 2")], "Bound 2 采样了什么")
    evidence = [SampleEvidence.model_validate(item) for item in payload["evidence"]]
    assert evidence
    assert evidence[0].source == "whosampled"
    assert evidence[0].confidence > 0.8

    class FakeAgent:
        def search_web_music(self, *_args, **_kwargs):
            return []

    located = locate_sample_sources(FakeAgent(), TrackRef(title="Bound 2", artist="Kanye West"), evidence)
    assert located["relations"]
    rel = located["relations"][0]
    assert rel["relation_type"] == "sample"
    assert rel["source_track"]["title"] == "Bound"
    assert "Ponderosa" in rel["source_track"]["artist"]


def test_sample_stream_returns_dossier_and_source_cards(agent, monkeypatch):
    from app.models import ExternalTrack

    monkeypatch.setattr(agent, "search_web_music", lambda *args, **kwargs: [
        ExternalTrack(
            external_id="bound-source",
            title="Bound",
            artist="Ponderosa Twins Plus One",
            source="netease",
            playback_url="https://music.163.com/song?id=1",
        )
    ])
    events = _events(agent, "Bound 2 采样了什么，源曲给我调出来")
    assert events[-1].type == "final"
    payload = events[-1].payload
    assert payload["sample_dossier"]["relations"]
    assert payload["sample_relations"]
    assert payload["cards"]
    assert any(event.type == "sample_relations" for event in events)


def test_knowledge_stream_returns_dossier_and_latency_budget(agent, monkeypatch):
    monkeypatch.setattr("app.knowledge.web_search_source.search_web_info", lambda *args, **kwargs: [])
    # 关掉所有结构化外部源，保证 dossier.partial 由 web 空决定，确定性（不依赖网络）。
    monkeypatch.setattr("app.config.settings.enable_musicbrainz", False)
    monkeypatch.setattr("app.config.settings.enable_spotify", False)
    monkeypatch.setattr("app.config.settings.enable_discogs", False)
    events = _events(agent, "讲讲 Blonde 这张专辑，乐评怎么说？")
    assert events[-1].type == "final"
    assert any(event.type == "dossier" for event in events)
    payload = events[-1].payload
    assert payload["dossier"]["partial"] is True
    latency = payload["trace_summary"]["latency_budget"]
    assert latency["budget_seconds"] == settings.knowledge_turn_budget_seconds
    assert latency["partial"] is True
    assert payload["trace_summary"]["recovery"] is False
