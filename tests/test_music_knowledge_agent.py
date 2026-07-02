from __future__ import annotations

import asyncio
import time

import pytest

from app.config import settings
from app.graph import nodes
from app.graph.builder import AgentGraphRunner
from app.models import AgentPlan, StreamEvent
from app.tools.contracts import ToolCall, ToolContext, ToolResult, ToolStatus
from app.tools.runtime import ToolRuntime


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
        ["music_metadata_lookup", "web_knowledge_search"],
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
        ["music_metadata_lookup", "web_knowledge_search"],
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
        "web_knowledge_search",
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
    from app.answer import collect_known_titles, guard_answer

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


@pytest.mark.network
def test_sample_source_ranking_and_relation_extraction():
    from app.knowledge import locate_sample_sources, search_sample_relations
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


@pytest.mark.network
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
    # web_knowledge 工具内部走 app.services.web_knowledge，patch 其 async 入口保证空结果。
    monkeypatch.setattr("app.services.web_knowledge.web_search_source.asearch_web_info", lambda *args, **kwargs: [])
    # 关掉所有结构化外部源与 parametric 兜底，保证 dossier.partial 由 web 空决定，确定性（不依赖网络）。
    monkeypatch.setattr("app.config.settings.enable_musicbrainz", False)
    monkeypatch.setattr("app.config.settings.enable_spotify", False)
    monkeypatch.setattr("app.config.settings.enable_discogs", False)
    monkeypatch.setattr("app.config.settings.deepseek_parametric_enabled", False)
    events = _events(agent, "讲讲 Blonde 这张专辑，乐评怎么说？")
    assert events[-1].type == "final"
    assert any(event.type == "dossier" for event in events)
    payload = events[-1].payload
    assert payload["dossier"]["partial"] is True
    latency = payload["trace_summary"]["latency_budget"]
    assert latency["budget_seconds"] == settings.knowledge_turn_budget_seconds
    assert latency["partial"] is True
    assert payload["trace_summary"]["recovery"] is False


def test_music_compare_resolves_two_artist_entities_from_natural_language():
    from app.knowledge import resolve_music_entities

    entities = resolve_music_entities(
        "比较 Drake 和 Future 的风格差异，并给我各自的入门歌",
        "music_compare",
        {"intent": "music_compare"},
    )

    assert len(entities) == 2
    assert entities[0].name == "Drake"
    assert entities[1].name == "Future"
    assert all(entity.type == "artist" for entity in entities)


def test_drake_future_compare_has_specific_axes_and_entry_tracks():
    from app.knowledge import build_dossier, dossier_answer
    from app.models import MusicEntity

    dossier = build_dossier(
        None,
        "比较 Drake 和 Future 的风格差异，并给我各自的入门歌",
        "music_compare",
        [MusicEntity(type="artist", name="Drake"), MusicEntity(type="artist", name="Future")],
        [], [], [], [], [],
    )
    text = dossier_answer(dossier)

    assert "Auto-Tune" in text
    assert "March Madness" in text
    assert "Passionfruit" in text
    assert "一个可能" not in text
    assert dossier.key_tracks


def test_drake_future_compare_profile_exposes_collabs_and_shared_ground():
    from app.knowledge import _artist_compare_profile
    from app.models import MusicEntity

    profile = _artist_compare_profile(
        MusicEntity(type="artist", name="Drake"),
        MusicEntity(type="artist", name="Future"),
    )

    assert profile is not None
    assert profile["collaboration_tracks"]
    assert "Jumpman" in profile["collaboration_tracks"]
    assert profile["shared_ground"]
    assert profile["artist_cards"][0]["name"] == "Drake"


def test_drake_weeknd_compare_has_specific_entry_tracks_and_collabs():
    from app.models import MusicEntity
    from app.tools.contracts import ToolContext
    from app.tools.handlers import _build_music_dossier

    ctx = ToolContext(
        thread_id="t",
        user_id="u",
        query="比较 Drake 和 The Weeknd 的风格差异，并给我各自的入门歌",
        plan={"intent": "music_compare"},
        prior_results=[{
            "type": "music_entity_resolution",
            "entities": [
                MusicEntity(type="artist", name="Drake").model_dump(mode="json"),
                MusicEntity(type="artist", name="The Weeknd").model_dump(mode="json"),
            ],
        }],
        agent=None,
    )

    result = _build_music_dossier({"query": ctx.query}, ctx)
    text = result.data["answer"]
    collabs = result.data["collaboration_tracks"]

    assert "Blinding Lights" in text
    assert "Headlines" in text
    assert any(track["title"] == "Crew Love" for track in collabs)
    assert "先听。" not in text


def test_music_compare_cache_is_pair_aware(agent):
    from app.knowledge import read_cached_dossier, write_cached_dossier
    from app.models import MusicDossier, MusicEntity

    drake = MusicEntity(type="artist", name="Drake")
    future = MusicEntity(type="artist", name="Future")
    weeknd = MusicEntity(type="artist", name="The Weeknd")
    cached = MusicDossier(
        entity=drake,
        summary="Drake vs Future cached compare",
        related_entities=[future],
        partial=False,
    )

    write_cached_dossier(agent, cached, intent="music_compare")

    future_hit = read_cached_dossier(agent, drake, related=[future], intent="music_compare")
    weeknd_hit = read_cached_dossier(agent, drake, related=[weeknd], intent="music_compare")

    assert future_hit is not None
    assert future_hit.summary == "Drake vs Future cached compare"
    assert weeknd_hit is None


def test_search_reviews_compare_adds_artist_and_pair_queries(monkeypatch):
    from app.knowledge import search_reviews
    from app.models import MusicEntity

    seen: list[str] = []

    def fake_search(query, max_results=5, api_key="", timeout=None):
        seen.append(query)
        return []

    monkeypatch.setattr("app.knowledge.web_search_source.search_web_info", fake_search)

    search_reviews(
        [MusicEntity(type="artist", name="Drake"), MusicEntity(type="artist", name="The Weeknd")],
        intent="music_compare",
        query="比较 Drake 和 The Weeknd 的风格差异，并给我各自的入门歌",
    )

    assert any("AllMusic biography" in query for query in seen)
    assert any("comparison style" in query for query in seen)
    assert any("Drake The Weeknd" in query for query in seen)


def test_compare_evidence_rows_keep_both_entities_in_answer():
    from app.models import MusicCitation, MusicEntity, TrackRef
    from app.tools.contracts import ToolContext
    from app.tools.handlers import _build_music_dossier

    ctx = ToolContext(
        thread_id="t",
        user_id="u",
        query="比较 Drake 和 The Weeknd 的风格差异，并给我各自的入门歌",
        plan={"intent": "music_compare"},
        prior_results=[
            {
                "type": "music_entity_resolution",
                "entities": [
                    MusicEntity(type="artist", name="Drake").model_dump(mode="json"),
                    MusicEntity(type="artist", name="The Weeknd").model_dump(mode="json"),
                ],
            },
            {
                "type": "music_metadata",
                "citations": [
                    MusicCitation(
                        source="allmusic",
                        title="Drake Biography",
                        url="https://example.com/drake",
                        kind="encyclopedia",
                        excerpt="Drake blends rap and R&B.",
                        confidence=0.8,
                    ).model_dump(mode="json"),
                    MusicCitation(
                        source="allmusic",
                        title="The Weeknd Biography",
                        url="https://example.com/weeknd",
                        kind="encyclopedia",
                        excerpt="The Weeknd is known for dark alternative R&B and synth-pop.",
                        confidence=0.8,
                    ).model_dump(mode="json"),
                ],
                "tracks": [
                    TrackRef(title="Headlines", artist="Drake", source="metadata").model_dump(mode="json"),
                    TrackRef(title="The Hills", artist="The Weeknd", source="metadata").model_dump(mode="json"),
                ],
                "metadata": [],
            },
        ],
        agent=None,
    )

    result = _build_music_dossier({"query": ctx.query}, ctx)
    evidence = result.data["evidence"]
    answer = result.data["answer"]

    assert any("Drake" in " / ".join(item.get("supports", [])) for item in evidence)
    assert any("The Weeknd" in " / ".join(item.get("supports", [])) for item in evidence)
    assert "Headlines" in answer
    assert "The Hills" in answer


def test_music_compare_resolves_real_cards_when_sources_available(monkeypatch):
    from app.models import ExternalTrack, MusicEntity
    from app.tools.contracts import ToolContext
    from app.tools.handlers import _build_music_dossier

    monkeypatch.setattr("app.search.verifier.verify_song", lambda title, artist: None)

    class Agent:
        def search_web_music(self, query, top_k=5, relevance_query="", **_kwargs):
            mapping = {
                ("Headlines", "Drake"): ExternalTrack(
                    external_id="d1",
                    title="Headlines",
                    artist="Drake",
                    source="netease",
                    cover_url="https://img.example.com/d1.jpg",
                    playback_url="https://music.163.com/song?id=d1",
                ),
                ("Jumpman", "Drake / Future"): ExternalTrack(
                    external_id="j1",
                    title="Jumpman",
                    artist="Drake & Future",
                    source="netease",
                    cover_url="https://img.example.com/j1.jpg",
                    playback_url="https://music.163.com/song?id=j1",
                ),
            }
            for (title, artist), track in mapping.items():
                if title.lower() in query.lower() and artist.split(" / ")[0].lower() in query.lower():
                    return [track]
            return []

    ctx = ToolContext(
        thread_id="t",
        user_id="u",
        query="比较 Drake 和 Future 的风格差异，并给我各自的入门歌",
        plan={"intent": "music_compare"},
        prior_results=[{
            "type": "music_entity_resolution",
            "entities": [
                MusicEntity(type="artist", name="Drake").model_dump(mode="json"),
                MusicEntity(type="artist", name="Future").model_dump(mode="json"),
            ],
        }],
        agent=Agent(),
    )

    result = _build_music_dossier({"query": ctx.query}, ctx)

    assert result.cards
    assert any(card["source"] == "netease" for card in result.cards)
    assert any(card["title"] == "Jumpman" for card in result.cards)


def test_music_compare_query_variants_help_collab_and_punctuation(monkeypatch):
    from app.models import MusicEntity
    from app.tools.contracts import ToolContext
    from app.tools.handlers import _build_music_dossier

    monkeypatch.setattr("app.search.verifier.verify_song", lambda title, artist: None)
    monkeypatch.setattr("app.sources.netease.search_netease_many", lambda query, limit=5: (
        [{
            "song_id": "d2",
            "title": "Hold On, We're Going Home",
            "artist": "Drake",
            "album": "Nothing Was the Same",
            "cover": "https://img.example.com/d2.jpg",
        }] if "hold on we re going home" in query.lower().replace("'", " ").replace(",", " ") else []
    ))

    class Agent:
        def __init__(self):
            self.queries = []

        def search_web_music(self, query, top_k=5, relevance_query="", **_kwargs):
            self.queries.append(query)
            return []

    agent = Agent()
    ctx = ToolContext(
        thread_id="t",
        user_id="u",
        query="比较 Drake 和 Future 的风格差异，并给我各自的入门歌",
        plan={"intent": "music_compare"},
        prior_results=[{
            "type": "music_entity_resolution",
            "entities": [
                MusicEntity(type="artist", name="Drake").model_dump(mode="json"),
                MusicEntity(type="artist", name="Future").model_dump(mode="json"),
            ],
        }],
        agent=agent,
    )

    result = _build_music_dossier({"query": ctx.query}, ctx)

    assert any(card["title"] == "Hold On, We're Going Home" and card["source"] == "netease" for card in result.cards)


# ── 直答路径正文/卡片去重 + md 后处理 + 流式切块 ──


def test_polish_narrative_strips_meta_tail_lines():
    """直答正文里的资料声明/风格标签/'没拿到乐评来源'尾巴应被剔除，正文叙述保留。"""
    from app.knowledge import _polish_narrative

    raw = (
        "## 背景\n"
        "Frank Ocean 在 2016 年发行了 Blonde。\n\n\n"
        "风格标签：艺术流行、另类 R&B\n"
        "资料状态：本档案主要基于模型先验知识（DeepSeek），未联网核实。\n"
        "乐评/资料共识：本轮没有拿到足够乐评来源，因此不硬凑专业评价。\n"
        "- **Nikes**：开篇曲，奠定疏离基调。\n"
    )
    out = _polish_narrative(raw)

    assert "Frank Ocean 在 2016 年发行了 Blonde。" in out
    assert "**Nikes**：开篇曲，奠定疏离基调。" in out
    assert "风格标签" not in out
    assert "资料状态" not in out
    assert "未联网核实" not in out
    assert "没有拿到足够乐评来源" not in out
    assert "\n\n\n" not in out  # 多余空行归一


def test_narrative_dossier_answer_is_body_only():
    """summary_is_narrative=True 时，dossier_answer 只回正文，不再追加机械尾巴。"""
    from app.knowledge import dossier_answer
    from app.models import MusicDossier, MusicEntity, TrackRef

    dossier = MusicDossier(
        entity=MusicEntity(type="album", name="Blonde", artist="Frank Ocean"),
        summary="## 背景\nFrank Ocean 的 Blonde 是一座孤峰。\n- **Nikes**：开篇曲。",
        summary_is_narrative=True,
        style_tags=["艺术流行", "另类 R&B"],
        key_tracks=[TrackRef(title="Nikes", artist="Frank Ocean", source="guide")],
        partial=True,
        degraded_reason="本档案主要基于模型先验知识（DeepSeek），未联网核实",
    )
    text = dossier_answer(dossier)

    assert text == dossier.summary  # 正文逐字透传
    assert "风格标签" not in text  # 标签由卡片承载，正文不重复
    assert "资料状态" not in text
    assert "可以先听" not in text


def test_parametric_dossier_answer_appends_short_note():
    """is_parametric=True 的直答：气泡末尾追加一行「未联网核实」短声明，不再是免责长文。"""
    from app.knowledge import dossier_answer
    from app.models import CareerPhase, MusicDossier, MusicEntity

    # album 直答（summary_is_narrative 分支）
    dossier = MusicDossier(
        entity=MusicEntity(type="album", name="Blonde", artist="Frank Ocean"),
        summary="## 背景\nFrank Ocean 的 Blonde 是一座孤峰。",
        summary_is_narrative=True,
        is_parametric=True,
    )
    text = dossier_answer(dossier)
    assert dossier.summary in text
    assert "未联网核实" in text
    assert "资料状态" not in text  # 不再出旧的免责长文
    assert "本档案主要基于模型先验知识" not in text

    # artist 直答（_artist_career_answer 分支）
    artist = MusicDossier(
        entity=MusicEntity(type="artist", name="The Weeknd"),
        summary="# The Weeknd：暗夜中的流行灵魂",
        summary_is_narrative=True,
        is_parametric=True,
        career_phases=[CareerPhase(phase_name="代表作品")],
    )
    atext = dossier_answer(artist)
    assert "未联网核实" in atext
    assert "资料状态" not in atext


def test_build_dossier_parametric_is_complete_not_partial():
    """parametric 直答是完整正文：is_parametric=True、不再被标 partial、无证据也不算"资料不完整"。

    回归点：旧实现把 parametric 免责声明塞进 partial_reasons 且 report.ok 也强压 partial=True，
    导致前端每张直答卡片都挂琥珀色"资料不完整"警告。
    """
    from app.knowledge import build_dossier
    from app.models import MusicEntity

    entity = MusicEntity(type="album", name="Blonde", artist="Frank Ocean")
    dossier = build_dossier(
        None, "Blonde", "album_deep_dive", [entity], [], [], [], [], [],
        web_knowledge_provider="deepseek_parametric",
        web_knowledge_answer="## 背景\nBlonde 是 Frank Ocean 2016 年的实验性 R&B 专辑。",
    )
    assert dossier.is_parametric is True
    assert dossier.summary_is_narrative is True
    assert dossier.partial is False  # 完整直答，不是"资料不完整"
    assert dossier.summary  # 正文照常生成


def test_extract_career_phases_from_parametric_text():
    """直答正文里的「年份+《作品》」应抽成职业时间线：左→右归年、无年份句不产阶段。"""
    from app.knowledge import _extract_career_phases_from_text

    body = (
        "# The Weeknd：暗夜中的流行灵魂\n\n"
        "## 背景与脉络\n"
        "2010年末，他匿名上传三张混音带《House of Balloons》《Thursday》《Echoes of Silence》。\n"
        "关键转折是2015年《Beauty Behind the Madness》——首张主流厂牌专辑。\n"
        "2020年《After Hours》与2022年《Dawn FM》进入概念化阶段。\n\n"
        "## 代表曲目与听法\n"
        "- **Can't Feel My Face**：流行巅峰，没有年份不应产阶段。\n"
    )
    phases = _extract_career_phases_from_text(body)
    assert [p.period for p in phases] == ["2010", "2015", "2020", "2022"]
    by_year = {p.period: p for p in phases}
    assert "House of Balloons" in by_year["2010"].key_releases
    assert "Thursday" in by_year["2010"].key_releases
    assert by_year["2015"].key_releases == ["Beauty Behind the Madness"]
    # 左→右归年：After Hours→2020、Dawn FM→2022，不串年
    assert by_year["2020"].key_releases == ["After Hours"]
    assert by_year["2022"].key_releases == ["Dawn FM"]
    # 无年份的听法段不产阶段
    assert all(p.phase_name != "代表作品" for p in phases)


def test_build_dossier_parametric_artist_career_from_text():
    """实体解析空（无专辑年表）的 artist 直答：career_phases 从正文抽，不再是空壳「代表作品」。"""
    from app.knowledge import build_dossier
    from app.models import MusicEntity

    entity = MusicEntity(type="artist", name="The Weeknd")
    answer = (
        "# The Weeknd\n"
        "2015年《Beauty Behind the Madness》大获成功。"
        "2020年《After Hours》成为现象级作品。"
    )
    dossier = build_dossier(
        None, "The Weeknd 的音乐路线", "artist_deep_dive", [entity], [], [], [], [], [],
        web_knowledge_provider="deepseek_parametric",
        web_knowledge_answer=answer,
    )
    assert dossier.career_phases, "应从正文抽出时间线"
    years = [p.period for p in dossier.career_phases]
    assert "2015" in years and "2020" in years
    # 不再是无年份空壳阶段
    assert all(p.phase_name != "代表作品" for p in dossier.career_phases)


def test_nonnarrative_dossier_answer_keeps_tail():
    """非直答(机械摘要)路径仍保留风格标签/资料状态尾巴，旧行为不回归。"""
    from app.knowledge import dossier_answer
    from app.models import MusicDossier, MusicEntity

    dossier = MusicDossier(
        entity=MusicEntity(type="album", name="某专辑"),
        summary="我整理了《某专辑》的可追溯音乐资料。",
        summary_is_narrative=False,
        style_tags=["流行"],
    )
    text = dossier_answer(dossier)

    assert text.startswith("某专辑：")
    assert "风格标签" in text


def test_chunk_for_stream_preserves_text_and_newlines():
    """切块只为流式观感，拼回去必须与原文逐字一致，且保住换行。"""
    from app.graph.nodes import _chunk_for_stream

    text = "## 标题\n第一段很长" + "啊" * 80 + "。结束。\n- 列表项\n"
    chunks = _chunk_for_stream(text)

    assert len(chunks) > 1  # 确实切成了多块
    assert "".join(chunks) == text  # 无损
    assert _chunk_for_stream("") == []


# ── 知识档案 × 用户曲库交叉命中（让介绍歌手/专辑结合你的库与口味）──


def _lib_agent(tmp_path, *, user_id="lib-user", tracks=()):
    """造一个带真实库的 agent，库里塞入指定 (title, artist, genre) 并听过它们，
    让 taste_profile 把这些歌手/曲风算进 top。"""
    from types import SimpleNamespace

    from app.agent import AudioVisualAgent
    from app.storage import JsonStore

    agent = AudioVisualAgent(JsonStore(tmp_path / "store"))
    for i, (title, artist, genre) in enumerate(tracks):
        t = SimpleNamespace(
            title=title, artist=artist, source="netease",
            external_id=f"id{i}", source_url="", cover_url=None, duration_seconds=200,
        )
        asset = agent.library_svc.ensure_asset_from_track(t)
        # ensure_asset_from_track 落的是 INGESTED，这里补成 analyzed + 打曲风标签，
        # 模拟歌单导入后的真实库（导入写的就是 analyzed）。
        from app.models import Asset, AssetStatus
        stored = agent.store.read_model("assets", asset.asset_id, Asset)
        stored.status = AssetStatus.ANALYZED
        stored.genre = list(genre)
        agent.store.write_model("assets", stored.asset_id, stored)
        agent.library_svc._invalidate_assets_cache()
        agent.record_listen(user_id, asset.asset_id, duration=200, completed=True)
    return agent


def test_knowledge_dossier_matches_library_by_artist(tmp_path):
    """问 Frank Ocean，库里有他的歌 → dossier.library_matches 命中 artist，正文带'结合你的曲库'。"""
    from app.knowledge import build_dossier, dossier_answer
    from app.models import MusicEntity

    agent = _lib_agent(tmp_path, tracks=[
        ("Nights", "Frank Ocean", ["R&B"]),
        ("Ivy", "Frank Ocean", ["R&B"]),
        ("无关歌", "别的歌手", ["流行"]),
    ])
    entity = MusicEntity(type="artist", name="Frank Ocean")
    dossier = build_dossier(
        agent, "讲讲 Frank Ocean", "artist_deep_dive", [entity],
        [], [], [], [], [], user_id="lib-user",
    )
    titles = {m.title for m in dossier.library_matches if m.relation == "artist"}
    assert {"Nights", "Ivy"} <= titles
    assert "无关歌" not in titles  # 别的歌手不命中 artist
    text = dossier_answer(dossier)
    assert "结合你的曲库" in text
    assert "Nights" in text or "Ivy" in text


def test_knowledge_dossier_matches_library_by_genre(tmp_path):
    """库里没这位歌手但有同曲风 → genre 扩展命中。"""
    from app.knowledge import build_dossier
    from app.models import MusicEntity

    agent = _lib_agent(tmp_path, tracks=[
        ("某说唱", "本地说唱歌手", ["说唱"]),
    ])
    entity = MusicEntity(type="artist", name="Kendrick Lamar")
    # style_tags 来自 metadata tags：喂一个说唱标签让 genre 命中。
    metadata = [{"entity": entity.model_dump(mode="json"), "summary": "rap", "tags": ["说唱"]}]
    dossier = build_dossier(
        agent, "讲讲 Kendrick Lamar", "artist_deep_dive", [entity],
        metadata, [], [], [], [], user_id="lib-user",
    )
    genre_hits = [m for m in dossier.library_matches if m.relation == "genre"]
    assert any(m.title == "某说唱" for m in genre_hits)


def test_knowledge_dossier_no_user_no_matches(tmp_path):
    """不传 user_id（或无 agent）→ 不做匹配，library_matches 为空，旧行为不变。"""
    from app.knowledge import build_dossier
    from app.models import MusicEntity

    agent = _lib_agent(tmp_path, tracks=[("Nights", "Frank Ocean", ["R&B"])])
    entity = MusicEntity(type="artist", name="Frank Ocean")
    dossier = build_dossier(
        agent, "讲讲 Frank Ocean", "artist_deep_dive", [entity],
        [], [], [], [], [],  # 不传 user_id
    )
    assert dossier.library_matches == []


def test_knowledge_dossier_cache_does_not_leak_library_across_users(tmp_path):
    """缓存按实体共享、不含用户维度：写缓存时 library_matches 必须被置空，
    命中时由 build_dossier 用当前 user_id 重算——否则会把别人的库命中串出来。"""
    from app.knowledge import read_cached_dossier, write_cached_dossier
    from app.models import LibraryMatch, MusicDossier, MusicEntity
    from app.storage import JsonStore

    class _Agent:
        def __init__(self, store):
            self.store = store

    agent = _Agent(JsonStore(tmp_path / "store"))
    entity = MusicEntity(type="artist", name="Frank Ocean")
    dossier = MusicDossier(
        entity=entity,
        summary="Frank Ocean 是当代 R&B 代表。",
        partial=False,  # 非 partial 才会写缓存
        library_matches=[LibraryMatch(title="Nights", artist="Frank Ocean", relation="artist")],
    )
    write_cached_dossier(agent, dossier, intent="artist_deep_dive")

    cached = read_cached_dossier(agent, entity, intent="artist_deep_dive")
    assert cached is not None
    assert cached.summary == "Frank Ocean 是当代 R&B 代表。"  # 知识正文照常缓存
    assert cached.library_matches == []  # 但 per-user 的库命中被置空，不跨用户泄漏

