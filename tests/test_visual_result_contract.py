"""可视结果契约：有曲目/专辑的工具必须贯通 SSE、final cards 与 trace。"""
from __future__ import annotations

import asyncio

from app.answer import collect_tracks as _collect_track_candidates
from app.graph.nodes import (
    _drop_tracks_from_results,
    _finalize_fallback,
    _run_tool_async,
    _select_listed_tracks,
    _trace_summary,
    _track_key,
)
from app.models import (
    AgentPlan,
    DailyRecommendation,
    ExternalTrack,
    RecommendedTrack,
    RetrievalPlan,
    SearchResponse,
    StreamEvent,
)
from app.tools.contracts import ToolCall


def _track(index: int, source: str = "netease") -> ExternalTrack:
    return ExternalTrack(
        external_id=str(index), title=f"Track {index}", artist="Artist", source=source,
    )


def _plan(intent: str, target: int | None = None) -> AgentPlan:
    return AgentPlan(
        intent=intent,
        strategy="online_first",
        tools_needed=[],
        target_count=target,
        retrieval_plan=RetrievalPlan(use_web=True, search_query="test"),
    )


def _run(fake_agent, tool: str, plan: AgentPlan):
    results, trace, events = [], [], []
    asyncio.run(_run_tool_async(
        fake_agent, ToolCall(name=tool), plan, "test", "u1", 5,
        [], results, trace, events, "thread", "run", False,
    ))
    return results, trace, events


class _Library:
    def upsert_external(self, track):
        return None


def test_search_emits_candidates_and_final_tracks():
    class Agent:
        library = _Library()

        def search(self, *args, **kwargs):
            return SearchResponse(external=[_track(1), _track(2)])

    plan = _plan("search", 2)
    results, _, events = _run(Agent(), "search", plan)

    candidate = next(event for event in events if event.type == "candidates")
    assert candidate.payload["count"] == 2
    assert len(_select_listed_tracks(results, plan)) == 2


def test_final_search_prefers_search_response_over_raw_web_results():
    raw = _track(1)
    authoritative = _track(2)
    results = [
        {"type": "web_music_search", "tracks": [raw]},
        {"type": "search", "response": SearchResponse(external=[authoritative])},
    ]

    selected = _select_listed_tracks(results, _plan("search"))

    assert [track.external_id for track in selected] == ["2"]


def test_final_recommend_prefers_reranked_results_over_raw_web_results():
    raw = _track(1)
    reranked = _track(2)
    recommendation = DailyRecommendation(
        user_id="u1",
        tracks=[RecommendedTrack(asset=reranked, score=0.9, reason="reranked")],
    )
    results = [
        {"type": "web_music_search", "tracks": [raw]},
        {"type": "daily_recommend", "recommendation": recommendation},
    ]

    selected = _select_listed_tracks(results, _plan("recommend"))

    assert [track.external_id for track in selected] == ["2"]


def test_shared_collector_uses_authoritative_results_before_raw_web_results():
    raw = _track(1)
    reranked = _track(2)
    recommendation = DailyRecommendation(
        user_id="u1",
        tracks=[RecommendedTrack(asset=reranked, score=0.9, reason="reranked")],
    )

    selected = _collect_track_candidates([
        {"type": "web_music_search", "tracks": [raw]},
        {"type": "daily_recommend", "recommendation": recommendation},
    ])

    assert {track.external_id for track in selected[:2]} == {"1", "2"}


def test_import_emits_candidates_and_final_tracks():
    class Agent:
        def import_netease_playlist(self, *args, **kwargs):
            return {
                "name": "Imported", "imported": 2, "skipped": 0, "total": 2,
                "tracks": [_track(1), _track(2)],
            }

    plan = _plan("import")
    results, trace, events = _run(Agent(), "import", plan)

    candidate = next(event for event in events if event.type == "candidates")
    assert candidate.payload["count"] == 2
    assert len(_select_listed_tracks(results, plan)) == 2
    summary = _trace_summary(plan, results, trace, candidate.payload["cards"])
    assert summary["final_cards"] == 2
    assert "netease" in summary["sources"]


def test_import_normalizes_dict_tracks_and_keeps_import_trace_alias():
    class Agent:
        def import_netease_playlist(self, *args, **kwargs):
            return {
                "name": "Imported", "imported": 2, "skipped": 0, "total": 2,
                "tracks": [_track(1).model_dump(mode="json"), _track(2).model_dump(mode="json")],
            }

    plan = _plan("import")
    results, trace, events = _run(Agent(), "import", plan)

    assert any(line.startswith("[import]") for line in trace)
    assert len(_select_listed_tracks(results, plan)) == 2
    candidate = next(event for event in events if event.type == "candidates")
    assert candidate.payload["cards"][0]["title"] == "Track 1"


def test_album_cards_are_counted_in_trace_summary():
    class Agent:
        async def recommend_artist_albums_async(self, *args, **kwargs):
            return [
                {"id": "a1", "name": "Album 1", "artist": "Artist"},
                {"id": "a2", "name": "Album 2", "artist": "Artist"},
            ]

    plan = _plan("artist_albums")
    results, trace, events = _run(Agent(), "artist_albums", plan)
    assert sum(event.type == "album_card" for event in events) == 2

    summary = _trace_summary(plan, results, trace, [])
    assert summary["final_cards"] == 2
    assert summary["sources"] == ["netease"]


def test_trace_summary_distinguishes_empty_execution_from_no_tool():
    plan = AgentPlan(
        intent="recommend", strategy="online_first",
        tools_needed=["web_music_search", "recommend"],
    )
    summary = _trace_summary(
        plan,
        [],
        [
            "[tool_status] tool=web_music_search status=empty candidates=0",
            "[tool_status] tool=recommend status=empty candidates=0",
        ],
        [],
    )
    assert summary["tool_execution_state"] == "empty"
    assert summary["tools_planned"] == ["web_music_search", "recommend"]
    assert summary["tools_executed"] == ["web_music_search", "recommend"]
    assert summary["empty_results"] == ["web_music_search", "recommend"]
    assert summary["sources"] == []


def test_trace_summary_exposes_tool_error():
    plan = AgentPlan(intent="recommend", tools_needed=["recommend"])
    summary = _trace_summary(plan, [], ["[tool_error] recommend 失败，已跳过：timeout"], [])
    assert summary["tool_execution_state"] == "error"
    assert summary["tool_errors"] == ["recommend"]
    assert summary["tool_error_details"] == [{"tool": "recommend", "message": "timeout"}]


def test_reflection_removes_search_tracks_from_response_shape():
    response = SearchResponse(external=[_track(1), _track(2)])
    results = [{"type": "search", "response": response}]

    _drop_tracks_from_results(results, {_track_key(_track(1))})

    assert [track.external_id for track in response.external] == ["2"]


def test_finalize_fallback_preserves_previously_streamed_cards():
    card = {
        "title": "Track 1", "artist": "Artist", "source": "netease", "source_id": "1",
    }
    state = {
        "query": "test",
        "plan": _plan("search"),
        "events": [StreamEvent(type="candidates", payload={"cards": [card]})],
        "trace": [],
        "context": {},
    }

    finalized = _finalize_fallback(state, RuntimeError("boom"))
    event = finalized["events"][-1]

    assert event.type == "final"
    assert event.payload["cards"] == [card]
    assert event.payload["trace_summary"]["final_cards"] == 1
