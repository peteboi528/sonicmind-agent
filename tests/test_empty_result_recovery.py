from __future__ import annotations

import asyncio

from app.agent import AudioVisualAgent
from app.config import settings
from app.graph import builder, nodes
from app.graph.builder import AgentGraphRunner
from app.graph.nodes import _trace_summary, reflect_async
from app.models import AgentPlan, DailyRecommendation, ExternalTrack, RecommendedTrack, RetrievalPlan
from app.storage import JsonStore


def _agent(tmp_path) -> AudioVisualAgent:
    return AudioVisualAgent(JsonStore(tmp_path / "store"))


def _state(plan: AgentPlan, outcomes: list[dict], *, results=None, query="来点深夜放松的歌曲") -> dict:
    return {
        "user_id": "u",
        "query": query,
        "top_k": 5,
        "plan": plan,
        "results": results or [],
        "tool_outcomes": outcomes,
        "trace": [],
        "events": [],
        "context": {"dialogue_state": {}},
        "_refine_count": 0,
    }


def _outcome(tool: str, status: str, *, query="q", kind="", message="") -> dict:
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
        "attempt": 0,
    }


def test_empty_web_result_rewrites_query_and_replans_stage(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "enable_empty_result_recovery", True)
    monkeypatch.setattr(settings, "empty_result_recovery_max_attempts", 1)
    plan = AgentPlan(
        intent="recommend",
        tools_needed=["web_music_search", "recommend"],
        target_count=5,
        retrieval_plan=RetrievalPlan(
            search_query="不要越南",
            entities=["The Weeknd"],
            mood_filter=["放松"],
            genre_filter=["R&B"],
            search_variants=["late night rnb"],
        ),
    )
    state = _state(plan, [_outcome("web_music_search", "empty", query="不要越南")], query="不要越南")
    state["context"]["dialogue_state"] = {"entities": ["The Weeknd"], "mood_tags": ["chill"]}

    out = asyncio.run(reflect_async(_agent(tmp_path), state))

    assert out["_need_refine"] is True
    assert "越南" not in out["plan"].retrieval_plan.search_query
    assert "The Weeknd" in out["plan"].retrieval_plan.search_query
    assert out["plan"].tools_needed == ["web_music_search", "recommend"]
    assert [event.type for event in out["events"]][-1] == "refine"


def test_recovery_removes_multilingual_aliases_from_all_seed_sources(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "enable_empty_result_recovery", True)
    plan = AgentPlan(
        intent="recommend", tools_needed=["web_music_search", "recommend"],
        retrieval_plan=RetrievalPlan(search_query="不要越南语", mood_filter=["放松"]),
    )
    state = _state(plan, [_outcome("web_music_search", "empty", query="不要越南语")], query="不要越南语")
    state["context"]["dialogue_state"] = {
        "entities": ["Vietnamese"], "genre_tags": ["R&B"], "mood_tags": ["chill"],
    }

    out = asyncio.run(reflect_async(_agent(tmp_path), state))
    query = out["plan"].retrieval_plan.search_query.lower()
    assert "越南" not in query
    assert "vietnamese" not in query
    assert "r&b" in query and "chill" in query


def test_network_error_switches_to_local_without_retrying_web(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "enable_empty_result_recovery", True)
    plan = AgentPlan(
        intent="recommend",
        tools_needed=["web_music_search", "recommend"],
        retrieval_plan=RetrievalPlan(search_query="深夜 放松"),
    )
    state = _state(plan, [_outcome(
        "web_music_search", "error", query="深夜 放松", kind="timeout", message="timed out",
    )])

    out = asyncio.run(reflect_async(_agent(tmp_path), state))

    assert out["_need_refine"] is True
    assert out["plan"].tools_needed == ["recommend"]
    assert out["plan"].online_required is False


def test_existing_candidates_only_retry_recommend(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "enable_empty_result_recovery", True)
    track = ExternalTrack(external_id="1", title="Night", artist="Artist", source="netease")
    plan = AgentPlan(intent="recommend", tools_needed=["web_music_search", "recommend"])
    state = _state(
        plan,
        [_outcome("web_music_search", "ok"), _outcome("recommend", "empty")],
        results=[{"type": "web_music_search", "tracks": [track]}],
    )

    out = asyncio.run(reflect_async(_agent(tmp_path), state))

    assert out["_need_refine"] is True
    assert out["plan"].tools_needed == ["recommend"]


def test_terminal_statuses_and_second_attempt_do_not_recover(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "enable_empty_result_recovery", True)
    monkeypatch.setattr(settings, "empty_result_recovery_max_attempts", 1)
    plan = AgentPlan(intent="recommend", tools_needed=["recommend"])
    auth = asyncio.run(reflect_async(_agent(tmp_path), _state(plan, [_outcome("recommend", "auth_required")])))
    assert auth["_need_refine"] is False

    second = _state(plan, [{**_outcome("recommend", "empty"), "attempt": 1}])
    second["_refine_count"] = 1
    stopped = asyncio.run(reflect_async(_agent(tmp_path), second))
    assert stopped["_need_refine"] is False


def test_trace_summary_prefers_structured_outcomes():
    plan = AgentPlan(intent="recommend", tools_needed=["recommend"])
    summary = _trace_summary(
        plan, [], ["[tool_status] tool=recommend status=ok candidates=9"], [],
        [_outcome("recommend", "error", kind="timeout", message="eight seconds")],
    )

    assert summary["tool_execution_state"] == "error"
    assert summary["tool_error_details"] == [{"tool": "recommend", "message": "eight seconds"}]


def test_stream_emits_refine_and_executes_rewritten_stage(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "enable_empty_result_recovery", True)
    monkeypatch.setattr(settings, "empty_result_recovery_max_attempts", 1)
    agent = _agent(tmp_path)
    plan = AgentPlan(
        intent="recommend",
        tools_needed=["web_music_search", "recommend"],
        target_count=1,
        retrieval_plan=RetrievalPlan(
            search_query="深夜 放松",
            search_variants=["late night chill"],
        ),
    )
    plan = nodes._materialize_tool_stages(plan, "来点深夜放松的歌曲", 1)
    async def fake_plan(_agent, state):
        return {
            **state,
            "plan": plan,
            "events": state.get("events", []),
            "trace": state.get("trace", []),
        }

    monkeypatch.setattr(builder, "plan_intent_async", fake_plan)
    searches: list[str] = []
    track = ExternalTrack(external_id="night-1", title="Night One", artist="Artist", source="netease")

    def search(query, **_kwargs):
        searches.append(query)
        return [] if len(searches) == 1 else [track]

    def recommend(user_id, _query, *, seed_tracks=None, **_kwargs):
        items = list(seed_tracks or [])
        return DailyRecommendation(
            user_id=user_id,
            tracks=[RecommendedTrack(asset=item, score=1.0, reason="match") for item in items],
        )

    monkeypatch.setattr(agent, "search_web_music", search)
    monkeypatch.setattr(agent, "recommend_for_query", recommend)

    async def collect():
        return [event async for event in AgentGraphRunner(agent).astream(
            "u", None, "来点深夜放松的歌曲", top_k=1,
        )]

    events = asyncio.run(collect())

    assert any(event.type == "refine" for event in events)
    assert searches == ["深夜 放松", "late night chill"]
    assert events[-1].type == "final"
    assert events[-1].payload["trace_summary"]["recovery"] is True


def test_llm_recovery_is_single_pass_and_filters_write_tools(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "enable_empty_result_recovery", True)
    monkeypatch.setattr(settings, "llm_api_key", "fake-key")
    calls = {"count": 0}

    class FakeLLM:
        async def agenerate(self, *_args, **_kwargs):
            calls["count"] += 1
            return (
                '{"action":"retry","reason":"换成本地搜索",'
                '"search_query":"late night chill","calls":["favorite_track","search"]}'
            )

    monkeypatch.setattr(nodes, "select_llm", lambda *_args: FakeLLM())
    plan = AgentPlan(intent="recommend", tools_needed=["recommend"])
    state = _state(
        plan,
        [_outcome("recommend", "error", kind="ValueError", message="bad result")],
    )

    out = asyncio.run(reflect_async(_agent(tmp_path), state))

    assert calls["count"] == 1
    assert out["_need_refine"] is True
    assert out["plan"].tools_needed == ["search"]
    assert out["context"]["recovery_llm_used"] is True
