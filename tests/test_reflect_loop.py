from __future__ import annotations

from app.agent import AudioVisualAgent
from app.config import settings
from app.graph import builder
from app.graph.builder import AgentGraphRunner, _fallback_invoke
from app.graph.nodes import reflect, route_after_reflect
from app.models import AgentPlan, ExternalTrack, StreamEvent
from app.storage import JsonStore


def _track(title: str, artist: str = "x", source: str = "netease", eid: str | None = None) -> ExternalTrack:
    return ExternalTrack(
        external_id=eid or title,
        title=title,
        artist=artist,
        source=source,
        genre=["流行"],
        mood=["欢快"],
    )


def test_reflect_marks_refine_and_excluded_tracks(tmp_path, monkeypatch):
    agent = AudioVisualAgent(JsonStore(tmp_path / "store"))
    monkeypatch.setattr(settings, "llm_api_key", "fake-key")
    # refine 回环默认关闭（提速：省掉第 4/5 次串行往返）；这里显式打开以测「剔除后应回环补量」这条路径。
    monkeypatch.setattr(settings, "enable_reflect_refine", True)
    agent.memory.add_exclusion("u", "抖音热歌")
    monkeypatch.setattr(agent.llm, "generate", lambda *a, **k: '{"drop": [0], "reason": "bad"}')

    keep = _track("Real Song", eid="keep")
    drop = _track("Douyin Hit", eid="drop")
    plan = AgentPlan(intent="recommend", target_count=2)
    state = {
        "user_id": "u",
        "query": "推荐两首，不要抖音热歌",
        "plan": plan,
        "results": [{"type": "web_music_search", "tracks": [drop, keep]}],
        "trace": [],
        "events": [],
        "_refine_count": 0,
    }

    out = reflect(agent, state)

    assert out["_need_refine"] is True
    assert route_after_reflect(out) == "refine"
    assert out["results"][0]["tracks"] == [keep]
    assert plan._excluded_tracks == [{
        "title": "Douyin Hit",
        "artist": "x",
        "source": "netease",
        "source_id": "drop",
    }]


def test_reflect_stops_refining_after_limit(tmp_path, monkeypatch):
    agent = AudioVisualAgent(JsonStore(tmp_path / "store"))
    monkeypatch.setattr(settings, "llm_api_key", "fake-key")
    agent.memory.add_exclusion("u", "抖音热歌")
    monkeypatch.setattr(agent.llm, "generate", lambda *a, **k: '{"drop": [0], "reason": "bad"}')

    keep = _track("Real Song", eid="keep")
    drop = _track("Douyin Hit", eid="drop")
    state = {
        "user_id": "u",
        "query": "推荐两首，不要抖音热歌",
        "plan": AgentPlan(intent="recommend", target_count=2),
        "results": [{"type": "web_music_search", "tracks": [drop, keep]}],
        "trace": [],
        "events": [],
        "_refine_count": 1,
    }

    out = reflect(agent, state)

    assert out["_need_refine"] is False
    assert route_after_reflect(out) == "finalize"


def test_fallback_invoke_refines_once(monkeypatch, tmp_path):
    agent = AudioVisualAgent(JsonStore(tmp_path / "store"))
    call_counts = {"execute": 0, "reflect": 0}

    monkeypatch.setattr(builder, "load_context", lambda agent, state: state)
    monkeypatch.setattr(builder, "plan_intent", lambda agent, state: {**state, "plan": AgentPlan(intent="recommend", target_count=2)})

    def fake_execute(agent, state):
        call_counts["execute"] += 1
        results = [{"type": "web_music_search", "tracks": [_track(f"Song {call_counts['execute']}", eid=str(call_counts["execute"]))]}]
        return {**state, "results": results, "_need_web_fallback": False, "_need_refine": False, "_refine_count": state.get("_refine_count", 0)}

    def fake_reflect(agent, state):
        call_counts["reflect"] += 1
        return {**state, "_need_refine": call_counts["reflect"] == 1}

    monkeypatch.setattr(builder, "execute_tools", fake_execute)
    monkeypatch.setattr(builder, "web_fallback", lambda agent, state: state)
    monkeypatch.setattr(builder, "reflect", fake_reflect)
    monkeypatch.setattr(builder, "finalize", lambda agent, state: {**state, "answer": "ok"})

    out = _fallback_invoke(agent, {"user_id": "u", "asset_id": None, "query": "q", "history": [], "top_k": 5, "_refine_count": 0})

    assert call_counts["execute"] == 2
    assert call_counts["reflect"] == 2
    assert out["answer"] == "ok"


def test_stream_runs_refine_loop_once(monkeypatch, tmp_path):
    agent = AudioVisualAgent(JsonStore(tmp_path / "store"))
    runner = AgentGraphRunner(agent)
    call_counts = {"execute": 0, "reflect": 0, "finalize": 0}

    monkeypatch.setattr(builder, "load_context", lambda agent, state: {**state, "events": [StreamEvent(type="plan", content="load")]})
    monkeypatch.setattr(builder, "plan_intent", lambda agent, state: {**state, "plan": AgentPlan(intent="recommend", target_count=2), "events": [*state.get("events", []), StreamEvent(type="plan", content="plan")]})

    def fake_execute(agent, state):
        call_counts["execute"] += 1
        events = [*state.get("events", []), StreamEvent(type="tool_result", content=f"execute-{call_counts['execute']}")]
        return {**state, "events": events, "_need_web_fallback": False, "_need_refine": False, "_refine_count": state.get("_refine_count", 0)}

    def fake_reflect(agent, state):
        call_counts["reflect"] += 1
        events = [*state.get("events", []), StreamEvent(type="eval", content=f"reflect-{call_counts['reflect']}")]
        return {**state, "events": events, "_need_refine": call_counts["reflect"] == 1}

    def fake_finalize_stream(agent, state):
        call_counts["finalize"] += 1
        yield StreamEvent(type="final", content="done")

    monkeypatch.setattr(builder, "execute_tools", fake_execute)
    monkeypatch.setattr(builder, "web_fallback", lambda agent, state: state)
    monkeypatch.setattr(builder, "reflect", fake_reflect)
    monkeypatch.setattr(builder, "finalize_stream", fake_finalize_stream)

    events = list(runner.stream("u", None, "q"))

    assert call_counts["execute"] == 2
    assert call_counts["reflect"] == 2
    assert call_counts["finalize"] == 1
    assert events[-1].type == "final"
