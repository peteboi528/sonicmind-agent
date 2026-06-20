from __future__ import annotations

import asyncio

from app.agent import AudioVisualAgent
from app.config import settings
from app.graph.nodes import reflect_async, route_after_reflect
from app.models import AgentPlan, ExternalTrack
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
