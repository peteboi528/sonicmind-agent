"""P1-F Reflection 自省节点测试。

reflect 在 mock 模式跳过（确定性 _filter_excluded 已跑），仅真实 LLM 下做语义核对。
测试对 settings.llm_api_key 显式设值，保证在「.env 有 key」和「无 key」两种环境下都确定。
"""
from __future__ import annotations

import pytest

from app.agent import AudioVisualAgent
from app.config import settings
from app.graph.nodes import _drop_tracks_from_results, _track_key, reflect
from app.models import AgentPlan, ExternalTrack
from app.storage import JsonStore


def _track(title: str, artist: str = "x", source: str = "netease", eid: str | None = None) -> ExternalTrack:
    return ExternalTrack(
        external_id=eid or title, title=title, artist=artist, source=source,
        genre=["流行"], mood=["欢快"],
    )


def test_track_key_normalizes():
    t = _track("Hello", "Adele", source="netease", eid="123")
    assert _track_key(t) == "hello|netease|123"


def test_drop_tracks_from_results_web_search():
    t0, t1 = _track("Douyin Hit"), _track("Real Song")
    results = [{"type": "web_music_search", "tracks": [t0, t1]}]
    out = _drop_tracks_from_results(results, {_track_key(t0)})
    assert out[0]["tracks"] == [t1]


def test_drop_tracks_unknown_type_graceful():
    """未知结果类型不崩，曲目保留。"""
    t0 = _track("X")
    results = [{"type": "some_new_type", "tracks": [t0]}]
    out = _drop_tracks_from_results(results, {_track_key(t0)})  # 不处理该类型 → 不动
    assert out[0]["tracks"] == [t0]


@pytest.fixture
def agent(tmp_path):
    return AudioVisualAgent(JsonStore(tmp_path / "store"))


def test_reflect_noop_when_mock(agent, monkeypatch):
    """mock 模式 reflect 直接返回，不调 LLM、不改 results。"""
    monkeypatch.setattr(settings, "llm_api_key", "")  # mock_mode = True
    t0, t1 = _track("A"), _track("B")
    state = {
        "user_id": "u", "query": "推荐几首",
        "plan": AgentPlan(intent="recommend"),
        "results": [{"type": "web_music_search", "tracks": [t0, t1]}],
        "trace": [], "events": [],
    }
    out = reflect(agent, state)
    assert len(out["results"][0]["tracks"]) == 2  # 未改动


def test_reflect_drops_violating_track(agent, monkeypatch):
    """非 mock + fake LLM 返回 drop=[0] → reflect 剔除第 0 首并记 trace。"""
    monkeypatch.setattr(settings, "llm_api_key", "fake-key")  # mock_mode = False
    agent.memory.add_exclusion("u", "抖音热歌")  # 让 _gather_constraints 非空

    def fake_generate(prompt, system=None, temperature=0.0):
        return '{"drop": [0], "reason": "抖音神曲"}'

    monkeypatch.setattr(agent.llm, "generate", fake_generate)

    t0, t1 = _track("Douyin Hit"), _track("Real Song")
    state = {
        "user_id": "u", "query": "推荐几首，不要抖音热歌",
        "plan": AgentPlan(intent="recommend"),
        "results": [{"type": "web_music_search", "tracks": [t0, t1]}],
        "trace": [], "events": [],
    }
    out = reflect(agent, state)
    assert len(out["results"][0]["tracks"]) == 1
    assert out["results"][0]["tracks"][0].title == "Real Song"
    assert any("[reflect]" in s for s in out["trace"])


def test_reflect_skips_non_listing_intent(agent, monkeypatch):
    """chat 等不列曲目的意图，reflect 不介入（不调 LLM）。"""
    monkeypatch.setattr(settings, "llm_api_key", "fake-key")
    called = {"gen": False}
    monkeypatch.setattr(agent.llm, "generate", lambda *a, **k: called.update(gen=True) or "{}")
    state = {
        "user_id": "u", "query": "你好",
        "plan": AgentPlan(intent="chat"),
        "results": [], "trace": [], "events": [],
    }
    reflect(agent, state)
    assert not called["gen"]


def test_reflect_no_constraints_skips_llm(agent, monkeypatch):
    """无任何约束时不调 LLM（没东西可核对）。"""
    monkeypatch.setattr(settings, "llm_api_key", "fake-key")
    called = {"gen": False}
    monkeypatch.setattr(agent.llm, "generate", lambda *a, **k: called.update(gen=True) or "{}")
    t0 = _track("A")
    state = {
        "user_id": "u", "query": "推荐几首",  # 无负面偏好，无排除规则
        "plan": AgentPlan(intent="recommend"),
        "results": [{"type": "web_music_search", "tracks": [t0]}],
        "trace": [], "events": [],
    }
    reflect(agent, state)
    assert not called["gen"]
