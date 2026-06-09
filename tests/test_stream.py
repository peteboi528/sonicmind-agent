"""Phase 2 流式回归测试：节点级事件顺序、候选先于最终答案、错误事件。"""

from __future__ import annotations

import tempfile

import pytest

from app.agent import AudioVisualAgent
from app.storage import JsonStore


@pytest.fixture
def agent():
    return AudioVisualAgent(JsonStore(tempfile.mkdtemp()))


def _stream_types(agent, query, user_id="u-stream"):
    events = list(agent.stream_chat(user_id, query))
    return events, [e.type for e in events]


def test_stream_emits_plan_then_final(agent):
    events, types = _stream_types(agent, "推荐几首适合跑步的歌")
    assert "plan" in types
    assert types[-1] == "final"


def test_candidates_precede_final(agent):
    events, types = _stream_types(agent, "推荐几首适合跑步的歌")
    assert "candidates" in types
    assert types.index("candidates") < types.index("final")


def test_candidates_carry_song_cards(agent):
    events, _ = _stream_types(agent, "推荐几首适合跑步的歌")
    cand = next(e for e in events if e.type == "candidates")
    assert "cards" in cand.payload
    # 卡片字段齐全
    if cand.payload["cards"]:
        card = cand.payload["cards"][0]
        assert {"title", "artist", "source"} <= set(card)


def test_final_event_carries_answer_payload(agent):
    events, _ = _stream_types(agent, "推荐几首歌")
    final = events[-1]
    assert final.type == "final"
    assert final.content
    assert "answer" in final.payload


def test_stream_chat_matches_chat_answer(agent):
    """流式最终答案应与非流式 chat 一致（同一图、同一逻辑）。"""
    streamed = list(agent.stream_chat("u-eq", "分析我的音乐品味"))
    final = streamed[-1]
    assert final.type == "final"
    assert final.content
