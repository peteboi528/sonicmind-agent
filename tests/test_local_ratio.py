"""「不要/减少 local」指令的真实生效：query 检测 → local_ratio 覆盖 → balance 排除本地。

此前 local_ratio 是硬编码默认（chat 0.4 / 每日 0.3），用户说「不要 local」也不起作用，
且 _balance_recommendation_sources 的 max(1, ...) 地板强制 ≥1 本地。这里锁定两处修复。
"""
from __future__ import annotations

import tempfile

from app.agent import AudioVisualAgent, _local_ratio_from_query
from app.models import ExternalTrack, RankingBreakdown
from app.storage import JsonStore


def _bd(score: float) -> RankingBreakdown:
    return RankingBreakdown(title="t", source="netease", score=score, reason="x", components={})


def _online(t: str) -> ExternalTrack:
    return ExternalTrack(external_id=t, title=t, artist="A", source="netease")


def _local(t: str) -> ExternalTrack:
    return ExternalTrack(external_id=t, title=t, artist="B", source="local", playback_url="http://x")


def test_no_local_phrases_map_to_zero():
    assert _local_ratio_from_query("推荐几首，不要local", 0.4) == 0.0
    assert _local_ratio_from_query("不要本地歌曲", 0.4) == 0.0
    assert _local_ratio_from_query("全要线上的", 0.4) == 0.0
    assert _local_ratio_from_query("只推线上", 0.4) == 0.0
    assert _local_ratio_from_query("no local please", 0.4) == 0.0


def test_reduce_local_phrases_map_to_low():
    assert _local_ratio_from_query("减少local", 0.4) == 0.15
    assert _local_ratio_from_query("少推本地", 0.3) == 0.15
    assert _local_ratio_from_query("少点本地", 0.4) == 0.15


def test_no_signal_keeps_default():
    assert _local_ratio_from_query("推荐几首适合跑步的歌", 0.4) == 0.4
    assert _local_ratio_from_query("推荐几首适合跑步的歌", 0.3) == 0.3
    assert _local_ratio_from_query("", 0.4) == 0.4


def test_balance_zero_ratio_excludes_local_even_when_local_scores_higher():
    """local_ratio=0：即便本地候选分更高也全部排除，只取线上。"""
    agent = AudioVisualAgent(JsonStore(tempfile.mkdtemp()))
    online = _online("On1")
    local = _local("Loc1")
    ranked = [(online, _bd(0.5)), (local, _bd(0.95))]  # 本地分更高
    out = agent._balance_recommendation_sources(ranked, top_k=5, local_ratio=0.0)
    tracks = [t for t, _ in out]
    assert local not in tracks
    assert online in tracks


def test_balance_default_ratio_keeps_local():
    """默认比例下本地仍参与（修复不能误伤正常混合推荐）。"""
    agent = AudioVisualAgent(JsonStore(tempfile.mkdtemp()))
    online = _online("On1")
    local = _local("Loc1")
    ranked = [(online, _bd(0.5)), (local, _bd(0.9))]
    out = agent._balance_recommendation_sources(ranked, top_k=5, local_ratio=0.4)
    tracks = [t for t, _ in out]
    assert local in tracks and online in tracks
