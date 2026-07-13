"""「不要/减少 local」指令的真实生效：query 检测 → local_ratio 覆盖 → balance 排除本地。

此前 local_ratio 是硬编码默认（chat 0.4 / 每日 0.3），用户说「不要 local」也不起作用，
且 _balance_recommendation_sources 的 max(1, ...) 地板强制 ≥1 本地。这里锁定两处修复。
"""

from __future__ import annotations

import asyncio
import tempfile

from app.agent import AudioVisualAgent, _local_ratio_from_query
from app.models import Asset, ExternalTrack, RankingBreakdown
from app.storage import JsonStore
from tests.offline_fakes import apply_pytest_monkeypatch


def _bd(score: float) -> RankingBreakdown:
    return RankingBreakdown(title="t", source="netease", score=score, reason="x", components={})


def _online(t: str) -> ExternalTrack:
    return ExternalTrack(external_id=t, title=t, artist="A", source="netease")


def _local(t: str) -> ExternalTrack:
    return ExternalTrack(external_id=t, title=t, artist="B", source="local", playback_url="http://x")


def test_no_local_phrases_map_to_zero():
    assert _local_ratio_from_query("推荐几首，不要local", 0.4) == 0.0
    assert _local_ratio_from_query("不要本地歌曲", 0.4) == 0.0
    assert _local_ratio_from_query("推荐几首，不要本地库里的", 0.4) == 0.0
    assert _local_ratio_from_query("只想要新的，不要我库里的", 0.4) == 0.0
    assert _local_ratio_from_query("全要线上的", 0.4) == 0.0
    assert _local_ratio_from_query("只推线上", 0.4) == 0.0
    assert _local_ratio_from_query("no local please", 0.4) == 0.0


def test_reduce_local_phrases_map_to_low():
    assert _local_ratio_from_query("减少local", 0.4) == 0.15
    assert _local_ratio_from_query("少推本地", 0.3) == 0.15
    assert _local_ratio_from_query("少点本地", 0.4) == 0.15
    assert _local_ratio_from_query("多用线上结果", 0.4) == 0.15
    assert _local_ratio_from_query("优先线上", 0.4) == 0.15


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


def test_chat_async_keeps_no_local_constraint_after_query_rewrite():
    """图编排会把 recommend 的 search_query 改写成正向检索词，但不能把'不要本地'约束丢掉。"""
    agent = AudioVisualAgent(JsonStore(tempfile.mkdtemp()))
    for idx in range(3):
        asset = Asset(
            asset_id=f"l{idx}",
            source_url=f"https://local/{idx}",
            title=f"Local Seed {idx}",
            duration_seconds=200,
            artist="Local Artist",
            genre=["R&B"],
            mood=["放松"],
            status="analyzed",
        )
        agent.store.write_model("assets", asset.asset_id, asset)
    agent._invalidate_assets_cache()
    agent.library.sync_assets(agent.list_assets())

    answer = asyncio.run(agent.chat_async("u", "推荐几首适合放松的歌，不要本地库里的"))

    assert answer.recommended_tracks
    assert all(track.source == "netease" for track in answer.recommended_tracks)


def test_chat_async_english_request_filters_out_chinese_local_tracks(monkeypatch):
    """“英文歌，不要中文”在 recommend 路径上也要挡住本地中文曲目。"""
    apply_pytest_monkeypatch(monkeypatch)
    agent = AudioVisualAgent(JsonStore(tempfile.mkdtemp()))
    for idx, title in enumerate(("七里香", "晴天"), 1):
        asset = Asset(
            asset_id=f"zh{idx}",
            source_url=f"https://local/zh/{idx}",
            title=title,
            duration_seconds=200,
            artist="周杰伦",
            genre=["流行"],
            mood=["放松"],
            status="analyzed",
        )
        agent.store.write_model("assets", asset.asset_id, asset)
    agent._invalidate_assets_cache()
    agent.library.sync_assets(agent.list_assets())

    answer = asyncio.run(agent.chat_async("u-en", "推荐几首适合放松的英文歌，不要中文"))

    titles = {track.title for track in answer.recommended_tracks}
    assert "七里香" not in titles
    assert "晴天" not in titles
