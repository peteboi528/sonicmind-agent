"""Phase 1/2 测试：行为分数回灌 + explore/exploit 配比 + 探索反馈闭环。"""

from app.memory import compute_behavior_scores
from app.models import Asset, AssetStatus, ListeningEvent, TasteProfile
from app.recommend.engine import (
    compute_taste_profile,
    score_track,
)


def _asset(aid: str, genre: list[str], duration: int = 200) -> Asset:
    return Asset(
        asset_id=aid,
        source_url=f"https://x/{aid}",
        title=aid,
        duration_seconds=duration,
        status=AssetStatus.ANALYZED,
        genre=genre,
        energy_level=0.5,
        tempo_bpm=110,
    )


def test_behavior_completed_positive_skip_negative():
    history = [
        ListeningEvent(asset_id="a", duration_listened=200, completed=True),
        ListeningEvent(asset_id="b", duration_listened=5, completed=False),
    ]
    scores = compute_behavior_scores(history, {"a": 200, "b": 200})
    assert scores["a"] > 0
    assert scores["b"] < 0


def test_behavior_repeated_completion_accumulates():
    history = [ListeningEvent(asset_id="a", duration_listened=200, completed=True) for _ in range(3)]
    scores = compute_behavior_scores(history, {"a": 200})
    assert scores["a"] > 2.5  # 3 次听完累加（衰减极小）


def test_behavior_score_lifts_ranking():
    taste = TasteProfile(top_genres=[["流行", 3]], top_moods=[])
    track = _asset("hit", ["流行"])
    base = score_track(track, taste, [], set())
    boosted = score_track(track, taste, [], set(), {"hit": 3.0})
    assert boosted > base


def test_discovery_openness_rises_on_explored_completion():
    """探索性收听（top_genres 之外）被听完 → openness 调高。"""
    assets = [_asset("p1", ["流行"]), _asset("p2", ["流行"]), _asset("jazz", ["爵士"])]
    # 流行占主导成为 top_genre，爵士是探索项且被反复听完
    history = [ListeningEvent(asset_id="p1", duration_listened=200, completed=True)] * 3 + [
        ListeningEvent(asset_id="jazz", duration_listened=200, completed=True)
    ] * 3
    taste = compute_taste_profile(assets, history)
    assert taste.discovery_openness > 0.3


def test_discovery_openness_drops_on_explored_skip():
    """探索项被秒跳 → openness 调低。"""
    assets = [_asset("p1", ["流行"]), _asset("jazz", ["爵士"])]
    history = [ListeningEvent(asset_id="p1", duration_listened=200, completed=True)] * 3 + [
        ListeningEvent(asset_id="jazz", duration_listened=3, completed=False)
    ] * 3
    taste = compute_taste_profile(assets, history)
    assert taste.discovery_openness < 0.3


def test_openness_default_without_signal():
    assets = [_asset("p1", ["流行"])]
    taste = compute_taste_profile(assets, [])
    assert taste.discovery_openness == 0.3


def test_collaborating_artists_split_into_individuals():
    """合作艺人（、/feat./& 拼接）必须拆开各自计数，不能合成一条。"""
    a1 = Asset(
        asset_id="c1",
        source_url="x",
        title="t1",
        duration_seconds=200,
        status=AssetStatus.ANALYZED,
        artist="kanye west、ye",
    )
    a2 = Asset(
        asset_id="c2",
        source_url="x",
        title="t2",
        duration_seconds=200,
        status=AssetStatus.ANALYZED,
        artist="a feat. b",
    )
    taste = compute_taste_profile([a1, a2], [])
    names = {a for a, _ in taste.top_artists}
    assert "kanye west" in names
    assert "ye" in names
    assert "a" in names and "b" in names
    assert "kanye west、ye" not in names  # 合并串不该出现
    assert "a feat. b" not in names
