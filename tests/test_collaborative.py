from __future__ import annotations

from app.models import TasteProfile
from app.recommend.collaborative import (
    build_cooccurrence,
    collaborative_scores,
    recent_listened_ids,
)
from app.recommend.rerank import PreferenceProfile, _normalized_weights, tri_anchor_rerank


class _Ev:
    def __init__(self, asset_id: str) -> None:
        self.asset_id = asset_id


def _track(title, genre=None, mood=None, external_id=None):
    class T:
        pass

    t = T()
    t.title = title
    t.genre = genre or []
    t.mood = mood or []
    t.external_id = external_id or title
    t.source = "netease"
    return t


# ---- 共现矩阵 ----


def test_cooccurrence_symmetric_and_no_self():
    histories = [["a", "b", "c"], ["a", "b"], ["c", "d"]]
    m = build_cooccurrence(histories)
    assert m["a"]["b"] == 2  # 两个用户都听过 a+b
    assert m["b"]["a"] == 2  # 对称
    assert "a" not in m.get("a", {})  # 无自共现
    assert m["c"]["d"] == 1


def test_cooccurrence_dedupes_within_user():
    # 同一用户重复听同一曲不应放大共现
    histories = [["a", "a", "b"]]
    m = build_cooccurrence(histories)
    assert m["a"]["b"] == 1


# ---- 候选打分 ----


def test_collaborative_scores_normalized_and_ranked():
    histories = [["x", "hit"], ["x", "hit"], ["x", "weak"]]
    cooc = build_cooccurrence(histories)
    scores, ok = collaborative_scores(["hit", "weak", "cold"], recent_item_ids=["x"], cooccurrence=cooc)
    assert ok
    assert scores[0] == 1.0  # hit 与 x 共现最强 → 归一到 1
    assert 0.0 < scores[1] < 1.0  # weak 较弱
    assert scores[2] == 0.0  # cold 从未共现


def test_collaborative_cold_start_returns_unavailable():
    # 无共现数据 / 无近期收听 → available=False
    scores, ok = collaborative_scores(["a", "b"], recent_item_ids=[], cooccurrence={})
    assert ok is False
    assert scores == [0.0, 0.0]
    scores2, ok2 = collaborative_scores(["a"], recent_item_ids=["x"], cooccurrence={"q": {"r": 3}})
    assert ok2 is False  # 候选与 recent 无任何共现


def test_recent_listened_ids_recent_first_deduped():
    history = [_Ev("a"), _Ev("b"), _Ev("a"), _Ev("c")]
    ids = recent_listened_ids(history, limit=10)
    assert ids == ["c", "a", "b"]  # 最近优先、去重


# ---- 4th anchor 权重重分配 ----


def test_weights_redistribute_without_cf():
    # CF 不可用时，三锚权重和仍为 1，CF 权重为 0
    w_sem, w_per, w_beh, w_col, w_exp = _normalized_weights(True, True, collaborative_ok=False)
    assert w_col == 0.0
    assert w_exp == 0.0
    assert abs(w_sem + w_per + w_beh - 1.0) < 1e-9


def test_cf_anchor_changes_ranking_when_enabled(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "tri_anchor_w_collaborative", 0.5, raising=False)
    taste = TasteProfile(top_genres=[("pop", 1.0)])
    profile = PreferenceProfile.from_taste(taste)
    # 两首标签相同的候选，仅 CF 分不同 → CF 高者应排前
    tracks = [_track("low", ["pop"]), _track("high", ["pop"])]
    ranked = tri_anchor_rerank(
        "x",
        tracks,
        profile,
        collaborative_scores=[0.0, 1.0],
        collaborative_ok=True,
    )
    assert ranked[0][0].title == "high"
    assert "collaborative" in ranked[0][1].components


def test_cf_ignored_when_length_mismatch():
    # CF 分数与候选数不等 → 安全忽略，不崩
    profile = PreferenceProfile.from_taste(None)
    tracks = [_track("a"), _track("b")]
    ranked = tri_anchor_rerank(
        "x",
        tracks,
        profile,
        collaborative_scores=[1.0],
        collaborative_ok=True,  # 长度不匹配
    )
    assert len(ranked) == 2
    assert "collaborative" not in ranked[0][1].components
