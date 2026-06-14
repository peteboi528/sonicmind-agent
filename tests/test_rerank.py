"""Phase 1 精排回归测试：三锚归一化、缺锚重分配、MMR 去重、Thompson 衰减/反馈环。"""

from __future__ import annotations

import statistics
import tempfile
from pathlib import Path

import pytest

from app.library import ResourceLibrary
from app.models import ExternalTrack, TasteProfile
from app.recommend.rerank import (
    PreferenceProfile,
    _normalized_weights,
    mmr_rerank,
    rerank_candidates,
    tri_anchor_rerank,
)


def _track(title, genre, mood, source="netease", ext_id=None):
    return ExternalTrack(
        external_id=ext_id or title, title=title, artist="A",
        genre=genre, mood=mood, source=source,
    )


# ---- 三锚归一化 ----

def test_tri_anchor_weights_sum_to_one():
    w_sem, w_per, w_beh = _normalized_weights(semantic_ok=True, behavior_ok=True)
    assert abs(w_sem + w_per + w_beh - 1.0) < 1e-9


def test_missing_anchors_reweighted():
    # 无语义模型 + 无行为数据 → 全部权重落到个性化锚
    w_sem, w_per, w_beh = _normalized_weights(semantic_ok=False, behavior_ok=False)
    assert w_sem == 0.0 and w_beh == 0.0
    assert abs(w_per - 1.0) < 1e-9


def test_personalize_anchor_prefers_matching_tags():
    taste = TasteProfile(top_genres=[("R&B", 3.0)], top_moods=[("放松", 2.0)])
    tracks = [
        _track("Match", ["R&B"], ["放松"]),
        _track("Miss", ["摇滚"], ["激昂"]),
    ]
    ranked = tri_anchor_rerank("chill", tracks, PreferenceProfile.from_taste(taste))
    assert ranked[0][0].title == "Match"
    assert ranked[0][1].components["personalize"] > ranked[1][1].components["personalize"]


def test_breakdown_has_components():
    taste = TasteProfile(top_genres=[("R&B", 1.0)], top_moods=[("放松", 1.0)])
    ranked = tri_anchor_rerank("x", [_track("S", ["R&B"], ["放松"])], PreferenceProfile.from_taste(taste))
    comp = ranked[0][1].components
    assert {"semantic", "personalize", "behavior", "w_semantic"} <= set(comp)


# ---- MMR 多样性 ----

def test_mmr_promotes_diversity():
    taste = TasteProfile(top_genres=[("R&B", 1.0), ("电子", 0.8)], top_moods=[("放松", 1.0), ("激昂", 0.8)])
    # 三首同质（都 R&B/放松），一首相关但不同风格（电子/激昂，也在口味里）
    tracks = [
        _track("RB1", ["R&B"], ["放松"]),
        _track("RB2", ["R&B"], ["放松"]),
        _track("RB3", ["R&B"], ["放松"]),
        _track("EDM", ["电子"], ["激昂"]),
    ]
    scored = tri_anchor_rerank("chill", tracks, PreferenceProfile.from_taste(taste))
    diversified = mmr_rerank(scored, top_k=2, lambda_=0.5)
    titles = [t.title for t, _ in diversified]
    # 第二首应换成相关但异质的 EDM，而非又一首同质 R&B
    assert "EDM" in titles


def test_mmr_does_not_let_irrelevant_leapfrog_relevant():
    """回归：低相关的噪声候选不得靠'多样性'反超明显更相关的候选。

    复现用户 bug——上传摇滚后，无关垃圾被 MMR 插到真实摇滚候选前面。
    """
    taste = TasteProfile(top_genres=[("摇滚", 1.0)], top_moods=[("励志", 1.0)])
    tracks = [
        _track("英伦摇滚合集", ["摇滚"], ["励志"]),
        _track("辣条史诗噪声", [], []),  # 零相关垃圾
        _track("摇滚现场", ["摇滚"], ["励志"]),
    ]
    scored = tri_anchor_rerank("摇滚", tracks, PreferenceProfile.from_taste(taste))
    diversified = mmr_rerank(scored, top_k=3, lambda_=0.7)
    titles = [t.title for t, _ in diversified]
    # 两首摇滚都必须排在垃圾之前
    assert titles.index("辣条史诗噪声") == 2


def test_rerank_candidates_respects_top_k():
    taste = TasteProfile(top_genres=[("R&B", 1.0)], top_moods=[("放松", 1.0)])
    tracks = [_track(f"S{i}", ["R&B"], ["放松"], ext_id=f"id{i}") for i in range(6)]
    out = rerank_candidates("chill", tracks, taste, top_k=3)
    assert len(out) == 3


# ---- Thompson Sampling ----

@pytest.fixture
def lib():
    d = tempfile.mkdtemp()
    return ResourceLibrary(Path(d) / "lib.sqlite")


def test_ts_positive_feedback_raises_score(lib):
    t = _track("Hit", ["R&B"], ["放松"])
    lib.upsert_external(t)
    before = statistics.mean(list(lib.sample_ts_scores([t]).values())[0] for _ in range(300))
    lib.update_ts_feedback(t, positive=True, weight=4.0)
    after = statistics.mean(list(lib.sample_ts_scores([t]).values())[0] for _ in range(300))
    assert after > before + 0.1


def test_ts_negative_feedback_lowers_score(lib):
    t = _track("Skip", ["R&B"], ["放松"])
    lib.upsert_external(t)
    before = statistics.mean(list(lib.sample_ts_scores([t]).values())[0] for _ in range(300))
    lib.update_ts_feedback(t, positive=False, weight=4.0)
    after = statistics.mean(list(lib.sample_ts_scores([t]).values())[0] for _ in range(300))
    assert after < before - 0.1


def test_ts_exposure_decay_lowers_prior(lib):
    t = _track("Exposed", ["R&B"], ["放松"])
    lib.upsert_external(t)
    before = statistics.mean(list(lib.sample_ts_scores([t]).values())[0] for _ in range(300))
    for _ in range(5):
        lib.decay_exposure_ts([t])
    after = statistics.mean(list(lib.sample_ts_scores([t]).values())[0] for _ in range(300))
    assert after < before


def test_ts_sample_for_unknown_track_is_uniform(lib):
    t = _track("Unknown", ["R&B"], ["放松"])
    score = list(lib.sample_ts_scores([t]).values())[0]
    assert 0.0 <= score <= 1.0
