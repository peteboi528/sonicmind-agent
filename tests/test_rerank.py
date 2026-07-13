"""Phase 1 精排回归测试：三锚归一化、缺锚重分配、MMR 去重、Thompson 衰减/反馈环。"""

from __future__ import annotations

import statistics
import tempfile
from pathlib import Path

import pytest

from app.library import ResourceLibrary
from app.models import ExternalTrack, RankingBreakdown, TasteProfile
from app.recommend.rerank import (
    PreferenceProfile,
    _normalized_weights,
    apply_profile_artist_adjust,
    bandit_select,
    mmr_rerank,
    rerank_candidates,
    tri_anchor_rerank,
)


def _track(title, genre, mood, source="netease", ext_id=None):
    return ExternalTrack(
        external_id=ext_id or title,
        title=title,
        artist="A",
        genre=genre,
        mood=mood,
        source=source,
    )


def _track_artist(title: str, artist: str) -> ExternalTrack:
    return ExternalTrack(external_id=title, title=title, artist=artist, genre=["pop"], mood=["放松"], source="netease")


def _bd(score: float) -> RankingBreakdown:
    return RankingBreakdown(title="t", source="netease", score=score, reason="x", components={})


# ---- 四锚归一化（默认 CF 不可用，退回三锚） ----


def test_tri_anchor_weights_sum_to_one():
    w_sem, w_per, w_beh, w_col, w_exp = _normalized_weights(semantic_ok=True, behavior_ok=True)
    assert abs(w_sem + w_per + w_beh + w_col + w_exp - 1.0) < 1e-9
    assert w_col == 0.0  # 默认 collaborative_ok=False，CF 权重不分配
    assert w_exp == 0.0  # 默认 explore_ok=False，探索锚不分配


def test_missing_anchors_reweighted():
    # 无语义模型 + 无行为数据 + 无 CF → 全部权重落到个性化锚
    w_sem, w_per, w_beh, w_col, w_exp = _normalized_weights(semantic_ok=False, behavior_ok=False)
    assert w_sem == 0.0 and w_beh == 0.0 and w_col == 0.0 and w_exp == 0.0
    assert abs(w_per - 1.0) < 1e-9


def test_collaborative_anchor_takes_share_when_enabled():
    # 四锚全可用时，CF 锚分到非零权重，四者归一化和为 1
    w_sem, w_per, w_beh, w_col, w_exp = _normalized_weights(semantic_ok=True, behavior_ok=True, collaborative_ok=True)
    assert w_col > 0.0
    assert w_exp == 0.0
    assert abs(w_sem + w_per + w_beh + w_col + w_exp - 1.0) < 1e-9


def test_explore_anchor_takes_share_when_enabled():
    w_sem, w_per, w_beh, w_col, w_exp = _normalized_weights(
        semantic_ok=True, behavior_ok=True, collaborative_ok=True, explore_ok=True
    )
    assert w_exp > 0.0
    assert abs(w_sem + w_per + w_beh + w_col + w_exp - 1.0) < 1e-9


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


def test_explore_components_added_when_ts_scores_exist():
    taste = TasteProfile(top_genres=[("R&B", 1.0)], top_moods=[("放松", 1.0)])
    tracks = [_track("S", ["R&B"], ["放松"], ext_id="s1")]
    ts_key = "netease|s1|s|a"
    ranked = tri_anchor_rerank(
        "chill",
        tracks,
        PreferenceProfile.from_taste(taste),
        ts_scores={ts_key: 0.92},
    )
    comp = ranked[0][1].components
    assert comp["explore"] == 0.92
    assert comp["w_explore"] > 0.0


# ---- 锚复活回归（旧 bug：语义/行为在生产恒为 0，三锚退化成单锚）----


def test_semantic_tf_fallback_is_live_anchor():
    """无 sentence-transformers 时 TF 兜底不再被清零——语义锚真正参与精排。

    回归旧 bug：_semantic_anchor 在 TF 兜底时返回 available=False，_normalized_weights
    随即把 w_semantic 清零、正确的语义匹配（如 query「说唱」命中说唱曲）被整体丢弃。
    """
    from app.recommend.rerank import _semantic_anchor

    tracks = [_track("RapTrack", ["说唱"], ["激昂"], ext_id="r1")]
    _, ok = _semantic_anchor("说唱 hip hop", tracks)
    assert ok is True  # TF 兜底现在是有效锚
    ranked = tri_anchor_rerank("说唱", tracks, PreferenceProfile.from_taste(TasteProfile(top_genres=[("流行", 1.0)])))
    assert ranked[0][1].components["w_semantic"] > 0.0  # 语义权重不再被清零


def test_behavior_anchor_moves_ranking_when_data_exists():
    """注入收听数据后行为锚有权重、能改变排序。

    回归旧 bug：前端从不调 /listen → listening_history 恒空 → behavior_scores 恒空
    → 行为锚 available=False → w_behavior 恒为 0，从未改变过任何推荐。
    """
    from app.recommend.rerank import _behavior_anchor

    # 两首口味/语义同分，但一首被反复听完(+行为)、一首被秒跳(-行为)
    tracks = [
        _track("Finished", ["R&B"], ["放松"], ext_id="fin"),
        _track("Skipped", ["R&B"], ["放松"], ext_id="skip"),
    ]
    behavior = {"fin": 3.0, "skip": -3.0}  # key = external_id，与候选 _track_id 同命名空间
    _, ok = _behavior_anchor(tracks, behavior)
    assert ok is True
    taste = TasteProfile(top_genres=[("R&B", 1.0)], top_moods=[("放松", 1.0)])
    ranked = tri_anchor_rerank("chill", tracks, PreferenceProfile.from_taste(taste), behavior_scores=behavior)
    assert ranked[0][1].components["w_behavior"] > 0.0
    assert ranked[0][0].title == "Finished"  # 听完的行为分高，排到前面


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


def test_bandit_select_reserves_explore_slot():
    taste = TasteProfile(top_genres=[("R&B", 1.0)], top_moods=[("放松", 1.0)])
    tracks = [
        _track("Safe1", ["R&B"], ["放松"], ext_id="safe1"),
        _track("Safe2", ["R&B"], ["放松"], ext_id="safe2"),
        _track("Explorer", ["电子"], ["梦幻"], ext_id="explore"),
    ]
    ts_scores = {
        "netease|safe1|safe1|a": 0.1,
        "netease|safe2|safe2|a": 0.1,
        "netease|explore|explorer|a": 0.99,
    }
    scored = tri_anchor_rerank("chill", tracks, PreferenceProfile.from_taste(taste), ts_scores=ts_scores)
    selected = bandit_select(scored, top_k=2, explore_ratio=0.5)
    assert "Explorer" in [track.title for track, _ in selected]


def test_bandit_select_ratio():
    scored = []
    for i in range(10):
        is_explorer = i >= 7
        track = _track(f"T{i}", ["R&B" if not is_explorer else "电子"], ["放松"], ext_id=f"id{i}")
        scored.append(
            (
                track,
                RankingBreakdown(
                    title=track.title,
                    source=track.source,
                    score=round(1.0 - i * 0.05, 4),
                    reason="test",
                    components={"explore": 0.99 if is_explorer else 0.01},
                ),
            )
        )
    selected = bandit_select(scored, top_k=10, explore_ratio=0.3)
    titles = [track.title for track, _ in selected]
    assert {"T7", "T8", "T9"} <= set(titles)


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


# ---- 画像艺人信号（core/rising 加分、avoid 减分）----


def test_profile_artist_adjust_boosts_core_and_penalizes_avoid():
    """同分候选：画像 core 艺人升到最前，avoid 艺人沉到最后。"""
    a = _track_artist("Song A", "Core Artist")
    b = _track_artist("Song B", "Neutral Artist")
    c = _track_artist("Song C", "Avoided Artist")
    scored = [(a, _bd(0.5)), (b, _bd(0.5)), (c, _bd(0.5))]
    out = apply_profile_artist_adjust(scored, boost={"Core Artist"}, penalty={"Avoided Artist"})
    assert [t.title for t, _ in out] == ["Song A", "Song B", "Song C"]
    assert out[0][1].components.get("profile_core_artist", 0) > 0
    assert out[-1][1].components.get("profile_avoid_artist", 0) < 0


def test_profile_artist_adjust_substring_match_on_compound_artist():
    """候选 artist 形如「A、B」，画像艺人作为子串命中即生效。"""
    t = _track_artist("Song", "Mac Miller、Asen")
    scored = [(t, _bd(0.4))]
    out = apply_profile_artist_adjust(scored, boost={"Mac Miller"}, penalty=None)
    assert out[0][1].score > 0.4
    assert "profile_core_artist" in out[0][1].components


def test_profile_artist_adjust_noop_without_signals():
    """boost/penalty 为空时不改分，行为与旧版一致。"""
    t = _track_artist("Song", "X")
    out = apply_profile_artist_adjust([(t, _bd(0.5))], None, None)
    assert out[0][1].score == 0.5
    assert "profile_core_artist" not in out[0][1].components


def test_rerank_candidates_accepts_profile_signals_without_crash():
    """rerank_candidates 新增的画像参数为可选：传与不传都不崩。"""
    tracks = [_track_artist("Song", "Core Artist"), _track_artist("Other", "Nobody")]
    taste = TasteProfile()
    # 不传画像（旧行为）
    rerank_candidates("test", tracks, taste, top_k=2, apply_mmr=False)
    # 传画像（加分生效、不崩）
    out = rerank_candidates(
        "test",
        tracks,
        taste,
        top_k=2,
        apply_mmr=False,
        profile_boost_artists={"Core Artist"},
        profile_penalty_artists=None,
    )
    assert len(out) == 2


# ---- 场景 vibe 微调（深夜 query 压低下午向候选等）----


def test_scene_vibe_adjust_penalizes_off_vibe_in_scene_query(monkeypatch):
    """深夜 query 下，vibe 偏离场景的候选（对比值低于阈值）被降分、排到后面。"""
    from app.recommend import rerank as rerank_mod

    a = _track_artist("Night Track", "Night Artist")
    b = _track_artist("Sunny Afternoon", "Day Artist")
    # mock scene_vibe_penalty：(fits, threshold)；a=0.8（高）、b=0.2（低，<0.5 阈值 → 降分）
    monkeypatch.setattr(
        rerank_mod, "scene_vibe_penalty", lambda texts, scene: ([0.8, 0.2], 0.5) if scene == "深夜" else (None, 0.0)
    )
    out = rerank_mod.apply_scene_vibe_adjust([(a, _bd(0.5)), (b, _bd(0.5))], "推荐几首适合深夜的歌")
    assert out[0][0].title == "Night Track"
    assert out[1][0].title == "Sunny Afternoon"
    assert out[1][1].components.get("scene_vibe_penalty") == -0.08


def test_scene_vibe_adjust_noop_without_scene(monkeypatch):
    """非场景 query 不启用 vibe 判别，分数不变。"""
    from app.recommend import rerank as rerank_mod

    t = _track_artist("X", "Y")
    out = rerank_mod.apply_scene_vibe_adjust([(t, _bd(0.5))], "推荐几首歌")
    assert out[0][1].score == 0.5


def test_scene_vibe_adjust_noop_when_embeddings_unavailable(monkeypatch):
    """embedding 不可用时安全降级，不改分（生产里模型没装也不影响推荐）。"""
    from app.recommend import rerank as rerank_mod

    monkeypatch.setattr(rerank_mod, "scene_vibe_penalty", lambda texts, scene: (None, 0.0))
    t = _track_artist("X", "Y")
    out = rerank_mod.apply_scene_vibe_adjust([(t, _bd(0.5))], "推荐几首适合深夜的歌")
    assert out[0][1].score == 0.5
