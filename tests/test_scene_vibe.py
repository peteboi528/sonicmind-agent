"""scene_vibe：场景 vibe 自动判官的机制测试。

vibe 判别的真实质量（能否区分深夜/下午 R&B）由 P1 eval 度量，不在此手工断言；
这里只锁机制：场景识别、max-cosine 打分数学、embedding 不可用安全降级。
"""
from __future__ import annotations

import pytest

import app.recommend.scene_vibe as sv

# ---- 场景识别 ----

def test_detect_time_of_day_and_scenes():
    assert sv.detect_scene_vibe("推荐几首适合深夜的歌") == "深夜"
    assert sv.detect_scene_vibe("夜晚一个人听") == "深夜"
    assert sv.detect_scene_vibe("适合下午听的") == "下午"
    assert sv.detect_scene_vibe("午后放松") == "下午"
    assert sv.detect_scene_vibe("早晨起床") == "早晨"
    assert sv.detect_scene_vibe("跑步 健身") == "运动"
    assert sv.detect_scene_vibe("学习专注") == "学习"
    assert sv.detect_scene_vibe("睡前助眠") == "睡眠"


def test_detect_non_scene_returns_none():
    assert sv.detect_scene_vibe("推荐几首歌") is None
    assert sv.detect_scene_vibe("Taylor Swift 的歌") is None
    assert sv.detect_scene_vibe("") is None


# ---- 打分数学（mock encode，确定性）----

def test_scores_are_max_cosine_over_prototypes(monkeypatch):
    """候选向量 == 某原型 → 1.0；与所有原型正交 → 0.5。"""
    e0, e1 = [1.0, 0.0], [0.0, 1.0]
    # 深夜有 4 个原型，全置为 e0；两个候选分别 e0（完美契合）、e1（正交）
    monkeypatch.setattr(sv, "embeddings_available", lambda: True)
    monkeypatch.setattr(sv, "encode", lambda texts: [e0] * 4 + [e0, e1])
    scores = sv.scene_vibe_scores(["track-a", "track-b"], "深夜")
    assert scores == [1.0, 0.5]


def test_score_in_zero_one_and_single_helper(monkeypatch):
    monkeypatch.setattr(sv, "embeddings_available", lambda: True)
    e0 = [1.0, 0.0]
    monkeypatch.setattr(sv, "encode", lambda texts: [e0] * 4 + [e0])
    assert sv.scene_vibe_score("x", "深夜") == 1.0


# ---- 安全降级 ----

def test_returns_none_when_embeddings_unavailable(monkeypatch):
    monkeypatch.setattr(sv, "embeddings_available", lambda: False)
    assert sv.scene_vibe_scores(["x"], "深夜") is None
    assert sv.scene_vibe_score("x", "深夜") is None


def test_returns_none_for_empty_or_unknown_scene(monkeypatch):
    monkeypatch.setattr(sv, "embeddings_available", lambda: True)
    monkeypatch.setattr(sv, "encode", lambda texts: [])
    assert sv.scene_vibe_scores([], "深夜") is None          # 空候选
    assert sv.scene_vibe_scores(["x"], "不存在的场景") is None  # 无原型


def test_returns_none_when_encode_fails(monkeypatch):
    """encode 返回 None（模型加载失败等）时安全降级，不抛。"""
    monkeypatch.setattr(sv, "embeddings_available", lambda: True)
    monkeypatch.setattr(sv, "encode", lambda texts: None)
    assert sv.scene_vibe_scores(["x"], "深夜") is None


# ---- 对比式打分（scene_vibe_penalty，rerank 实际用的入口）----

def test_penalty_contrastive_for_time_scene(monkeypatch):
    """时段场景用对比式 fit_scene − fit_anti，threshold=0.0（负=偏 anti-scene）。"""
    monkeypatch.setattr(sv, "embeddings_available", lambda: True)
    monkeypatch.setattr(sv, "scene_vibe_scores",
                        lambda texts, scene: {"深夜": [0.7, 0.6], "下午": [0.5, 0.8]}.get(scene))
    fits, threshold = sv.scene_vibe_penalty(["a", "b"], "深夜")
    assert threshold == 0.0
    assert fits[0] == pytest.approx(0.2, abs=1e-6)    # 0.7−0.5：偏深夜
    assert fits[1] == pytest.approx(-0.2, abs=1e-6)   # 0.6−0.8：偏下午 → 深夜里 <0 应降权


def test_penalty_positive_for_non_time_scene(monkeypatch):
    """运动等无干净反义的场景用正向 fit，threshold=0.45。"""
    monkeypatch.setattr(sv, "embeddings_available", lambda: True)
    monkeypatch.setattr(sv, "scene_vibe_scores", lambda texts, scene: [0.3, 0.6])
    fits, threshold = sv.scene_vibe_penalty(["a", "b"], "运动")
    assert threshold == 0.45
    assert fits == [0.3, 0.6]


def test_penalty_degrades_when_embeddings_off(monkeypatch):
    """embedding 不可用 → (None, 0.0)，调用方据此跳过。"""
    monkeypatch.setattr(sv, "embeddings_available", lambda: False)
    fits, threshold = sv.scene_vibe_penalty(["a"], "深夜")
    assert fits is None
