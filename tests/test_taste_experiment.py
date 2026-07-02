from __future__ import annotations

import pytest

from app.agent import AudioVisualAgent
from app.models import (
    ExternalTrack,
    TasteExperimentFeedbackRequest,
    TasteExperimentTrack,
    TrackRef,
)
from app.rules.taste_experiment import (
    bucket_taste_experiment_candidates,
    taste_experiment_track_key,
    taste_familiarity,
)
from app.services.taste_experiment import TasteExperimentService
from app.storage import JsonStore


@pytest.fixture
def agent(tmp_path):
    return AudioVisualAgent(JsonStore(tmp_path / "store"))


def test_generate_taste_experiment_has_three_buckets(agent):
    exp = agent.generate_taste_experiment("taste-user", "推荐点不一样的，做个品味实验", total=9)

    assert exp.experiment_id.startswith("taste_")
    assert [segment.name for segment in exp.segments] == ["safe", "stretch", "bold"]
    assert sum(len(segment.tracks) for segment in exp.segments) >= 3
    assert agent.get_taste_experiment("taste-user", exp.experiment_id) is not None


def test_taste_experiment_feedback_and_report(agent):
    exp = agent.generate_taste_experiment("feedback-user", "探索我的口味", total=9)
    items = [item for segment in exp.segments for item in segment.tracks]
    assert items

    for index, item in enumerate(items[:6]):
        signal = "liked" if index % 2 == 0 else "completed"
        req = TasteExperimentFeedbackRequest(
            user_id="feedback-user",
            experiment_id=exp.experiment_id,
            track_key=taste_experiment_track_key(item),
            signal=signal,
        )
        exp = agent.record_taste_experiment_feedback(req)

    assert exp.status in {"ready", "reported"}
    report = agent.summarize_taste_experiment("feedback-user", exp.experiment_id)
    assert report.bucket_stats
    assert report.summary
    assert report.next_recommendation_strategy


def test_taste_experiment_report_waits_for_more_feedback(agent):
    exp = agent.generate_taste_experiment("cold-user", "推荐点不一样的", total=9)
    report = agent.summarize_taste_experiment("cold-user", exp.experiment_id)

    assert "继续" in report.hypothesis_result
    assert "反馈" in report.summary


def test_delete_taste_experiment(agent):
    """collecting 等任意状态的实验都能删除；删后读不回来，重复删返回 False。"""
    exp = agent.generate_taste_experiment("del-user", "探索口味", total=9)
    assert agent.get_taste_experiment("del-user", exp.experiment_id) is not None

    assert agent.delete_taste_experiment("del-user", exp.experiment_id) is True
    assert agent.get_taste_experiment("del-user", exp.experiment_id) is None
    assert agent.delete_taste_experiment("del-user", exp.experiment_id) is False  # 已删，幂等


# ---- 锚复活后的分桶与反馈回流 ----

def _cand(personalize, semantic, behavior=0.0):
    """合成一个候选 tuple：(track, components, reason, score)。"""
    return (None, {"personalize": personalize, "semantic": semantic, "behavior": behavior}, "r", 0.0)


def test_bucketing_splits_by_familiarity_into_three_even_buckets(agent):
    """重做后的分桶：按 familiarity 排名切成均衡三档，safe 最像口味、bold 最探索。

    回归旧 bug：旧行为锚死后阈值分桶把三档全塌向 bold。
    """
    candidates = [
        _cand(0.9, 0.5), _cand(0.8, 0.4),
        _cand(0.5, 0.3), _cand(0.4, 0.2),
        _cand(0.2, 0.1), _cand(0.1, 0.0),
    ]
    buckets = bucket_taste_experiment_candidates(candidates, 2)
    assert len(buckets["safe"]) == 2
    assert len(buckets["stretch"]) == 2
    assert len(buckets["bold"]) == 2
    safe_fam = taste_familiarity(buckets["safe"][0])
    bold_fam = taste_familiarity(buckets["bold"][0])
    assert safe_fam > bold_fam


def test_bucketing_refuses_fake_labels_when_scores_are_indistinguishable(agent):
    candidates = [
        _cand(0.05, 0.05), _cand(0.04, 0.05), _cand(0.04, 0.04),
        _cand(0.03, 0.04), _cand(0.03, 0.03), _cand(0.02, 0.03),
    ]
    buckets = bucket_taste_experiment_candidates(candidates, 2)
    assert buckets["safe"] == []
    assert buckets["bold"] == []
    assert len(buckets["stretch"]) == 6


def test_prompt_negative_constraints_become_hard_rules(agent):
    rules = TasteExperimentService.taste_prompt_exclusions("推荐点不一样的，别太吵，不要 Type Beat")
    assert "激昂" in rules
    assert "type beat" in rules


def test_taste_feedback_feeds_listening_history(agent):
    """实验反馈写进 listening_history（key=source_id），打通「反馈→行为锚→下一轮」闭环。

    回归旧 bug：实验 completed/skipped 只进 TS 反馈库，从不喂行为锚，实验无法自我学习。
    """
    online = TasteExperimentTrack(
        track=TrackRef(title="T", artist="A", source="netease", source_id="netease-123"),
        bucket="safe",
    )
    local = TasteExperimentTrack(track=TrackRef(title="L", source="local"), bucket="safe")

    before = len(agent.memory.get_memory("u").listening_history)
    agent._record_taste_experiment_listen("u", online, "completed", None)
    assert len(agent.memory.get_memory("u").listening_history) == before + 1
    ev = agent.memory.get_memory("u").listening_history[-1]
    assert ev.asset_id == "netease-123"  # key = source_id，与候选 _track_id 同命名空间
    assert ev.completed is True

    agent._record_taste_experiment_listen("u", online, "skipped", None)
    assert agent.memory.get_memory("u").listening_history[-1].completed is False

    n = len(agent.memory.get_memory("u").listening_history)
    agent._record_taste_experiment_listen("u", local, "completed", None)
    assert len(agent.memory.get_memory("u").listening_history) == n  # 无在线 id 的曲不记


# ---- 探索页：库外优先候选 ----

def test_collect_candidates_online_only_drops_local_and_listened(agent, monkeypatch):
    """online_only=True：跳过库内推荐路径，只走 web 搜索；剔除本地曲与已听过的曲目。

    回归根因：旧行为里本地曲 personalize 分天然偏高，三档会被库内歌占满，
    探索页失去"发现库外新歌/新歌手"的意义。
    """
    svc = agent._taste_experiment_service()

    local_track = ExternalTrack(external_id="loc-1", title="本地曲", artist="A", source="local")
    listened = ExternalTrack(external_id="netease-listened", title="听过的", artist="B", source="netease")
    fresh = ExternalTrack(external_id="netease-fresh", title="全新曲", artist="C", source="netease")

    # 库内推荐路径若被调用就报错——online_only 必须完全跳过它。
    def _boom(*args, **kwargs):
        raise AssertionError("online_only 不应调用库内推荐路径")

    monkeypatch.setattr(svc, "_recommend_for_query", _boom)
    monkeypatch.setattr(svc, "_search_web_music", lambda *a, **k: [local_track, listened, fresh])
    monkeypatch.setattr(svc, "_is_recommendation_quality_track", lambda t: True)
    monkeypatch.setattr(svc, "_dedupe_tracks", lambda tracks: tracks)

    class _BD:
        components = {"semantic": 0.5, "personalize": 0.0, "behavior": 0.0}
        reason = "r"
        score = 0.5

    monkeypatch.setattr(svc, "_rerank_tracks", lambda u, q, tracks, top_k: [(t, _BD()) for t in tracks])

    # 标记 "听过的" 已在听歌历史里
    agent.memory.record_listen("explore-user", "netease-listened", 200, True, context="player")

    result = svc.collect_taste_candidates("explore-user", ["neo soul"], 9, online_only=True)
    titles = {getattr(t, "title", "") for t, *_ in result}
    assert "全新曲" in titles
    assert "本地曲" not in titles      # source==local 被剔除
    assert "听过的" not in titles      # 已在 listening_history 被剔除

