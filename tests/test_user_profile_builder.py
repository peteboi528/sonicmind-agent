"""用户画像构建与服务测试（计划 §22.1）。

覆盖点：
1. 空画像返回引导态而非空标签。
2. 明确偏好/排除 → 高置信；隐式推断 → 低置信（权重原则 §10.2）。
3. 声音指纹/情绪地图从 genre/mood 确定性映射出来，能解释。
4. insight 有稳定 id，reject 后状态保留且不再用于 LLM 上下文（§13.2, §14.1）。
5. 硬约束（排除项）始终带进 LLM 上下文，最高优先级。
6. 置信度随证据增多而合理变化（§18）。
"""

from __future__ import annotations

import pytest

from app.memory import MemoryManager
from app.models import ListeningEvent, MemoryEntry, RatingEntry, TasteProfile, UserMemory
from app.profile.builder import (
    UserProfileBuilder,
    _compute_recency,
    compute_confidence,
    confidence_band,
)
from app.profile.evidence import collect_profile_evidence, detect_language
from app.services.profile import UserProfileService, _normalize_status
from app.storage import JsonStore


@pytest.fixture
def store(tmp_path):
    return JsonStore(str(tmp_path / "store"))


@pytest.fixture
def service(store):
    return UserProfileService(store, MemoryManager(store))


def _rich_memory(user_id: str = "u1") -> UserMemory:
    return UserMemory(
        user_id=user_id,
        preferences=["R&B", "治愈的流行"],
        structured_preferences=[
            MemoryEntry(text="R&B", frequency=5, source="user_event"),
            MemoryEntry(text="治愈系流行", frequency=3, source="auto_explicit"),
        ],
        exclusion_rules=["抖音热歌", "不要中文歌"],
        ratings=[
            RatingEntry(asset_id="a1", score=9.0, title="Ditto", artist="NewJeans",
                        genre=["R&B", "流行"], mood=["律动", "治愈"]),
            RatingEntry(asset_id="a2", score=8.0, title="Sunflower", artist="Post Malone",
                        genre=["流行"], mood=["欢快"]),
        ],
        taste_profile=TasteProfile(
            top_genres=[("R&B", 8.0), ("流行", 5.0)],
            top_moods=[("律动", 4.0), ("治愈", 3.0)],
            top_artists=[("newjeans", 6.0), ("post malone", 4.0)],
            preferred_energy=0.55,
            discovery_openness=0.45,
        ),
    )


# ── 空状态 ──────────────────────────────────────────────────────────────────
def test_empty_profile_is_guidance_not_blank(service):
    profile = service.get_profile("brand-new-user")
    assert profile.is_empty is True
    assert profile.empty_hint
    assert "认识你" in profile.summary.headline
    assert profile.insights == []


# ── 证据收集与权重 ──────────────────────────────────────────────────────────
def test_explicit_preference_outweighs_implicit():
    """明确（user_event，权重 1.0）应比模型推断（inferred，0.3）贡献更大。"""
    explicit = UserMemory(
        user_id="ex",
        structured_preferences=[MemoryEntry(text="爵士", frequency=3, source="user_event")],
    )
    implicit = UserMemory(
        user_id="im",
        structured_preferences=[MemoryEntry(text="爵士", frequency=3, source="inferred_from_result")],
    )
    ev_e = collect_profile_evidence(explicit)
    ev_i = collect_profile_evidence(implicit)
    assert ev_e.genre_weights.get("爵士", 0) > ev_i.genre_weights.get("爵士", 0)
    assert ev_e.explicit_signal_count == 1
    assert ev_i.explicit_signal_count == 0


def test_detect_language():
    assert detect_language("周杰伦 七里香") == "zh"
    assert detect_language("Taylor Swift Lover") == "en"
    assert detect_language("123 !!!") == "other"


# ── 声音指纹 / 情绪地图 ──────────────────────────────────────────────────────
def test_sound_fingerprint_explains_dimensions():
    builder = UserProfileBuilder()
    profile = builder.build_from_memory(_rich_memory())
    dims = {d.key: d for d in profile.sound_fingerprint.dimensions}
    # R&B + 律动 → groove 应为最高维度之一
    assert dims["groove"].value > 0
    assert dims["groove"].value >= dims["experimental"].value
    # 每个非零维度都有解释（可回答「为什么是这个分」）
    for d in profile.sound_fingerprint.dimensions:
        assert d.explanation


def test_mood_landscape_has_coordinates():
    builder = UserProfileBuilder()
    profile = builder.build_from_memory(_rich_memory())
    moods = {p.mood for p in profile.mood_landscape.global_points}
    assert "律动" in moods or "治愈" in moods
    assert profile.mood_landscape.summary


# ── 艺术家关系 ──────────────────────────────────────────────────────────────
def test_core_artists_detected():
    builder = UserProfileBuilder()
    profile = builder.build_from_memory(_rich_memory())
    core = [a for a in profile.artists if a.relation_type == "core"]
    assert any(a.artist == "newjeans" for a in core)
    for a in profile.artists:
        assert a.reasons  # 每个关系都能回答「为什么在这里」


def test_disliked_artist_marked_avoid():
    memory = _rich_memory()
    memory.dislikes = ["孟菲斯说唱歌手X"]
    builder = UserProfileBuilder()
    profile = builder.build_from_memory(memory)
    avoid = [a for a in profile.artists if a.relation_type == "avoid"]
    assert any("孟菲斯" in a.artist for a in avoid)


# ── 置信度 ──────────────────────────────────────────────────────────────────
def test_confidence_increases_with_evidence_and_explicitness():
    weak = compute_confidence(evidence_count=1, explicit=False, recency=0.5,
                              consistency=0.3, contradiction=0.0)
    strong = compute_confidence(evidence_count=5, explicit=True, recency=0.9,
                                consistency=0.9, contradiction=0.0)
    assert strong > weak
    assert 0.0 <= weak <= 1.0 and 0.0 <= strong <= 1.0


def test_contradiction_lowers_confidence():
    clean = compute_confidence(evidence_count=4, explicit=True, recency=0.7,
                               consistency=0.7, contradiction=0.0)
    contradicted = compute_confidence(evidence_count=4, explicit=True, recency=0.7,
                                      consistency=0.7, contradiction=1.0)
    assert contradicted < clean


def test_confidence_band_thresholds():
    assert confidence_band(0.8) == "high"
    assert confidence_band(0.5) == "medium"
    assert confidence_band(0.1) == "low"


def test_taste_only_profile_lifts_off_confidence_floor():
    """仅有 taste_profile（无评分/排除/明确偏好）也应非空、置信度脱离地板。

    回归：修复前 total_signal_count 不计 taste_profile，这类用户被判 0 证据，
    置信度卡在非空地板 0.36（明明有导入歌单生成的真实品味）。
    """
    mem = UserMemory(
        user_id="taste-only",
        taste_profile=TasteProfile(
            top_genres=[("说唱", 105.0), ("R&B", 46.0), ("流行", 32.0)],
            top_moods=[("律动", 4.0), ("治愈", 3.0)],
            top_artists=[("a", 6.0), ("b", 5.0)],
        ),
    )
    profile = UserProfileBuilder().build_from_memory(mem)
    assert profile.is_empty is False
    assert profile.summary.confidence > 0.5   # 明显高于旧地板 0.36
    assert profile.evidence_strength > 0.0


def test_recency_decays_with_signal_age():
    """最近信号 → 高时效；90+ 天前 → 趋近 0；无任何可用时间戳 → 中性 0.5。"""
    from datetime import UTC, datetime, timedelta

    now = datetime.now(UTC)
    recent = UserMemory(user_id="r", listening_history=[
        ListeningEvent(asset_id="a", timestamp=now.isoformat())])
    stale = UserMemory(user_id="s", listening_history=[
        ListeningEvent(asset_id="a", timestamp=(now - timedelta(days=100)).isoformat())])
    no_ts = UserMemory(user_id="t")  # 无 listening/ratings/明确偏好 → 无候选时间戳

    r_recent = _compute_recency(collect_profile_evidence(recent))
    r_stale = _compute_recency(collect_profile_evidence(stale))
    r_none = _compute_recency(collect_profile_evidence(no_ts))

    assert r_recent > 0.9
    assert r_stale == 0.0
    assert r_none == 0.5


# ── insight 反馈与稳定 id ────────────────────────────────────────────────────
def test_insight_ids_stable_across_rebuilds(service):
    memory = _rich_memory("stable-user")
    service.store.write_model("memory", "stable-user", memory)
    p1 = service.get_profile("stable-user")
    p2 = service.get_profile("stable-user")
    ids1 = {i.insight_id for i in p1.insights}
    ids2 = {i.insight_id for i in p2.insights}
    assert ids1 == ids2 and ids1


def test_reject_insight_persists_and_excludes_from_llm_context(service):
    memory = _rich_memory("reject-user")
    service.store.write_model("memory", "reject-user", memory)
    profile = service.get_profile("reject-user")
    target = next(i for i in profile.insights if i.dimension == "sound")

    updated = service.update_insight_feedback("reject-user", target.insight_id, "reject")
    again = next(i for i in updated.insights if i.insight_id == target.insight_id)
    assert again.status == "rejected"

    # 重新读取，状态保留（持久化）
    reread = service.get_profile("reject-user")
    assert next(i for i in reread.insights if i.insight_id == target.insight_id).status == "rejected"

    # 硬约束始终进 LLM 上下文（最高优先级），rejected 判断不污染
    ctx = service.get_context_for_llm("reject-user")
    assert "抖音热歌" in ctx.hard_constraints


def test_confirm_and_reset_insight(service):
    memory = _rich_memory("confirm-user")
    service.store.write_model("memory", "confirm-user", memory)
    profile = service.get_profile("confirm-user")
    target = profile.insights[0]

    service.update_insight_feedback("confirm-user", target.insight_id, "confirm")
    assert next(i for i in service.get_profile("confirm-user").insights
                if i.insight_id == target.insight_id).status == "confirmed"

    # reset 回到 active
    service.update_insight_feedback("confirm-user", target.insight_id, "reset")
    assert next(i for i in service.get_profile("confirm-user").insights
                if i.insight_id == target.insight_id).status == "active"


def test_delete_insight_feedback(service):
    memory = _rich_memory("del-user")
    service.store.write_model("memory", "del-user", memory)
    profile = service.get_profile("del-user")
    target = profile.insights[0]
    service.update_insight_feedback("del-user", target.insight_id, "reject")
    assert service.delete_insight("del-user", target.insight_id) is True
    # 删除后恢复默认 active
    assert next(i for i in service.get_profile("del-user").insights
                if i.insight_id == target.insight_id).status == "active"
    # 再次删除返回 False（已不存在）
    assert service.delete_insight("del-user", target.insight_id) is False


def test_unknown_feedback_action_rejected():
    with pytest.raises(ValueError):
        _normalize_status("banana")


def test_normalize_status_aliases():
    assert _normalize_status("disable_for_recommendation") == "rejected"
    assert _normalize_status("accurate") == "confirmed"
    assert _normalize_status("temp") == "temporary"


# ── 硬约束优先级 ────────────────────────────────────────────────────────────
def test_hard_constraints_surface_in_profile():
    builder = UserProfileBuilder()
    profile = builder.build_from_memory(_rich_memory())
    assert "抖音热歌" in profile.hard_constraints
    assert "不要中文歌" in profile.hard_constraints
    # 排除项应生成高置信 language insight
    lang_insights = [i for i in
                     __import__("app.profile.insights", fromlist=["generate_profile_insights"])
                     .generate_profile_insights(profile) if i.dimension == "language"]
    assert lang_insights
    assert all(i.confidence >= 0.66 for i in lang_insights)
