"""profile_context_text：画像仪表盘 → query_plan 品味上下文的格式化与健壮性。

get_context_for_llm 此前是「已留未接线」的死接口；现在 agent.profile_context_text 把它
格式化进 query_plan。这里锁定格式 + 空画像/异常安全降级（返空串，调用方跳过注入）。
"""
from __future__ import annotations

import tempfile

import app.profile.service as profile_service
from app.agent import AudioVisualAgent
from app.profile.models import (
    ArtistRelation,
    ProfileContextForLLM,
    ProfileInsight,
    UserProfileResponse,
)
from app.storage import JsonStore


def _agent() -> AudioVisualAgent:
    return AudioVisualAgent(JsonStore(tempfile.mkdtemp()))


def _profile(artists: list[tuple[str, str]], insights: list[ProfileInsight]) -> UserProfileResponse:
    """构造一个非空画像：artists=[(name, relation_type)]，insights 自带 status/dimension。"""
    return UserProfileResponse(
        user_id="u",
        is_empty=False,
        artists=[ArtistRelation(artist=name, relation_type=rel, confidence=0.8) for name, rel in artists],
        insights=insights,
    )


def test_empty_profile_returns_empty(monkeypatch):
    monkeypatch.setattr(
        profile_service.UserProfileService, "get_context_for_llm",
        lambda self, uid: ProfileContextForLLM(),
    )
    assert _agent().profile_context_text("u") == ""


def test_formats_all_fields(monkeypatch):
    ctx = ProfileContextForLLM(
        taste_summary="偏爱 R&B 与说唱",
        active_scene_preference="深夜：偏慵懒氛围",
        discovery_mode="平衡探索型",
        hard_constraints=["不要电音", "不要抖音神曲"],
        avoid_features=["激烈鼓点"],
    )
    monkeypatch.setattr(
        profile_service.UserProfileService, "get_context_for_llm",
        lambda self, uid: ctx,
    )
    text = _agent().profile_context_text("u")
    assert "偏爱 R&B 与说唱" in text
    assert "深夜" in text
    assert "平衡探索型" in text
    assert "不要电音" in text
    assert "激烈鼓点" in text


def test_failure_returns_empty(monkeypatch):
    """画像服务抛异常时不应阻断规划——返回空串，调用方跳过注入。"""
    def boom(self, uid):
        raise RuntimeError("profile unavailable")
    monkeypatch.setattr(profile_service.UserProfileService, "get_context_for_llm", boom)
    assert _agent().profile_context_text("u") == ""


# ---- 纠错回流：rejected insight 真正影响推荐 ----

def test_profile_context_text_includes_rejected(monkeypatch):
    """用户否定的画像判断进 query_plan 提示，让 LLM 不再据此推荐。"""
    ctx = ProfileContextForLLM(
        taste_summary="偏爱 R&B",
        rejected_signals=["Drake 是你的核心艺人", "你的情绪偏好集中在「激烈」一带"],
    )
    monkeypatch.setattr(
        profile_service.UserProfileService, "get_context_for_llm",
        lambda self, uid: ctx,
    )
    text = _agent().profile_context_text("u")
    assert "已否定" in text
    assert "Drake" in text


def test_get_context_for_llm_surfaces_rejected_insights(monkeypatch):
    """get_context_for_llm 此前算了 rejected 集合却丢弃；现在须真正 surface 出去。"""
    profile = _profile(
        artists=[("Drake", "core")],
        insights=[
            ProfileInsight(insight_id="x", title="Drake 是你的核心艺人", dimension="artist", status="rejected"),
            ProfileInsight(insight_id="y", title="你的探索风格：平衡探索型", dimension="discovery", status="active"),
        ],
    )
    monkeypatch.setattr(profile_service.UserProfileService, "get_profile", lambda self, uid: profile)
    svc = profile_service.UserProfileService(None, None)
    ctx = svc.get_context_for_llm("u")
    assert ctx.rejected_signals == ["Drake 是你的核心艺人"]  # 只收 rejected，active 不收


def test_rejected_core_artist_reversed_to_penalty(monkeypatch):
    """否定「X 是核心艺人」→ X 从加分移到减分（别再按核心推）。"""
    profile = _profile(
        artists=[("Drake", "core"), ("The Weeknd", "core")],
        insights=[ProfileInsight(
            insight_id="x", title="Drake 是你的核心艺人", dimension="artist", status="rejected",
        )],
    )
    monkeypatch.setattr(profile_service.UserProfileService, "get_profile", lambda self, uid: profile)
    boost, penalty = _agent()._profile_rerank_signals("u")
    assert "Drake" in penalty and "Drake" not in boost
    assert "The Weeknd" in boost  # 未被否定的核心艺人照常加分


def test_rejected_avoid_artist_un_penalized(monkeypatch):
    """否定「不要推 X」→ X 从减分移除（用户其实不排斥 X）。"""
    profile = _profile(
        artists=[("Justin Bieber", "avoid")],
        insights=[ProfileInsight(
            insight_id="y", title="你不希望被推荐 Justin Bieber 这类", dimension="artist", status="rejected",
        )],
    )
    monkeypatch.setattr(profile_service.UserProfileService, "get_profile", lambda self, uid: profile)
    _boost, penalty = _agent()._profile_rerank_signals("u")
    assert "Justin Bieber" not in penalty


def test_non_rejected_insights_do_not_reverse(monkeypatch):
    """active/confirmed 的洞察不触发反转——只有 rejected 才回流。"""
    profile = _profile(
        artists=[("Drake", "core")],
        insights=[ProfileInsight(
            insight_id="z", title="Drake 是你的核心艺人", dimension="artist", status="confirmed",
        )],
    )
    monkeypatch.setattr(profile_service.UserProfileService, "get_profile", lambda self, uid: profile)
    boost, penalty = _agent()._profile_rerank_signals("u")
    assert "Drake" in boost and "Drake" not in penalty
