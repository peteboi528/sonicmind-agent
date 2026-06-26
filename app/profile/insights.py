"""把结构化画像转成用户可读、带证据、可纠错的洞察（计划 §9, §16.3）。

每条 insight 有稳定 insight_id（基于维度 + 关键词的确定性哈希），这样用户的
反馈（确认/否定/暂时/不用于推荐）能跨请求绑定到同一条判断——即使画像重算，
只要那条判断还在，反馈状态就保留。
"""

from __future__ import annotations

import hashlib

from app.profile.builder import confidence_band
from app.profile.models import (
    ProfileFeedbackState,
    ProfileInsight,
    UserProfileResponse,
)


def _insight_id(dimension: str, key: str) -> str:
    """维度 + 关键词 → 稳定短 id。画像重算后同一判断 id 不变，反馈状态可保留。"""
    digest = hashlib.sha256(f"{dimension}::{key}".encode()).hexdigest()
    return f"{dimension}-{digest[:10]}"


def generate_profile_insights(profile: UserProfileResponse) -> list[ProfileInsight]:
    """从画像各维度抽取可解释洞察。证据足→高置信，证据少→低置信并说明。"""
    insights: list[ProfileInsight] = []

    def add(dimension: str, key: str, title: str, explanation: str,
            confidence: float, evidence: list[str]) -> None:
        insights.append(ProfileInsight(
            insight_id=_insight_id(dimension, key),
            title=title,
            explanation=explanation,
            confidence=round(confidence, 2),
            confidence_band=confidence_band(confidence),
            evidence=[e for e in evidence if e],
            dimension=dimension,
        ))

    # 1) 声音指纹的高分维度 → 一条洞察。
    for dim in sorted(profile.sound_fingerprint.dimensions, key=lambda d: d.value, reverse=True)[:3]:
        if dim.value < 0.45:
            continue
        add(
            "sound", dim.key,
            f"你偏好{dim.label}强的声音",
            dim.explanation,
            confidence=min(1.0, 0.45 + dim.value * 0.4),
            evidence=[dim.explanation],
        )

    # 2) 情绪常驻区域。
    if profile.mood_landscape.global_points:
        top_mood = profile.mood_landscape.global_points[0]
        add(
            "mood", top_mood.mood,
            f"你的情绪偏好集中在「{top_mood.mood}」一带",
            profile.mood_landscape.summary,
            confidence=min(1.0, 0.4 + top_mood.weight * 0.4),
            evidence=[profile.mood_landscape.summary],
        )

    # 3) 场景偏好。
    for scene in profile.scenes[:3]:
        add(
            "scene", scene.scene,
            f"{scene.label}场景：你有明确的听歌习惯",
            scene.recommendation_strategy,
            confidence=scene.confidence,
            evidence=[f"偏好风格：{('、'.join(scene.preferred_genres) or '—')}",
                      scene.recommendation_strategy],
        )

    # 4) 核心 / 回避艺人。
    for artist in profile.artists:
        if artist.relation_type == "core":
            add(
                "artist", artist.artist,
                f"{artist.artist} 是你的核心艺人",
                "；".join(artist.reasons),
                confidence=artist.confidence,
                evidence=artist.reasons,
            )
        elif artist.relation_type == "avoid":
            add(
                "artist", artist.artist,
                f"你不希望被推荐 {artist.artist} 这类",
                "；".join(artist.reasons),
                confidence=max(artist.confidence, 0.6),
                evidence=artist.reasons,
            )

    # 5) 语言 / 硬约束（最高优先级，明确表达 → 高置信）。
    for rule in profile.hard_constraints[:4]:
        add(
            "language", rule,
            f"你明确排除：{rule}",
            "这是你主动表达的硬约束，推荐会严格遵守。",
            confidence=0.9,
            evidence=[f"用户明确排除「{rule}」"],
        )

    # 6) 探索风格。
    ds = profile.discovery_style
    add(
        "discovery", ds.label,
        f"你的探索风格：{ds.label}",
        ds.explanation,
        confidence=0.5 + abs(ds.novelty_tolerance - 0.5) * 0.6,
        evidence=[ds.explanation],
    )

    return insights


def apply_feedback(
    insights: list[ProfileInsight], feedback: ProfileFeedbackState | None,
) -> list[ProfileInsight]:
    """把用户存储的反馈状态贴回 insight 列表（reject/temporary/confirm）。"""
    if not feedback or not feedback.insights:
        return insights
    for insight in insights:
        fb = feedback.insights.get(insight.insight_id)
        if fb:
            insight.status = fb.status
    return insights
