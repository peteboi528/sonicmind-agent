"""用户画像数据契约（计划 §3-9, §13）。

刻意与 app/models.py 分开：那个文件已近千行，画像是独立子系统，模型应自包含。
所有 confidence ∈ [0,1]；所有文本字段为中文用户可读解释。空画像时各列表为空、
summary 给「还在认识你」的引导，绝不编造维度。
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


# ── 模块一：品味摘要 ────────────────────────────────────────────────────────
class TasteSummary(BaseModel):
    """画像首页摘要：用自然语言总结当前品味，而非罗列标签。"""

    headline: str = ""                                  # 当前品味一句话
    core_preferences: list[str] = Field(default_factory=list)  # 三个核心偏好
    recommendation_hint: str = ""                       # 一个系统判断（推荐时应注意什么）
    chips: list[str] = Field(default_factory=list)      # 首页小组件用的极简标签（轻快律动·治愈流行…）
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    updated_at: str = Field(default_factory=_utc_now_iso)


# ── 模块二：声音指纹 ────────────────────────────────────────────────────────
class SoundDimension(BaseModel):
    """声音指纹的一个可解释维度（带分值与归因解释）。"""

    key: str
    label: str
    value: float = Field(default=0.0, ge=0.0, le=1.0)   # 0-1，前端按需 ×100 展示
    explanation: str = ""


class SoundFingerprint(BaseModel):
    """声音结构，而非风格标签。每维都能解释「为什么是这个分」。"""

    dimensions: list[SoundDimension] = Field(default_factory=list)
    explanation: str = ""


# ── 模块三：情绪地图 ────────────────────────────────────────────────────────
class MoodPoint(BaseModel):
    """情绪地图上的一个点：valence(明亮↔阴郁) × arousal(平静↔激昂)。"""

    mood: str
    valence: float = Field(default=0.0, ge=-1.0, le=1.0)
    arousal: float = Field(default=0.0, ge=-1.0, le=1.0)
    weight: float = Field(default=0.0, ge=0.0)
    evidence_count: int = 0


class MoodLandscape(BaseModel):
    global_points: list[MoodPoint] = Field(default_factory=list)
    scene_points: dict[str, list[MoodPoint]] = Field(default_factory=dict)
    summary: str = ""


# ── 模块四：场景偏好 ────────────────────────────────────────────────────────
class ScenePreference(BaseModel):
    """不同场景下的不同音乐策略（跑步 ≠ 学习 ≠ 睡前）。"""

    scene: str
    label: str = ""
    preferred_genres: list[str] = Field(default_factory=list)
    preferred_moods: list[str] = Field(default_factory=list)
    avoid_features: list[str] = Field(default_factory=list)
    recommendation_strategy: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    examples: list[str] = Field(default_factory=list)


# ── 模块五：艺术家星系 ──────────────────────────────────────────────────────
class ArtistRelation(BaseModel):
    """用户与艺人的关系，而非「喜欢 X、Y、Z」的罗列。"""

    artist: str
    relation_type: Literal["core", "rising", "explore", "occasional", "avoid"] = "occasional"
    reasons: list[str] = Field(default_factory=list)
    evidence_tracks: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


# ── 模块六：探索倾向 ────────────────────────────────────────────────────────
class DiscoveryStyle(BaseModel):
    """用户愿意探索多远：保守 / 平衡 / 探索型。"""

    label: str = ""  # 中文人格标签：保守型/平衡探索型/探索型
    novelty_tolerance: float = Field(default=0.0, ge=0.0, le=1.0)
    mainstream_preference: float = Field(default=0.0, ge=0.0, le=1.0)
    niche_openness: float = Field(default=0.0, ge=0.0, le=1.0)
    language_openness: float = Field(default=0.0, ge=0.0, le=1.0)
    explanation: str = ""


# ── 模块七：画像置信度与可编辑反馈 ──────────────────────────────────────────
class ProfileInsight(BaseModel):
    """一条可解释、带证据、可纠错的画像判断。

    status 由用户反馈驱动（计划 §9.4, §13.2）：
      active     默认（系统推断，未经用户确认）
      confirmed  用户确认「准确」→ 推荐可加权信任
      temporary  用户标记「只是最近喜欢」→ 当短期偏好，不进长期画像
      rejected   用户标记「不准确 / 不用于推荐」→ 推荐不再使用
    """

    insight_id: str
    title: str
    explanation: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence_band: Literal["high", "medium", "low"] = "low"
    evidence: list[str] = Field(default_factory=list)
    status: Literal["active", "confirmed", "temporary", "rejected"] = "active"
    dimension: str = ""  # 关联维度：sound/mood/scene/artist/discovery/language…


class ProfileInsightFeedback(BaseModel):
    """用户对某条 insight 的反馈写入存储（service 层落地）。"""

    insight_id: str
    status: Literal["active", "confirmed", "temporary", "rejected"]
    title: str = ""
    dimension: str = ""
    updated_at: str = Field(default_factory=_utc_now_iso)


class ProfileFeedbackState(BaseModel):
    """整个用户的 insight 反馈集合，按 insight_id 持久化在 profile_feedback/{user_id}。"""

    user_id: str
    insights: dict[str, ProfileInsightFeedback] = Field(default_factory=dict)
    updated_at: str = Field(default_factory=_utc_now_iso)


# ── 顶层响应（GET /profile）────────────────────────────────────────────────
class UserProfileResponse(BaseModel):
    user_id: str
    is_empty: bool = False  # 数据不足：前端显示空状态引导而非空标签
    empty_hint: str = ""
    summary: TasteSummary = Field(default_factory=TasteSummary)
    sound_fingerprint: SoundFingerprint = Field(default_factory=SoundFingerprint)
    mood_landscape: MoodLandscape = Field(default_factory=MoodLandscape)
    scenes: list[ScenePreference] = Field(default_factory=list)
    artists: list[ArtistRelation] = Field(default_factory=list)
    discovery_style: DiscoveryStyle = Field(default_factory=DiscoveryStyle)
    insights: list[ProfileInsight] = Field(default_factory=list)
    hard_constraints: list[str] = Field(default_factory=list)  # 用户明确排除项（最高优先级）
    evidence_strength: float = Field(default=0.0, ge=0.0, le=1.0)  # 整体证据充分度
    updated_at: str = Field(default_factory=_utc_now_iso)


# ── 注入 Agent/LLM 的压缩画像（计划 §14.1，控制上下文长度）──────────────────
class ProfileContextForLLM(BaseModel):
    taste_summary: str = ""
    active_scene_preference: str = ""
    hard_constraints: list[str] = Field(default_factory=list)
    avoid_features: list[str] = Field(default_factory=list)
    discovery_mode: str = ""
    # 用户在画像仪表盘「纠错」否定掉的判断（status=rejected）。推荐应反转/回避这些信号，
    # 否则纠错只是表面按钮、不影响结果。原文字标题直接给 LLM 读；rerank 另抽艺人名做硬反转。
    rejected_signals: list[str] = Field(default_factory=list)
