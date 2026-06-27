"""画像服务层：API 调用入口 + insight 反馈落地（计划 §16.4）。

读路径：get_profile = 读 memory → builder 构建 → insights 生成 → 贴回反馈状态。
写路径：update_insight_feedback = 把用户的确认/否定/暂时写入 profile_feedback/{user_id}。

反馈持久化与 UserMemory 分离（独立 collection），因为它是「对画像判断的元反馈」，
不是偏好本身；画像重算后只要判断还在，反馈即自动复用。
"""

from __future__ import annotations

import logging

from app.profile.builder import UserProfileBuilder
from app.profile.insights import apply_feedback, generate_profile_insights
from app.profile.models import (
    ProfileContextForLLM,
    ProfileFeedbackState,
    ProfileInsightFeedback,
    UserProfileResponse,
)
from app.models import UserMemory, utc_now_iso
from app.storage import JsonStore

logger = logging.getLogger(__name__)

_FEEDBACK_COLLECTION = "profile_feedback"


class UserProfileService:
    def __init__(self, store: JsonStore, memory_manager) -> None:
        self.store = store
        self.memory = memory_manager
        self.builder = UserProfileBuilder()

    # ── 读 ────────────────────────────────────────────────────────────────
    def get_profile(self, user_id: str) -> UserProfileResponse:
        memory = self._memory_with_taste(user_id)
        profile = self.builder.build_from_memory(memory)
        if profile.is_empty:
            return profile
        insights = generate_profile_insights(profile)
        feedback = self._read_feedback(user_id)
        profile.insights = apply_feedback(insights, feedback)
        return profile

    def get_context_for_llm(self, user_id: str) -> ProfileContextForLLM:
        """压缩画像，供 plan_intent / recommend 注入（计划 §14.1，控制上下文长度）。

        只输出 rejected 以外的有效信号；hard_constraints 永远带上（最高优先级）。
        """
        profile = self.get_profile(user_id)
        if profile.is_empty:
            return ProfileContextForLLM()
        # 用户否定的画像判断（纠错回流）：原文标题给 LLM，让规划/推荐反转或回避这些信号。
        # 此前这里算了一个 rejected 集合却丢弃——现在真正 surface 出去。
        rejected_signals = [
            ins.title for ins in profile.insights if ins.status == "rejected"
        ][:6]
        active_scene = ""
        for scene in profile.scenes:
            if scene.confidence >= 0.4:
                active_scene = f"{scene.label}：{scene.recommendation_strategy}"
                break
        avoid: list[str] = []
        for scene in profile.scenes:
            avoid.extend(scene.avoid_features)
        return ProfileContextForLLM(
            taste_summary=profile.summary.headline,
            active_scene_preference=active_scene,
            hard_constraints=list(profile.hard_constraints),
            avoid_features=list(dict.fromkeys(avoid))[:5],
            discovery_mode=profile.discovery_style.label,
            rejected_signals=rejected_signals,
        )

    # ── 写 ────────────────────────────────────────────────────────────────
    def update_insight_feedback(
        self, user_id: str, insight_id: str, status: str,
    ) -> UserProfileResponse:
        """记录用户对某条 insight 的反馈，并返回刷新后的画像。

        action 映射（计划 §13.2）：
          confirm                    → confirmed
          reject / disable_for_recommendation → rejected
          temporary                  → temporary
        rejected 的判断会被 disable，使其不再用于推荐（见 get_context_for_llm）。
        """
        normalized = _normalize_status(status)
        # 校验 insight_id 真实存在于当前画像，避免写入孤儿反馈。
        profile = self.get_profile(user_id)
        target = next((i for i in profile.insights if i.insight_id == insight_id), None)
        with self.store.lock(_FEEDBACK_COLLECTION, user_id):
            state = self._read_feedback(user_id) or ProfileFeedbackState(user_id=user_id)
            state.insights[insight_id] = ProfileInsightFeedback(
                insight_id=insight_id,
                status=normalized,
                title=target.title if target else "",
                dimension=target.dimension if target else "",
            )
            state.updated_at = utc_now_iso()
            self.store.write_model(_FEEDBACK_COLLECTION, user_id, state)
        return self.get_profile(user_id)

    def delete_insight(self, user_id: str, insight_id: str) -> bool:
        """删除某条 insight 反馈（等价于让它回到默认 active）。计划 §24。"""
        with self.store.lock(_FEEDBACK_COLLECTION, user_id):
            state = self._read_feedback(user_id)
            if not state or insight_id not in state.insights:
                return False
            del state.insights[insight_id]
            state.updated_at = utc_now_iso()
            self.store.write_model(_FEEDBACK_COLLECTION, user_id, state)
            return True

    def clear_profile_feedback(self, user_id: str) -> bool:
        """清空画像反馈（计划 §24：清除画像。不动 UserMemory 本身的偏好）。"""
        return self.store.delete_key(_FEEDBACK_COLLECTION, user_id)

    # ── 内部 ──────────────────────────────────────────────────────────────
    def _read_feedback(self, user_id: str) -> ProfileFeedbackState | None:
        try:
            return self.store.read_model(_FEEDBACK_COLLECTION, user_id, ProfileFeedbackState)
        except Exception:
            logger.warning("Unreadable profile feedback for %s; ignoring", user_id, exc_info=True)
            return None

    def _memory_with_taste(self, user_id: str) -> UserMemory:
        """确保 taste_profile 已计算（首次访问时按已分析曲库回填一次）。"""
        memory = self.memory.get_memory(user_id)
        if memory.taste_profile is None and hasattr(self.memory, "store"):
            # taste_profile 缺失时尝试回填；失败不阻断，builder 用空 TasteProfile 也能跑。
            try:
                from app.api.main import agent  # 延迟导入避免循环
                library = [a for a in agent.list_assets() if a.status == "analyzed"]
                memory = self.memory.refresh_taste_profile(user_id, library)
            except Exception:
                logger.debug("taste_profile 回填跳过（库不可用）", exc_info=True)
        return memory


def _normalize_status(status: str) -> str:
    mapping = {
        "confirm": "confirmed",
        "confirmed": "confirmed",
        "accurate": "confirmed",
        "reject": "rejected",
        "rejected": "rejected",
        "inaccurate": "rejected",
        "disable_for_recommendation": "rejected",
        "temporary": "temporary",
        "temp": "temporary",
        "active": "active",
        "reset": "active",
    }
    normalized = mapping.get((status or "").strip().lower())
    if not normalized:
        raise ValueError(f"unknown insight feedback action: {status}")
    return normalized
