"""ProfileSignals —— 画像消费层：把用户画像转成推荐/对话可用的信号与文本。

从 `AudioVisualAgent` 抽离：summarize_taste（听歌频次画像 → 摘要文本，供对话/playlist）、
profile_context_text（画像仪表盘 → query_plan 品位上下文）、profile_rerank_signals
（画像仪表盘 → rerank 艺人 boost/penalty，含纠错回流）。依赖 store/memory + list_assets
回调；UserProfileService 在方法内局部构造（无状态）。agent 保留同名薄委托。
"""

from __future__ import annotations

import logging
from typing import Callable

from app.memory import MemoryManager
from app.models import Asset, TasteProfile, UserMemory
from app.storage import JsonStore

logger = logging.getLogger(__name__)


class ProfileSignals:
    def __init__(
        self,
        store: JsonStore,
        memory: MemoryManager,
        *,
        list_assets: Callable[[], list[Asset]],
    ) -> None:
        self.store = store
        self.memory = memory
        self._list_assets = list_assets

    def summarize_taste(self, user_id: str, *, include_artists: bool = True, memory: UserMemory | None = None) -> str:
        # memory 可由调用方传入（recommend_for_query 已 get_memory 过），省掉同请求内
        # 对同一用户的重复读盘+校验。不传则照旧自行读取，行为不变。
        if memory is None:
            memory = self.memory.get_memory(user_id)
        if not memory.taste_profile:
            library = [asset for asset in self._list_assets() if asset.status == "analyzed"]
            memory = self.memory.refresh_taste_profile(user_id, library)
        taste = memory.taste_profile or TasteProfile()
        genres = [genre for genre, _ in taste.top_genres[:4]]
        moods = [mood for mood, _ in taste.top_moods[:4]]
        artists = [artist for artist, _ in taste.top_artists[:5]]
        prefs = memory.preferences[-3:]
        genre_text = "、".join(genres) if genres else "未形成稳定风格"
        mood_text = "、".join(moods) if moods else "暂无明显偏好"
        pref_text = "；".join(prefs) if prefs else "暂无"
        artist_text = "、".join(artists) if artists else ""
        parts = [
            f"你的品味目前更偏向 {genre_text}，"
            f"情绪上常出现 {mood_text}，"
        ]
        if include_artists and artist_text:
            parts.append(f"偏好的艺人有 {artist_text}，")
        parts.append(f"显式表达过的偏好包括 {pref_text}。")
        return "".join(parts)

    def profile_context_text(self, user_id: str) -> str:
        """压缩画像仪表盘（app/profile/）为 query_plan 用的品位上下文（软参考）。

        与 summarize_taste（听歌历史频次）互补：这里带场景偏好、探索风格、画像级排除/回避，
        以及被用户「纠错」后落地的信号。空画像/异常返 ""——调用方据此跳过注入，行为不变。
        """
        try:
            from app.services.profile import UserProfileService
            ctx = UserProfileService(self.store, self.memory).get_context_for_llm(user_id)
        except Exception:
            logger.debug("profile_context_text 失败，跳过画像注入", exc_info=True)
            return ""
        parts: list[str] = []
        if ctx.taste_summary:
            parts.append(f"当前品味：{ctx.taste_summary}")
        if ctx.active_scene_preference:
            parts.append(f"常听场景：{ctx.active_scene_preference}")
        if ctx.discovery_mode:
            parts.append(f"探索风格：{ctx.discovery_mode}")
        if ctx.hard_constraints:
            parts.append(f"明确排除：{'、'.join(ctx.hard_constraints[:5])}")
        if ctx.avoid_features:
            parts.append(f"场景应回避：{'、'.join(ctx.avoid_features[:5])}")
        if ctx.rejected_signals:
            # 用户在画像页「纠错」否定过的判断——推荐应反转/回避，别再按这些推。
            parts.append(f"用户已否定（勿据此推荐）：{'；'.join(ctx.rejected_signals[:4])}")
        return "；".join(parts)

    def profile_rerank_signals(self, user_id: str) -> tuple[set[str], set[str]]:
        """从画像仪表盘取 rerank 艺人信号：core/rising→加分，avoid→减分。

        与 memory.taste_profile（听歌频次）互补——画像是可解释的艺人关系判断。
        **纠错回流**：用户在画像页否定的艺人洞察会反转关系——
          否定「X 是核心艺人」→ X 从加分移到减分（别再按核心推）；
          否定「不要推 X」→ X 从减分移除（用户其实不排斥 X）。
        空画像/异常返回空集，rerank 行为不变。
        """
        try:
            from app.services.profile import UserProfileService
            profile = UserProfileService(self.store, self.memory).get_profile(user_id)
        except Exception:
            logger.debug("profile_rerank_signals 失败，降级无画像信号", exc_info=True)
            return set(), set()
        if getattr(profile, "is_empty", True):
            return set(), set()
        # 纠错回流：被否定的艺人洞察（title 含艺人名）决定关系反转方向——
        #   core/rising 被否定 → 从加分改减分；avoid 被否定 → 不再减分。
        # 直接按关系+是否被否定一次建表，避免先加进 penalty 又被第二轮扫描误删。
        rejected_titles = [
            i.title for i in profile.insights
            if i.status == "rejected" and i.dimension == "artist"
        ]

        def _rejected(name: str) -> bool:
            return any(name in t for t in rejected_titles)

        boost: set[str] = set()
        penalty: set[str] = set()
        for a in profile.artists:
            if not a.artist:
                continue
            if a.relation_type in {"core", "rising"}:
                (penalty if _rejected(a.artist) else boost).add(a.artist)
            elif a.relation_type == "avoid" and not _rejected(a.artist):
                penalty.add(a.artist)
        return boost, penalty
