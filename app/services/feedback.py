"""FeedbackService —— 用户行为/反馈写入与品味档案读取。

从 `AudioVisualAgent` 抽离的用户交互态层：收听记录（record_listen）、评分
（rate_asset）、品味档案读取（get_taste_profile）、记忆更新（update_memory）、
片段反馈（record_feedback）、不喜欢（record_dislike）。

核心编排是「用户行为 → memory 落盘 + library 的 Thompson 在线学习反馈」：
听完→正反馈、秒跳→负反馈、高分→正、低分/明确不喜欢→负。依赖通过构造注入
（store/memory/library + list_assets 回调）；agent 侧保留同名薄委托。
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from types import SimpleNamespace

from app.library import ResourceLibrary
from app.memory import MemoryManager
from app.models import (
    Asset,
    DislikeRequest,
    FeedbackRequest,
    MemoryUpdateRequest,
    Segment,
    TasteProfile,
    UserMemory,
    utc_now_iso,
)
from app.storage import JsonStore

logger = logging.getLogger(__name__)


class FeedbackService:
    def __init__(
        self,
        store: JsonStore,
        memory: MemoryManager,
        library: ResourceLibrary,
        *,
        list_assets: Callable[[], list[Asset]],
    ) -> None:
        self.store = store
        self.memory = memory
        self.library = library
        self._list_assets = list_assets

    def record_listen(
        self, user_id: str, asset_id: str, duration: int, completed: bool, context: str | None = None,
        title: str = "", artist: str = "", cover_url: str = "", source: str = "", source_id: str = "",
    ) -> UserMemory:
        memory = self.memory.record_listen(
            user_id, asset_id, duration, completed, context,
            title=title, artist=artist, cover_url=cover_url, source=source, source_id=source_id,
        )
        # Thompson 在线学习反馈环：听完 → 正反馈(α+1)，秒跳 → 负反馈(β+0.5)。
        asset = self.store.read_model("assets", asset_id, Asset)
        if asset is not None:
            if completed:
                self.library.update_ts_feedback(asset, positive=True, weight=1.0)
            elif duration and asset.duration_seconds and duration < asset.duration_seconds * 0.3:
                self.library.update_ts_feedback(asset, positive=False, weight=0.5)
        # 收听后刷新品味档案——否则收听历史只累积、永不重塑派生品味（习惯记录不到）。
        # 关键：把用户真实听过的曲目纳入计算，即使它只是 INGESTED 的在线曲（无分析标签
        # 也有艺人信号）。只用 analyzed 会把"实际在听的歌"全滤掉，习惯无从积累。
        memory = self.memory.refresh_taste_profile(user_id, self._taste_library(user_id))
        return memory

    def _taste_library(self, user_id: str) -> list[Asset]:
        """品味计算用的曲库：已分析曲 + 用户真实听过的曲（含 INGESTED 在线曲）。

        refresh_taste_profile 内部会再按 listening_history 过滤，这里只负责"别在分析状态
        门槛上把听过的在线曲提前筛掉"——它们带艺人/曲风信号，是习惯的主要载体。
        """
        listened_ids = {ev.asset_id for ev in self.memory.get_memory(user_id).listening_history if ev.asset_id}
        out: list[Asset] = []
        for asset in self._list_assets():
            if asset.status == "analyzed" or asset.asset_id in listened_ids:
                out.append(asset)
        return out

    def rate_asset(self, user_id: str, asset_id: str, score: float) -> UserMemory:
        asset = self.store.read_model("assets", asset_id, Asset)
        if asset is None:
            raise ValueError(f"Unknown asset_id: {asset_id}")
        memory = self.memory.record_rating(user_id, asset, score)
        # 高分 → Thompson 正反馈，低分 → 负反馈。
        if score >= 7.0:
            self.library.update_ts_feedback(asset, positive=True, weight=(score - 6.0) / 4.0)
        elif score <= 3.0:
            self.library.update_ts_feedback(asset, positive=False, weight=(4.0 - score) / 4.0)
        # 评分后立即刷新品味档案
        library = [a for a in self._list_assets() if a.status == "analyzed"]
        memory = self.memory.refresh_taste_profile(user_id, library)
        return memory

    def get_taste_profile(self, user_id: str) -> TasteProfile:
        memory = self.memory.get_memory(user_id)
        if not memory.taste_profile:
            library = [a for a in self._list_assets() if a.status == "analyzed"]
            memory = self.memory.refresh_taste_profile(user_id, library)
        return memory.taste_profile or TasteProfile()

    def update_memory(self, request: MemoryUpdateRequest) -> tuple[UserMemory, bool]:
        return self.memory.update_memory(request)

    def record_feedback(self, request: FeedbackRequest) -> UserMemory:
        segments: list[Segment] = []
        for key in self.store.list_keys("segments"):
            segments.extend(self.store.read_models("segments", key, Segment))
        target = next((s for s in segments if s.segment_id == request.segment_id), None)
        if target is None:
            raise ValueError(f"Unknown segment_id: {request.segment_id}")
        return self.memory.record_feedback(request.user_id, target, request.accepted)

    def record_dislike(self, request: DislikeRequest) -> UserMemory:
        self.library.add_dislike(request)
        # 负反馈也推给 Thompson：明确不喜欢 → ts_beta 大幅上调，后续探索几乎不再选中。
        self.library.update_ts_feedback(
            SimpleNamespace(
                title=request.title, artist=request.artist,
                source=request.source, external_id=request.source_id, asset_id=request.source_id,
            ),
            positive=False, weight=3.0,
        )
        memory = self.memory.get_memory(request.user_id)
        key = " - ".join(part for part in [request.title, request.artist] if part) or request.source_id or request.source
        if key and key not in memory.dislikes:
            memory.dislikes.append(key)
            memory.updated_at = utc_now_iso()
            self.store.write_model("memory", request.user_id, memory)
        return memory
