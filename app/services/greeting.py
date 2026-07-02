"""GreetingService —— 对话入口的个性化问候语生成。

从 `AudioVisualAgent` 抽离：基于画像/偏好/时间/目标/收听历史/曲库规模拼一段
「我先看了一眼你的音乐状态」式开场白。纯文本生成，依赖 memory + list_assets 回调；
agent 保留同名薄委托。
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

from app.memory import MemoryManager
from app.models import Asset


class GreetingService:
    def __init__(
        self,
        memory: MemoryManager,
        *,
        list_assets: Callable[[], list[Asset]],
    ) -> None:
        self.memory = memory
        self._list_assets = list_assets

    def generate_greeting(self, user_id: str) -> str:
        memory = self.memory.get_memory(user_id)
        assets = self._list_assets()
        goal = self.memory.get_active_goal(user_id)
        parts = ["嘿，我先看了一眼你的音乐状态。"]

        if memory.taste_profile and memory.taste_profile.top_genres:
            top_genre = memory.taste_profile.top_genres[0][0]
            parts.append(f"你最近的品味更偏 {top_genre}。")
        elif memory.preferences:
            parts.append(f"我记得你提过：{memory.preferences[-1]}。")

        hour = datetime.now().hour
        if 6 <= hour < 11:
            parts.append("现在适合先找一些轻快但不吵的真实曲目。")
        elif 22 <= hour or hour < 2:
            parts.append("夜深了，我会优先找更松弛、耐听的版本。")

        if goal:
            parts.append(f"上次的目标还在：{goal.goal}")

        if memory.listening_history:
            recent = memory.listening_history[-3:]
            completed = sum(1 for item in recent if item.completed)
            if completed >= 2:
                parts.append("最近你完整听完的歌比较多，我会延续这个方向。")
            elif len(recent) >= 2 and completed == 0:
                parts.append("最近跳过比较多，我会少依赖本地库，多去线上找新候选。")

        if len(assets) < 3:
            parts.append("曲库还不多，我可以先联网找真实候选，或者导入网易云歌单再推荐。")
        else:
            parts.append("我会把真实线上候选放前面，本地库只当作你的口味参考。")

        return " ".join(parts)
