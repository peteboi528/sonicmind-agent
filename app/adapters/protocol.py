"""Bot 适配器统一协议。

每个平台（飞书 / 微信 / Web）实现 BotAdapter 的三个方法：
  parse_request  — 平台原始请求 → IncomingMessage
  format_response — BotResponse → 平台特定格式
  verify_request — 验签 / 鉴权

适配器是无状态的：对话历史由 Agent Memory 管理。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class IncomingMessage:
    """平台无关的统一输入消息。"""

    user_id: str  # 带平台前缀：feishu_{open_id} / wechat_{openid} / 直传
    content: str  # 用户文本
    raw: dict[str, Any] = field(default_factory=dict)  # 原始 payload（调试用）
    message_id: str = ""  # 去重


@dataclass
class SongCard:
    """歌曲卡片：跨平台展示用。"""

    title: str
    artist: str
    cover_url: str = ""
    playback_url: str = ""
    reason: str = ""
    source: str = ""
    score: float | None = None


@dataclass
class BotResponse:
    """适配器统一输出。"""

    text: str = ""
    cards: list[SongCard] = field(default_factory=list)


class BotAdapter(Protocol):
    """Bot 适配器协议（结构化子类型，无需继承）。"""

    def parse_request(self, body: bytes, headers: dict[str, str]) -> IncomingMessage | None:
        """解析平台原始请求。验证类请求（如微信 echo）返回 None（已内部处理）。"""
        ...

    def format_response(self, response: BotResponse) -> Any:
        """将 BotResponse 转为平台特定格式。"""
        ...

    def verify_request(self, body: bytes, headers: dict[str, str]) -> bool:
        """验证请求签名 / 合法性。"""
        ...
