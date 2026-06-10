"""Bot 适配器包：一套 Agent 代码，多平台复用。

支持的接入方式：
  - Web 前端（index.html）
  - 飞书 Bot（Interactive Card）
  - 微信公众号（图文消息 + 客服 API）

所有适配器实现 BotAdapter 协议（结构化子类型，与 app/sources/protocol.py 一致）。
"""
from __future__ import annotations

from app.adapters.protocol import BotAdapter, BotResponse, IncomingMessage, SongCard

__all__ = ["BotAdapter", "BotResponse", "IncomingMessage", "SongCard"]
