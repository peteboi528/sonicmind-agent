"""微信公众号适配器。

核心流程：
  1. GET 验签 → SHA1(sort(token, timestamp, nonce)) == signature → 返回 echostr
  2. POST 回调 → 解析 XML → 异步回复（客服消息 API）

关键约束：微信要求 5 秒内回复，否则断开。
策略：立即返回 "success"，后台 asyncio.create_task 处理后通过客服 API 回复。

依赖：stdlib（xml.etree, hashlib）+ httpx（已有）
不依赖微信 SDK。
"""

from __future__ import annotations

import hashlib
import logging
import time
import xml.etree.ElementTree as ET
from typing import Any

import httpx

from app.adapters.protocol import BotResponse, IncomingMessage

logger = logging.getLogger(__name__)


class WeChatAdapter:
    """微信公众号适配器（实现 BotAdapter 协议）。"""

    def __init__(self, token: str, app_id: str, app_secret: str):
        self.token = token
        self.app_id = app_id
        self.app_secret = app_secret
        self._access_token: str = ""
        self._token_expires: float = 0

    # ---- 签名验证 ----

    def verify_signature(self, timestamp: str, nonce: str, signature: str) -> bool:
        """微信签名校验：SHA1(sort(token, timestamp, nonce))。"""
        parts = sorted([self.token, timestamp, nonce])
        raw = "".join(parts)
        expected = hashlib.sha1(raw.encode("utf-8")).hexdigest()
        return expected == signature

    def verify_request(self, body: bytes, headers: dict[str, str]) -> bool:
        """签名验证（由路由层传参调用 verify_signature）。"""
        return True  # 实际验证在路由层完成

    # ---- BotAdapter 协议方法 ----

    def parse_request(self, body: bytes, headers: dict[str, str]) -> IncomingMessage | None:
        """解析微信 XML 消息。"""
        try:
            xml_str = body.decode("utf-8")
        except UnicodeDecodeError:
            logger.warning("WeChat: invalid body encoding")
            return None

        msg = self._parse_xml(xml_str)
        if not msg:
            return None

        msg_type = msg.get("MsgType", "")
        if msg_type != "text":
            # 图片/语音/事件等暂时回复默认文本
            return None

        content = msg.get("Content", "").strip()
        if not content:
            return None

        openid = msg.get("FromUserName", "")
        msg_id = msg.get("MsgId", "")

        return IncomingMessage(
            user_id=f"wechat_{openid}",
            content=content,
            raw=msg,
            message_id=msg_id,
        )

    def format_response(self, response: BotResponse) -> Any:
        """将 BotResponse 转为微信 XML 文本回复。"""
        # 微信通过客服 API 发送时用 JSON 格式
        return self._build_text_xml("from", "to", response.text)

    # ---- XML 处理 ----

    def _parse_xml(self, xml_str: str) -> dict[str, str]:
        """解析微信 XML 消息为字典。"""
        try:
            root = ET.fromstring(xml_str)
            return {child.tag: child.text or "" for child in root}
        except ET.ParseError:
            logger.warning("WeChat: XML parse error")
            return {}

    def _build_text_xml(self, from_user: str, to_user: str, text: str) -> str:
        """构建微信文本回复 XML。"""
        ts = str(int(time.time()))
        return (
            f"<xml>"
            f"<ToUserName><![CDATA[{to_user}]]></ToUserName>"
            f"<FromUserName><![CDATA[{from_user}]]></FromUserName>"
            f"<CreateTime>{ts}</CreateTime>"
            f"<MsgType><![CDATA[text]]></MsgType>"
            f"<Content><![CDATA[{text[:2048]}]]></Content>"
            f"</xml>"
        )

    def _build_news_json(self, openid: str, response: BotResponse) -> dict:
        """构建微信图文消息（客服消息 API 格式）。"""
        articles = []
        for card in response.cards[:8]:  # 微信最多 8 条
            articles.append(
                {
                    "title": card.title,
                    "description": card.reason or card.artist,
                    "url": card.playback_url or "",
                    "picurl": card.cover_url or "",
                }
            )

        return {
            "touser": openid,
            "msgtype": "news",
            "news": {"articles": articles},
        }

    def _build_text_json(self, openid: str, text: str) -> dict:
        """构建微信文本消息（客服消息 API 格式）。"""
        return {
            "touser": openid,
            "msgtype": "text",
            "text": {"content": text[:2048]},
        }

    # ---- 客服消息发送 ----

    async def send_customer_service(self, openid: str, response: BotResponse) -> bool:
        """通过微信客服消息 API 发送回复。"""
        token = self._get_access_token()
        if not token:
            logger.error("WeChat: no access token")
            return False

        # 如果有歌曲卡片，发图文消息
        if response.cards:
            body = self._build_news_json(openid, response)
        else:
            body = self._build_text_json(openid, response.text)

        url = f"https://api.weixin.qq.com/cgi-bin/message/custom/send?access_token={token}"
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(url, json=body)
                data = resp.json()
                if data.get("errcode", 0) == 0:
                    return True
                logger.error("WeChat send failed: %s", data)
                return False
        except Exception:
            logger.exception("WeChat send error")
            return False

    # ---- Token Management ----

    def _get_access_token(self) -> str:
        """获取微信 access_token（缓存至过期）。"""
        if self._access_token and time.time() < self._token_expires:
            return self._access_token

        url = (
            f"https://api.weixin.qq.com/cgi-bin/token"
            f"?grant_type=client_credential"
            f"&appid={self.app_id}"
            f"&secret={self.app_secret}"
        )
        try:
            resp = httpx.get(url, timeout=10)
            data = resp.json()
            self._access_token = data.get("access_token", "")
            expire = data.get("expires_in", 7200)
            self._token_expires = time.time() + expire - 120  # 提前 2 分钟刷新
            return self._access_token
        except Exception:
            logger.exception("WeChat token fetch failed")
            return ""
