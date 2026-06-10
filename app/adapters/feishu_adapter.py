"""飞书 Bot 适配器。

核心流程：
  1. url_verification challenge → 解密后回复 challenge 字符串
  2. im.message.receive_v1 事件 → 验签 → 解析 → 调 agent → 发交互卡片

依赖：httpx（已有）+ cryptography（新增，AES-CBC 解密）
不依赖飞书 SDK，纯 HTTP + stdlib。
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import time
from typing import Any

import httpx

from app.adapters.protocol import BotAdapter, BotResponse, IncomingMessage, SongCard

logger = logging.getLogger(__name__)


class FeishuAdapter:
    """飞书 Bot 适配器（实现 BotAdapter 协议）。"""

    def __init__(
        self,
        app_id: str,
        app_secret: str,
        verification_token: str,
        encrypt_key: str = "",
    ):
        self.app_id = app_id
        self.app_secret = app_secret
        self.verification_token = verification_token
        self.encrypt_key = encrypt_key
        self._tenant_token: str = ""
        self._token_expires: float = 0

    # ---- BotAdapter 协议方法 ----

    def verify_request(self, body: bytes, headers: dict[str, str]) -> bool:
        """验证飞书请求来源是否可信。

        两种校验路径，按可用凭证自动选择：
        1. 配了 encrypt_key 且带 X-Lark-Signature 头 → 做 SHA256 签名校验
           （飞书算法：sha256(timestamp + nonce + encrypt_key + body)）。
        2. 否则回退到 verification_token 校验（明文事件的 header.token / 顶层 token）。

        两者都无法校验（既没配 encrypt_key 又没配 verification_token）时，
        记录告警并放行——仅适合本地开发，公网部署必须至少配一项。
        """
        # 统一大小写不敏感地取 header
        h = {k.lower(): v for k, v in headers.items()}
        signature = h.get("x-lark-signature", "")

        if self.encrypt_key and signature:
            timestamp = h.get("x-lark-request-timestamp", "")
            nonce = h.get("x-lark-request-nonce", "")
            return self.verify_signature(timestamp, nonce, body, signature)

        # 无签名头：用 verification_token 校验 body 内的 token
        if self.verification_token:
            return self._verify_token(body)

        logger.warning(
            "Feishu: 未配置 encrypt_key 或 verification_token，请求未经校验放行"
            "（仅限本地开发，公网部署不安全）"
        )
        return True

    def verify_signature(
        self, timestamp: str, nonce: str, body: bytes, signature: str
    ) -> bool:
        """飞书事件订阅签名校验：sha256(timestamp + nonce + encrypt_key + body)。"""
        if not (timestamp and nonce and signature):
            return False
        raw = timestamp.encode() + nonce.encode() + self.encrypt_key.encode() + body
        expected = hashlib.sha256(raw).hexdigest()
        return expected == signature

    def _verify_token(self, body: bytes) -> bool:
        """校验明文事件 body 里的 verification token（v2 在 header.token，v1 在顶层）。"""
        try:
            payload = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return False
        token = payload.get("header", {}).get("token") or payload.get("token", "")
        return token == self.verification_token

    def _decode_body(self, body: bytes) -> dict | None:
        """把原始 body 解析成事件 dict。

        配了 encrypt_key 时飞书会把整个包加密成 {"encrypt": "..."}，
        这里统一解密还原成明文 dict；未加密则直接 json 解析。
        """
        try:
            payload = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            logger.warning("Feishu: invalid JSON body")
            return None
        # 加密包：解密 encrypt 字段
        if isinstance(payload, dict) and "encrypt" in payload and self.encrypt_key:
            plaintext = self._decrypt(payload["encrypt"])
            if not plaintext:
                return None
            try:
                return json.loads(plaintext)
            except json.JSONDecodeError:
                logger.warning("Feishu: decrypted body is not valid JSON")
                return None
        return payload

    def parse_request(
        self, body: bytes, headers: dict[str, str]
    ) -> IncomingMessage | None:
        """解析飞书事件回调。

        返回 None 表示这是一个 verification challenge（已内部处理）。
        """
        payload = self._decode_body(body)
        if payload is None:
            return None

        # 1) URL verification challenge
        if payload.get("type") == "url_verification":
            # 已在路由层处理，这里返回 None
            return None

        # 2) 事件回调
        event = payload.get("event", {})
        if not event:
            logger.debug("Feishu: non-event payload: %s", payload.get("type"))
            return None

        # 提取消息内容
        msg = event.get("message", {})
        msg_type = msg.get("message_type", "")
        msg_id = msg.get("message_id", "")

        if msg_type != "text":
            # 目前只处理文本消息
            return None

        # 解析文本
        content_str = msg.get("content", "{}")
        try:
            content = json.loads(content_str) if isinstance(content_str, str) else content_str
        except json.JSONDecodeError:
            content = {}

        text = content.get("text", "").strip()
        if not text:
            return None

        # 提取用户 ID
        sender = event.get("sender", {})
        open_id = sender.get("sender_id", {}).get("open_id", "")
        user_id = msg.get("chat_id", "") or open_id

        return IncomingMessage(
            user_id=f"feishu_{open_id}" if open_id else f"feishu_{user_id}",
            content=text,
            raw=payload,
            message_id=msg_id,
        )

    def format_response(self, response: BotResponse) -> Any:
        """将 BotResponse 转为飞书 Interactive Card JSON。"""
        if not response.cards:
            # 纯文本消息
            return {"msg_type": "text", "content": json.dumps({"text": response.text}, ensure_ascii=False)}

        return self._build_song_card(response.text, response.cards)

    # ---- 飞书 API 调用 ----

    def send_reply(self, open_id: str, response: BotResponse) -> bool:
        """通过飞书消息 API 发送回复。"""
        card = self.format_response(response)
        token = self._get_tenant_token()
        if not token:
            logger.error("Feishu: no tenant token")
            return False

        url = "https://open.feishu.cn/open-apis/im/v1/messages"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        body = {
            "receive_id": open_id,
            "msg_type": card.get("msg_type", "interactive"),
            "content": card.get("content", "{}") if card.get("msg_type") == "text" else json.dumps(card, ensure_ascii=False),
        }

        try:
            resp = httpx.post(
                f"{url}?receive_id_type=open_id",
                headers=headers,
                json=body,
                timeout=10,
            )
            if resp.status_code == 200:
                return True
            logger.error("Feishu send failed: %s %s", resp.status_code, resp.text[:200])
            return False
        except Exception:
            logger.exception("Feishu send error")
            return False

    def handle_challenge(self, body: bytes) -> dict | None:
        """处理飞书 URL verification challenge。

        加密模式下整个 body 是 {"encrypt": "..."}，解密后才有 challenge；
        明文模式下 body 直接含 challenge。两种都通过 _decode_body 归一处理。
        """
        payload = self._decode_body(body)
        if payload is None:
            return None

        if payload.get("type") != "url_verification":
            return None

        if payload.get("token") and payload["token"] != self.verification_token:
            logger.warning("Feishu challenge token mismatch")

        return {"challenge": payload.get("challenge", "")}

    # ---- Card Builder ----

    def _build_song_card(self, intro_text: str, cards: list[SongCard]) -> dict:
        """构建飞书 Interactive Card（歌曲列表）。"""
        elements = []

        # 引言文字
        if intro_text:
            elements.append({
                "tag": "markdown",
                "content": intro_text[:600],
            })

        # 歌曲列表（最多 5 首）
        for i, card in enumerate(cards[:5], 1):
            elements.append({
                "tag": "column_set",
                "flex_mode": "bisect",
                "background_style": "default",
                "columns": [
                    {
                        "tag": "column",
                        "width": "weighted",
                        "weight": 1,
                        "elements": [
                            {
                                "tag": "markdown",
                                "content": f"**{i}. {card.title}**\n"
                                          f"{card.artist}"
                                          + (f" · {card.source}" if card.source else "")
                                          + (f"\n_{card.reason}_" if card.reason else ""),
                            },
                        ],
                    },
                ],
            })

        return {
            "msg_type": "interactive",
            "card": {
                "elements": elements,
            },
        }

    # ---- Token Management ----

    def _get_tenant_token(self) -> str:
        """获取飞书 tenant_access_token（缓存至过期）。"""
        if self._tenant_token and time.time() < self._token_expires:
            return self._tenant_token

        url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
        try:
            resp = httpx.post(
                url,
                json={
                    "app_id": self.app_id,
                    "app_secret": self.app_secret,
                },
                timeout=10,
            )
            data = resp.json()
            self._tenant_token = data.get("tenant_access_token", "")
            expire = data.get("expire", 7200)
            self._token_expires = time.time() + expire - 60  # 提前 60s 刷新
            return self._tenant_token
        except Exception:
            logger.exception("Feishu token fetch failed")
            return ""

    # ---- Crypto ----

    def _decrypt(self, encrypted: str) -> str:
        """AES-CBC 解密飞书加密包，返回明文 JSON 字符串。

        key = sha256(encrypt_key)，前 16 字节为 IV，PKCS7 去填充。
        失败返回空字符串（调用方据此降级）。
        """
        try:
            from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
            from cryptography.hazmat.primitives import padding as sym_padding

            key = hashlib.sha256(self.encrypt_key.encode()).digest()
            raw = base64.b64decode(encrypted)
            iv = raw[:16]
            cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
            decryptor = cipher.decryptor()
            decrypted = decryptor.update(raw[16:]) + decryptor.finalize()

            # PKCS7 unpad
            unpadder = sym_padding.PKCS7(128).unpadder()
            plaintext = unpadder.update(decrypted) + unpadder.finalize()
            return plaintext.decode("utf-8")
        except Exception:
            logger.exception("Feishu decrypt failed")
            return ""
