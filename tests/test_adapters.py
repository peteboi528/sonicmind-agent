"""Bot 适配器协议层 + 飞书/微信解析 测试。"""
from __future__ import annotations

import hashlib
import json

import pytest

from app.adapters.base import answer_to_bot_response, stream_events_to_bot_response
from app.adapters.feishu_adapter import FeishuAdapter
from app.adapters.protocol import BotResponse, IncomingMessage, SongCard
from app.adapters.wechat_adapter import WeChatAdapter
from app.models import AgentAnswer, Segment, StreamEvent

# ---- Protocol 数据类 ----


class TestProtocol:
    def test_incoming_message_defaults(self):
        msg = IncomingMessage(user_id="u1", content="hello")
        assert msg.user_id == "u1"
        assert msg.raw == {}
        assert msg.message_id == ""

    def test_song_card_defaults(self):
        card = SongCard(title="Song", artist="Artist")
        assert card.cover_url == ""
        assert card.score is None

    def test_bot_response_defaults(self):
        resp = BotResponse()
        assert resp.text == ""
        assert resp.cards == []


# ---- Base 工具 ----


class TestBaseConversion:
    def test_answer_to_bot_response_empty(self):
        answer = AgentAnswer(answer="Hello!", evidences=[], recommended_segments=[])
        resp = answer_to_bot_response(answer)
        assert resp.text == "Hello!"
        assert resp.cards == []

    def test_answer_to_bot_response_with_segments(self):
        seg = Segment(
            segment_id="s1",
            asset_id="a1",
            start_seconds=0,
            end_seconds=30,
            transcript="nice beat",
            scene_summary="测试歌曲",
            audio_tags=["pop", "happy"],
        )
        answer = AgentAnswer(answer="推荐以下歌曲", evidences=[], recommended_segments=[seg])
        resp = answer_to_bot_response(answer)
        assert resp.text == "推荐以下歌曲"
        assert len(resp.cards) == 1
        assert resp.cards[0].title == "测试歌曲"
        assert "pop" in resp.cards[0].artist

    def test_stream_events_to_bot_response(self):
        events = [
            StreamEvent(type="thinking", content="规划中..."),
            StreamEvent(type="candidates", payload={
                "cards": [
                    {"title": "夜曲", "artist": "周杰伦", "source": "netease", "reason": "经典"},
                ]
            }),
            StreamEvent(type="final", content="为你推荐了一首歌"),
        ]
        resp = stream_events_to_bot_response(events)
        assert resp.text == "为你推荐了一首歌"
        assert len(resp.cards) == 1
        assert resp.cards[0].title == "夜曲"

    def test_stream_events_song_card(self):
        events = [
            StreamEvent(type="song_card", payload={
                "title": "晴天", "artist": "周杰伦", "cover_url": "http://x.com/c.jpg",
            }),
            StreamEvent(type="final", content="OK"),
        ]
        resp = stream_events_to_bot_response(events)
        assert len(resp.cards) == 1
        assert resp.cards[0].title == "晴天"
        assert resp.cards[0].cover_url == "http://x.com/c.jpg"


# ---- 飞书适配器 ----


class TestFeishuAdapter:
    @pytest.fixture
    def adapter(self):
        return FeishuAdapter(
            app_id="cli_test",
            app_secret="secret_test",
            verification_token="token_test",
            encrypt_key="",
        )

    def test_handle_challenge(self, adapter):
        body = json.dumps({
            "type": "url_verification",
            "challenge": "abc123",
            "token": "token_test",
        }).encode()
        result = adapter.handle_challenge(body)
        assert result == {"challenge": "abc123"}

    def test_handle_challenge_non_verification(self, adapter):
        body = json.dumps({"type": "event_callback"}).encode()
        result = adapter.handle_challenge(body)
        assert result is None

    def test_parse_text_message(self, adapter):
        body = json.dumps({
            "event": {
                "message": {
                    "message_type": "text",
                    "content": json.dumps({"text": "推荐几首歌"}),
                    "message_id": "msg_001",
                    "chat_id": "oc_xxx",
                },
                "sender": {
                    "sender_id": {"open_id": "ou_abc123"},
                    "sender_type": "user",
                },
            },
        }).encode()
        msg = adapter.parse_request(body, {})
        assert msg is not None
        assert msg.content == "推荐几首歌"
        assert msg.user_id == "feishu_ou_abc123"
        assert msg.message_id == "msg_001"

    def test_parse_non_text_message(self, adapter):
        body = json.dumps({
            "event": {
                "message": {
                    "message_type": "image",
                    "content": "{}",
                    "message_id": "msg_002",
                },
                "sender": {"sender_id": {"open_id": "ou_x"}},
            },
        }).encode()
        msg = adapter.parse_request(body, {})
        assert msg is None

    def test_format_response_text_only(self, adapter):
        resp = BotResponse(text="你好！")
        result = adapter.format_response(resp)
        assert result["msg_type"] == "text"

    def test_format_response_with_cards(self, adapter):
        resp = BotResponse(
            text="推荐歌曲：",
            cards=[
                SongCard(title="夜曲", artist="周杰伦", source="netease"),
                SongCard(title="晴天", artist="周杰伦"),
            ],
        )
        result = adapter.format_response(resp)
        assert result["msg_type"] == "interactive"
        assert "card" in result


# ---- 飞书验签（安全） ----


class TestFeishuVerification:
    def test_verify_signature_valid(self):
        adapter = FeishuAdapter(
            app_id="cli", app_secret="s", verification_token="tok",
            encrypt_key="enc_key_123",
        )
        body = b'{"encrypt":"xxx"}'
        ts, nonce = "1700000000", "nonce_abc"
        raw = ts.encode() + nonce.encode() + b"enc_key_123" + body
        sig = hashlib.sha256(raw).hexdigest()
        headers = {
            "X-Lark-Signature": sig,
            "X-Lark-Request-Timestamp": ts,
            "X-Lark-Request-Nonce": nonce,
        }
        assert adapter.verify_request(body, headers) is True

    def test_verify_signature_invalid_rejected(self):
        adapter = FeishuAdapter(
            app_id="cli", app_secret="s", verification_token="tok",
            encrypt_key="enc_key_123",
        )
        headers = {
            "X-Lark-Signature": "deadbeef",
            "X-Lark-Request-Timestamp": "1700000000",
            "X-Lark-Request-Nonce": "nonce_abc",
        }
        assert adapter.verify_request(b'{"encrypt":"xxx"}', headers) is False

    def test_verify_token_path_when_no_signature(self):
        """无签名头时回退到 verification_token 校验。"""
        adapter = FeishuAdapter(
            app_id="cli", app_secret="s", verification_token="tok_secret",
            encrypt_key="",
        )
        good = json.dumps({"header": {"token": "tok_secret"}, "type": "event"}).encode()
        bad = json.dumps({"header": {"token": "wrong"}}).encode()
        assert adapter.verify_request(good, {}) is True
        assert adapter.verify_request(bad, {}) is False

    def test_verify_token_v1_top_level(self):
        adapter = FeishuAdapter(
            app_id="cli", app_secret="s", verification_token="tok_secret",
            encrypt_key="",
        )
        v1 = json.dumps({"token": "tok_secret", "type": "event_callback"}).encode()
        assert adapter.verify_request(v1, {}) is True

    def test_no_credentials_allows_dev_mode(self):
        """既没配 encrypt_key 也没配 token → 放行（本地开发），不抛异常。"""
        adapter = FeishuAdapter(
            app_id="cli", app_secret="s", verification_token="", encrypt_key="",
        )
        assert adapter.verify_request(b'{"type":"event"}', {}) is True

    def test_encrypted_body_decrypt_roundtrip(self):
        """加密 body 能被 _decode_body 解密还原成事件 dict。"""
        import base64 as _b64
        import os as _os

        from cryptography.hazmat.primitives import padding as sym_padding
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

        encrypt_key = "my_encrypt_key"
        adapter = FeishuAdapter(
            app_id="cli", app_secret="s", verification_token="tok",
            encrypt_key=encrypt_key,
        )
        plaintext = json.dumps({
            "type": "url_verification", "challenge": "chal_42", "token": "tok",
        }).encode()
        key = hashlib.sha256(encrypt_key.encode()).digest()
        iv = _os.urandom(16)
        padder = sym_padding.PKCS7(128).padder()
        padded = padder.update(plaintext) + padder.finalize()
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
        enc = cipher.encryptor()
        ciphertext = enc.update(padded) + enc.finalize()
        encrypt_b64 = _b64.b64encode(iv + ciphertext).decode()

        body = json.dumps({"encrypt": encrypt_b64}).encode()
        result = adapter.handle_challenge(body)
        assert result == {"challenge": "chal_42"}


# ---- 微信适配器 ----


class TestWeChatAdapter:
    @pytest.fixture
    def adapter(self):
        return WeChatAdapter(
            token="test_token",
            app_id="wx_test",
            app_secret="secret_test",
        )

    def test_verify_signature_valid(self, adapter):
        import hashlib
        parts = sorted(["test_token", "1234567890", "nonce123"])
        sig = hashlib.sha1("".join(parts).encode()).hexdigest()
        assert adapter.verify_signature("1234567890", "nonce123", sig)

    def test_verify_signature_invalid(self, adapter):
        assert not adapter.verify_signature("1234", "nonce", "badsignature")

    def test_parse_text_xml(self, adapter):
        xml = (
            '<xml>'
            '<ToUserName><![CDATA[gh_abc]]></ToUserName>'
            '<FromUserName><![CDATA[o_user123]]></FromUserName>'
            '<CreateTime>1700000000</CreateTime>'
            '<MsgType><![CDATA[text]]></MsgType>'
            '<Content><![CDATA[推荐好听的歌]]></Content>'
            '<MsgId>123456</MsgId>'
            '</xml>'
        )
        msg = adapter.parse_request(xml.encode(), {})
        assert msg is not None
        assert msg.content == "推荐好听的歌"
        assert msg.user_id == "wechat_o_user123"
        assert msg.message_id == "123456"

    def test_parse_non_text_xml(self, adapter):
        xml = (
            '<xml>'
            '<MsgType><![CDATA[image]]></MsgType>'
            '<FromUserName><![CDATA[o_user]]></FromUserName>'
            '<ToUserName><![CDATA[gh_abc]]></ToUserName>'
            '</xml>'
        )
        msg = adapter.parse_request(xml.encode(), {})
        assert msg is None

    def test_build_news_json(self, adapter):
        resp = BotResponse(
            text="推荐",
            cards=[
                SongCard(title="夜曲", artist="周杰伦", cover_url="http://x.com/c.jpg"),
            ],
        )
        result = adapter._build_news_json("o_user123", resp)
        assert result["msgtype"] == "news"
        assert len(result["news"]["articles"]) == 1
        assert result["news"]["articles"][0]["title"] == "夜曲"

    def test_build_text_json(self, adapter):
        result = adapter._build_text_json("o_user123", "你好")
        assert result["msgtype"] == "text"
        assert result["text"]["content"] == "你好"
