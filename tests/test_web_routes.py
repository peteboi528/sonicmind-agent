"""Web 前端路由 测试。"""

from __future__ import annotations

import io
import uuid

import pytest
from httpx import ASGITransport, AsyncClient

from app import netease_auth
from app.api.main import agent, app
from app.config import settings
from app.models import Asset
from app.services.cover_recognizer import CoverRecognition


def _tiny_png(size=(8, 8), color=(255, 128, 0)) -> bytes:
    """Pillow 现造小 PNG，供封面上传测试用。"""
    from PIL import Image

    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


class TestWebRoutes:
    @pytest.mark.anyio
    async def test_serve_index(self, client):
        resp = await client.get("/web")
        assert resp.status_code == 200
        # Vue SPA：入口含挂载点和标题
        assert 'id="app"' in resp.text
        assert "SONICMIND" in resp.text

    @pytest.mark.anyio
    async def test_serve_index_trailing_slash(self, client):
        resp = await client.get("/web/")
        assert resp.status_code == 200

    @pytest.mark.anyio
    async def test_serve_assets_blocks_traversal(self, client):
        """资源路由必须防目录穿越。"""
        resp = await client.get("/web/assets/../../config.py")
        assert resp.status_code == 404

    @pytest.mark.anyio
    async def test_serve_missing_asset_404(self, client):
        resp = await client.get("/web/assets/nonexistent-xyz.js")
        assert resp.status_code == 404

    @pytest.mark.anyio
    async def test_playback_audio_no_body(self, client):
        resp = await client.post("/api/playback/audio", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert "url" in data

    @pytest.mark.anyio
    async def test_playback_mv_no_body(self, client):
        resp = await client.post("/api/playback/mv", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert "url" in data

    @pytest.mark.anyio
    async def test_playback_audio_with_track(self, client):
        resp = await client.post(
            "/api/playback/audio",
            json={
                "track": {"title": "测试歌", "artist": "测试歌手", "source": "mock"},
                "user_id": "test_user",
            },
        )
        assert resp.status_code == 200

    @pytest.mark.anyio
    async def test_playback_audio_does_not_persist_asset(self, client, monkeypatch):
        """播放 ≠ 入库：playback 成功只回一个逻辑 asset_id 供收听采集，不能把歌写进库。"""
        monkeypatch.setattr(agent, "get_audio_url", lambda *_args, **_kwargs: "https://cdn.example.com/test.mp3")
        source_id = f"manual-{uuid.uuid4().hex[:10]}"
        resp = await client.post(
            "/api/playback/audio",
            json={
                "track": {
                    "title": "播放测试歌",
                    "artist": "测试歌手",
                    "source": "netease",
                    "source_id": source_id,
                    "cover_url": "https://img.example.com/cover.jpg",
                },
                "user_id": "test_user",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["reason"] == "ok"
        assert data["asset_id"]  # 有逻辑 id 给前端做 listen keying

        # 但库里不该有这条——播放没入库
        asset = agent.store.read_model("assets", data["asset_id"], Asset)
        assert asset is None

    @pytest.mark.anyio
    async def test_playback_audio_returns_reason(self, client):
        """无 URL 时返回结构化原因，便于前端区分提示。"""
        resp = await client.post(
            "/api/playback/audio",
            json={
                "track": {"title": "无流歌", "artist": "x", "source": "bilibili"},
                "user_id": "nobody",
            },
        )
        data = resp.json()
        assert "reason" in data
        assert data["reason"] in {"ok", "vip_required", "not_found", "error"}

    @pytest.mark.anyio
    async def test_playback_audio_netease_no_cookie_vip(self, client):
        """网易云无 cookie 取不到流 → vip_required 提示登录。"""
        resp = await client.post(
            "/api/playback/audio",
            json={
                "track": {"title": "付费歌", "artist": "x", "source": "netease"},
                "user_id": "no_cookie_user",
            },
        )
        data = resp.json()
        # 拿到流则 ok，否则应提示需要 VIP/登录
        assert data["reason"] in {"ok", "vip_required"}

    @pytest.mark.anyio
    async def test_playback_audio_anonymous_ignores_body_user_id(self, client, monkeypatch):
        """匿名模式（AUTH_ENABLED=false）下播放代理不能按 body user_id 加载他人 cookie。"""
        loaded = []

        def spy_load(user_id):  # noqa: ARG001
            loaded.append(user_id)
            return None

        monkeypatch.setattr(netease_auth, "load_cookie", spy_load)

        resp = await client.post(
            "/api/playback/audio",
            json={
                "track": {"title": "x", "artist": "y", "source": "netease"},
                "user_id": "victim_user",
            },
        )
        assert resp.status_code == 200
        assert loaded == ["web_user"]


class TestIdentifyAlbum:
    """上传专辑封面识别端点（multipart）。recognize 全程 mock，离线。"""

    @pytest.mark.anyio
    async def test_identify_ok_returns_query_and_thumbnail(self, client, monkeypatch):
        async def fake_recognize(_b, _mime):  # noqa: ARG001
            return CoverRecognition(album="Blonde", artist="Frank Ocean", confidence=0.92, method="vision")

        monkeypatch.setattr("app.api.web_routes.recognize_album_cover", fake_recognize)
        monkeypatch.setattr("app.api.web_routes.build_thumbnail_data_url", lambda _b: "data:image/jpeg;base64,AAAA")

        resp = await client.post(
            "/api/identify-album",
            files={"file": ("cover.png", _tiny_png(), "image/png")},
            data={"user_id": "u1"},
        )
        assert resp.status_code == 200
        d = resp.json()
        assert d["recognized"]["album"] == "Blonde"
        assert d["recognized"]["artist"] == "Frank Ocean"
        # 结构化多行 query（album\nBlonde\nFrank Ocean\n解读这张专辑），可被 resolve_music_entities 解析
        assert d["query"].startswith("album\n") and "Blonde" in d["query"]
        assert d["thumbnail_url"].startswith("data:image/")
        assert d["user_id"] == "u1"

    @pytest.mark.anyio
    async def test_identify_none_query_is_null(self, client, monkeypatch):
        async def fake_recognize(_b, _mime):  # noqa: ARG001
            return CoverRecognition(method="none", note="没认出")

        monkeypatch.setattr("app.api.web_routes.recognize_album_cover", fake_recognize)
        monkeypatch.setattr("app.api.web_routes.build_thumbnail_data_url", lambda _b: "data:image/jpeg;base64,A")

        resp = await client.post(
            "/api/identify-album",
            files={"file": ("cover.png", _tiny_png(), "image/png")},
            data={"user_id": "u"},
        )
        d = resp.json()
        assert d["query"] is None
        assert d["recognized"]["method"] == "none"

    @pytest.mark.anyio
    async def test_identify_415_wrong_type(self, client):
        resp = await client.post(
            "/api/identify-album",
            files={"file": ("x.gif", b"abc", "image/gif")},
            data={"user_id": "u"},
        )
        assert resp.status_code == 415

    @pytest.mark.anyio
    async def test_identify_400_empty_file(self, client, monkeypatch):
        # 类型合法但内容为空 → 400（在校验读取之后）
        resp = await client.post(
            "/api/identify-album",
            files={"file": ("empty.png", b"", "image/png")},
            data={"user_id": "u"},
        )
        assert resp.status_code == 400

    @pytest.mark.anyio
    async def test_identify_413_too_large(self, client, monkeypatch):
        monkeypatch.setattr(settings, "album_cover_max_bytes", 8)
        resp = await client.post(
            "/api/identify-album",
            files={"file": ("big.png", b"0123456789abcdef", "image/png")},
            data={"user_id": "u"},
        )
        assert resp.status_code == 413


class TestBotRoutes:
    @pytest.mark.anyio
    async def test_feishu_webhook_not_configured(self, client):
        """未配置飞书凭证时返回错误提示。"""
        resp = await client.post("/webhook/feishu", json={"type": "event"})
        # 可能是 200 {"error": "..."} 或正常处理
        assert resp.status_code in (200, 503)

    @pytest.mark.anyio
    async def test_wechat_verify_not_configured(self, client):
        """未配置微信凭证时返回 503。"""
        resp = await client.get(
            "/webhook/wechat",
            params={
                "signature": "xxx",
                "timestamp": "123",
                "nonce": "n",
                "echostr": "echo",
            },
        )
        assert resp.status_code in (200, 403, 503)
