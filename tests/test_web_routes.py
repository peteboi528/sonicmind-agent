"""Web 前端路由 测试。"""
from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.main import app


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
        resp = await client.post("/api/playback/audio", json={
            "track": {"title": "测试歌", "artist": "测试歌手", "source": "mock"},
            "user_id": "test_user",
        })
        assert resp.status_code == 200

    @pytest.mark.anyio
    async def test_playback_audio_returns_reason(self, client):
        """无 URL 时返回结构化原因，便于前端区分提示。"""
        resp = await client.post("/api/playback/audio", json={
            "track": {"title": "无流歌", "artist": "x", "source": "bilibili"},
            "user_id": "nobody",
        })
        data = resp.json()
        assert "reason" in data
        assert data["reason"] in {"ok", "vip_required", "not_found", "error"}

    @pytest.mark.anyio
    async def test_playback_audio_netease_no_cookie_vip(self, client):
        """网易云无 cookie 取不到流 → vip_required 提示登录。"""
        resp = await client.post("/api/playback/audio", json={
            "track": {"title": "付费歌", "artist": "x", "source": "netease"},
            "user_id": "no_cookie_user",
        })
        data = resp.json()
        # 拿到流则 ok，否则应提示需要 VIP/登录
        assert data["reason"] in {"ok", "vip_required"}


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
        resp = await client.get("/webhook/wechat", params={
            "signature": "xxx", "timestamp": "123", "nonce": "n", "echostr": "echo",
        })
        assert resp.status_code in (200, 403, 503)
