"""网易云认证 + 歌单导入路由测试。

netease_auth 涉及真实网络，全部用 monkeypatch 打桩，保证测试离线可跑。
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from app.api import auth_routes
from app.api.main import app


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


class TestNeteaseQR:
    @pytest.mark.anyio
    async def test_qr_key(self, client, monkeypatch):
        monkeypatch.setattr(auth_routes.netease_auth, "get_qr_key", lambda: "UNIKEY123")
        resp = await client.get("/auth/netease/qr/key")
        data = resp.json()
        assert data["unikey"] == "UNIKEY123"
        assert "UNIKEY123" in data["qr_url"]

    @pytest.mark.anyio
    async def test_qr_key_failure_graceful(self, client, monkeypatch):
        def _boom():
            raise RuntimeError("network down")

        monkeypatch.setattr(auth_routes.netease_auth, "get_qr_key", _boom)
        resp = await client.get("/auth/netease/qr/key")
        data = resp.json()
        assert data["unikey"] == ""
        assert "error" in data

    @pytest.mark.anyio
    async def test_qr_status_waiting(self, client, monkeypatch):
        monkeypatch.setattr(auth_routes.netease_auth, "check_qr_status", lambda k: {"code": 801, "cookie": None})
        resp = await client.get("/auth/netease/qr/status", params={"unikey": "x"})
        data = resp.json()
        assert data["code"] == 801
        assert "nickname" not in data

    @pytest.mark.anyio
    async def test_qr_status_success_persists(self, client, monkeypatch):
        saved = {}
        monkeypatch.setattr(
            auth_routes.netease_auth,
            "check_qr_status",
            lambda k: {
                "code": 803,
                "cookie": "MUSIC_U=abc",
                "nickname": "小明",
                "avatar": "http://a.jpg",
                "vip_type": 11,
                "vip_label": "黑胶 SVIP",
            },
        )
        monkeypatch.setattr(
            auth_routes.netease_auth,
            "save_cookie",
            lambda uid, cookie, **kw: saved.update({"uid": uid, "cookie": cookie, **kw}),
        )
        resp = await client.get("/auth/netease/qr/status", params={"unikey": "x", "user_id": "u1"})
        data = resp.json()
        assert data["code"] == 803
        assert data["nickname"] == "小明"
        assert data["vip_label"] == "黑胶 SVIP"
        # cookie 已持久化，但不回传明文
        assert "cookie" not in data
        assert saved["uid"] == "u1"
        assert saved["cookie"] == "MUSIC_U=abc"


class TestNeteaseAccount:
    @pytest.mark.anyio
    async def test_account_unbound(self, client, monkeypatch):
        monkeypatch.setattr(auth_routes.netease_auth, "load_cookie", lambda uid: None)
        resp = await client.get("/auth/netease/account", params={"user_id": "nobody"})
        assert resp.json() == {"bound": False}

    @pytest.mark.anyio
    async def test_account_bound_no_cookie_leak(self, client, monkeypatch):
        monkeypatch.setattr(
            auth_routes.netease_auth,
            "load_cookie",
            lambda uid: {"cookie": "SECRET", "nickname": "小红", "vip_type": 0, "vip_label": "非会员"},
        )
        resp = await client.get("/auth/netease/account", params={"user_id": "u1"})
        data = resp.json()
        assert data["bound"] is True
        assert data["nickname"] == "小红"
        assert "cookie" not in data
        assert "SECRET" not in resp.text

    @pytest.mark.anyio
    async def test_unbind(self, client, monkeypatch):
        cleared = []
        monkeypatch.setattr(auth_routes.netease_auth, "clear_cookie", lambda uid: cleared.append(uid))
        resp = await client.post("/auth/netease/unbind", json={"user_id": "u1"})
        assert resp.json() == {"unbound": True}
        assert cleared == ["u1"]


class TestNeteaseImport:
    @pytest.mark.anyio
    async def test_import_missing_ref(self, client):
        resp = await client.post("/playlist/import/netease", json={"user_id": "u1"})
        assert "error" in resp.json()

    @pytest.mark.anyio
    async def test_import_success(self, client, monkeypatch):
        monkeypatch.setattr(auth_routes.netease_auth, "load_cookie", lambda uid: {"cookie": "MUSIC_U=abc"})
        monkeypatch.setattr(
            auth_routes.agent,
            "import_netease_playlist",
            lambda ref, cookie="", user_id=None, limit=200: {
                "name": "我的歌单",
                "total": 50,
                "imported": 48,
                "skipped": 2,
                "tracks": [object()] * 48,
            },
        )
        resp = await client.post(
            "/playlist/import/netease", json={"user_id": "u1", "playlist_ref": "12345", "limit": 50}
        )
        data = resp.json()
        assert data["name"] == "我的歌单"
        assert data["imported"] == 48
        assert "tracks" not in data  # 详情不回传

    @pytest.mark.anyio
    async def test_import_bad_ref_returns_error(self, client, monkeypatch):
        monkeypatch.setattr(auth_routes.netease_auth, "load_cookie", lambda uid: None)

        def _raise(ref, **kw):
            raise ValueError("无法识别歌单链接")

        monkeypatch.setattr(auth_routes.agent, "import_netease_playlist", _raise)
        resp = await client.post("/playlist/import/netease", json={"user_id": "u1", "playlist_ref": "bad"})
        assert resp.json()["error"] == "无法识别歌单链接"


class TestNeteasePlaylistList:
    @pytest.mark.anyio
    async def test_list_not_logged_in(self, client, monkeypatch):
        monkeypatch.setattr(auth_routes.netease_auth, "load_cookie", lambda uid: None)
        resp = await client.get("/playlist/netease/list", params={"user_id": "u1"})
        data = resp.json()
        assert data["playlists"] == []
        assert "error" in data

    @pytest.mark.anyio
    async def test_list_success(self, client, monkeypatch):
        monkeypatch.setattr(auth_routes.netease_auth, "load_cookie", lambda uid: {"cookie": "MUSIC_U=abc"})
        monkeypatch.setattr(
            auth_routes.netease_auth,
            "fetch_user_playlists",
            lambda cookie: [{"id": "1", "name": "夜跑", "cover": "", "count": 30}],
        )
        resp = await client.get("/playlist/netease/list", params={"user_id": "u1"})
        data = resp.json()
        assert len(data["playlists"]) == 1
        assert data["playlists"][0]["name"] == "夜跑"
