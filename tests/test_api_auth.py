"""AUTH_ENABLED 门禁：开启后缺少/错误 X-API-Key 返回 401，公开端点放行。"""
from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from app import netease_auth
from app.api import main as main_module


@pytest.fixture
def auth_client(monkeypatch):
    # 中间件在请求时读取 settings 属性，monkeypatch 即时生效、用例结束自动还原。
    monkeypatch.setattr(main_module.settings, "auth_enabled", True)
    monkeypatch.setattr(main_module.settings, "api_key", "secret-xyz")
    monkeypatch.setattr(main_module.settings, "user_api_keys", {})
    return TestClient(main_module.app)


def test_missing_key_rejected(auth_client):
    r = auth_client.post("/chat", json={"user_id": "u", "message": "hi"})
    assert r.status_code == 401


def test_wrong_key_rejected(auth_client):
    r = auth_client.post(
        "/chat",
        json={"user_id": "u", "message": "hi"},
        headers={"X-API-Key": "wrong"},
    )
    assert r.status_code == 401


def test_correct_key_passes_gate(auth_client):
    # 正确 key 应通过门禁（不再返回 401）；不验证 /chat 业务内容，只验证放行。
    r = auth_client.post(
        "/chat",
        json={"user_id": "u", "message": "你好"},
        headers={"X-API-Key": "secret-xyz"},
    )
    assert r.status_code != 401


def test_health_open_without_key(auth_client):
    r = auth_client.get("/health")
    assert r.status_code == 200


def test_per_user_key_overrides_client_user_id(monkeypatch):
    bound_user = f"auth-alice-{uuid.uuid4().hex}"
    victim_user = f"victim-{uuid.uuid4().hex}"
    monkeypatch.setattr(main_module.settings, "auth_enabled", True)
    monkeypatch.setattr(main_module.settings, "api_key", "")
    monkeypatch.setattr(main_module.settings, "user_api_keys", {"alice-key": bound_user})
    client = TestClient(main_module.app)

    r = client.post(
        "/memory/update",
        json={"user_id": victim_user, "event": "我喜欢 city pop"},
        headers={"X-API-Key": "alice-key"},
    )

    assert r.status_code == 200
    assert main_module.agent.memory.get_memory(bound_user).preferences
    assert not main_module.agent.memory.get_memory(victim_user).preferences


def test_playback_user_id_bound_to_auth_user(monkeypatch):
    """播放代理在鉴权模式下必须用 API key 绑定的用户加载 cookie，不能用 body user_id 伪造。"""
    bound_user = f"auth-alice-{uuid.uuid4().hex}"
    victim_user = f"victim-{uuid.uuid4().hex}"
    monkeypatch.setattr(main_module.settings, "auth_enabled", True)
    monkeypatch.setattr(main_module.settings, "api_key", "")
    monkeypatch.setattr(main_module.settings, "user_api_keys", {"alice-key": bound_user})

    loaded = []
    def spy_load(user_id):
        loaded.append(user_id)
        return None
    monkeypatch.setattr(netease_auth, "load_cookie", spy_load)

    client = TestClient(main_module.app)
    r = client.post(
        "/api/playback/audio",
        json={"track": {"title": "x", "artist": "y", "source": "netease"}, "user_id": victim_user},
        headers={"X-API-Key": "alice-key"},
    )
    assert r.status_code == 200
    assert loaded == [bound_user]


def test_web_static_open_under_auth(auth_client):
    # 开鉴权时前端静态资源必须放行——浏览器无法自定义 X-API-Key 头，
    # 若连 /web 都拦则「开鉴权 = 前端不可用」。不依赖 dist 是否构建：不被门禁拦（≠401）即可。
    assert auth_client.get("/web").status_code != 401
    assert auth_client.get("/web/assets/nonexistent.css").status_code != 401


def test_webhook_not_accidentally_public_under_auth(auth_client):
    """/web 静态资源白名单不能扩大成 /webhook 前缀白名单。"""
    assert auth_client.post("/webhook/feishu", json={}).status_code == 401
