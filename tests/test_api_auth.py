"""AUTH_ENABLED 门禁：开启后缺少/错误 X-API-Key 返回 401，公开端点放行。"""
from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

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
