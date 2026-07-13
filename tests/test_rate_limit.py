"""限流中间件 + 令牌桶测试。"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api import main as main_module
from app.rate_limit import RateLimiter, TokenBucket

# ---- 单元：令牌桶 / 限流器 ----


def test_token_bucket_burst_then_deny():
    b = TokenBucket(rpm=60, capacity=2)
    assert b.allow()[0]
    assert b.allow()[0]
    ok, retry = b.allow()
    assert ok is False
    assert retry > 0


def test_rate_limiter_per_key_isolation():
    rl = RateLimiter({"chat": 1})
    assert rl.acquire("chat", "userA")[0]
    assert not rl.acquire("chat", "userA")[0]  # userA 用尽
    assert rl.acquire("chat", "userB")[0]  # userB 独立桶


def test_rate_limiter_unknown_tier_passthrough():
    rl = RateLimiter({"chat": 1})
    assert rl.acquire("playback", "u")[0]  # 未配置档直接放行


# ---- 集成：中间件 429 ----


@pytest.fixture
def _fast_chat(monkeypatch):
    """mock 掉 agent.chat_async，避免 /chat 跑真实图编排。"""

    async def _fast(*a, **kw):
        return {"answer": "ok"}

    monkeypatch.setattr(main_module.agent, "chat_async", _fast)


def test_middleware_returns_429(monkeypatch, _fast_chat):
    monkeypatch.setattr(main_module.settings, "rate_limit_enabled", True)
    monkeypatch.setattr(main_module, "_rate_limiter", RateLimiter({"chat": 1, "playback": 1}))
    client = TestClient(main_module.app)

    r1 = client.post("/chat", json={"user_id": "u", "message": "hi"})
    r2 = client.post("/chat", json={"user_id": "u", "message": "hi"})
    assert r1.status_code != 429
    assert r2.status_code == 429
    assert "Retry-After" in r2.headers
    assert r2.json()["detail"] == "rate limited"


def test_middleware_disabled(monkeypatch, _fast_chat):
    monkeypatch.setattr(main_module.settings, "rate_limit_enabled", False)
    monkeypatch.setattr(main_module, "_rate_limiter", RateLimiter({"chat": 1}))
    client = TestClient(main_module.app)

    for _ in range(5):
        r = client.post("/chat", json={"user_id": "u", "message": "hi"})
        assert r.status_code != 429


def test_middleware_playback_tier(monkeypatch):
    monkeypatch.setattr(main_module.settings, "rate_limit_enabled", True)
    monkeypatch.setattr(main_module, "_rate_limiter", RateLimiter({"chat": 100, "playback": 1}))
    monkeypatch.setattr(main_module.agent, "get_audio_url", lambda *a, **kw: None)
    client = TestClient(main_module.app)
    body = {"track": {"title": "t", "source": "netease"}, "user_id": "u"}

    r1 = client.post("/api/playback/audio", json=body)
    r2 = client.post("/api/playback/audio", json=body)
    assert r1.status_code != 429
    assert r2.status_code == 429  # playback 档独立配额，第 2 次超限


def test_middleware_keys_per_user_api_key(monkeypatch, _fast_chat):
    """限流应看到鉴权中间件绑定的用户，而不是把所有 key 合并到同一 IP 桶。"""
    monkeypatch.setattr(main_module.settings, "auth_enabled", True)
    monkeypatch.setattr(main_module.settings, "user_api_keys", {"key-a": "alice", "key-b": "bob"})
    monkeypatch.setattr(main_module.settings, "rate_limit_enabled", True)
    monkeypatch.setattr(main_module, "_rate_limiter", RateLimiter({"chat": 1}))
    client = TestClient(main_module.app)

    assert client.post("/chat", json={"message": "hi"}, headers={"X-API-Key": "key-a"}).status_code != 429
    assert client.post("/chat", json={"message": "hi"}, headers={"X-API-Key": "key-b"}).status_code != 429
    assert client.post("/chat", json={"message": "hi"}, headers={"X-API-Key": "key-a"}).status_code == 429
