"""入库 URL 治理测试（Issue 5）。

覆盖：scheme/结构校验、SSRF/私网阻断（常开，用字面 IP 不走 DNS）、站点白名单（ALLOW_ANY_URL=false）
后缀匹配、端点 400 集成。
"""
from __future__ import annotations

import pytest

from app.security.ingest_guard import IngestURLError, validate_ingest_url


def _bypass_ssrf(monkeypatch):
    """跳过 DNS（hostname 测试用），只验 scheme/白名单逻辑。SSRF 用字面 IP 单独测，不走这里。"""
    monkeypatch.setattr("app.security.ingest_guard._host_is_private", lambda host: False)


# ---- scheme / 结构 ----

def test_valid_public_url_passes(monkeypatch):
    _bypass_ssrf(monkeypatch)
    monkeypatch.setattr("app.config.settings.allow_any_url", True, raising=False)
    validate_ingest_url("https://example.com/song")  # 不抛即通过
    validate_ingest_url("http://music.163.com/song?id=1")


def test_empty_url_rejected():
    with pytest.raises(IngestURLError):
        validate_ingest_url("")
    with pytest.raises(IngestURLError):
        validate_ingest_url("   ")


def test_too_long_url_rejected(monkeypatch):
    _bypass_ssrf(monkeypatch)
    with pytest.raises(IngestURLError):
        validate_ingest_url("https://example.com/" + "a" * 2100)


@pytest.mark.parametrize("bad", [
    "file:///etc/passwd",
    "ftp://example.com/x",
    "gopher://example.com/x",
    "javascript:alert(1)",
    "data:text/html,x",
    "/etc/passwd",        # 无 scheme
    "//example.com/x",    # 协议相对
])
def test_bad_scheme_rejected(monkeypatch, bad):
    _bypass_ssrf(monkeypatch)
    with pytest.raises(IngestURLError):
        validate_ingest_url(bad)


# ---- SSRF / 私网阻断（常开，字面 IP 不走 DNS）----

@pytest.mark.parametrize("url", [
    "http://127.0.0.1/x",
    "http://10.0.0.1/x",
    "http://192.168.1.1/x",
    "http://172.16.0.1/x",
    "http://169.254.1.1/x",
    "http://localhost/x",
    "http://[::1]/x",
])
def test_private_hosts_blocked(url):
    # 不 bypass SSRF——这是常开校验，ALLOW_ANY_URL=true 也挡
    with pytest.raises(IngestURLError):
        validate_ingest_url(url)


def test_ssrf_blocks_even_when_any_url_true(monkeypatch):
    monkeypatch.setattr("app.config.settings.allow_any_url", True, raising=False)
    with pytest.raises(IngestURLError):
        validate_ingest_url("http://127.0.0.1/secret")


# ---- 白名单（ALLOW_ANY_URL=false）----

def test_allowlist_rejects_non_allowlist_host(monkeypatch):
    _bypass_ssrf(monkeypatch)
    monkeypatch.setattr("app.config.settings.allow_any_url", False, raising=False)
    with pytest.raises(IngestURLError):
        validate_ingest_url("https://evil.example.com/x")


def test_allowlist_accepts_known_hosts(monkeypatch):
    _bypass_ssrf(monkeypatch)
    monkeypatch.setattr("app.config.settings.allow_any_url", False, raising=False)
    validate_ingest_url("https://www.youtube.com/watch?v=x")  # 不抛
    validate_ingest_url("https://music.163.com/song?id=1")
    validate_ingest_url("https://www.bilibili.com/video/BV1")


def test_allowlist_suffix_match(monkeypatch):
    """music.163.com 应命中白名单 163.com（点号后缀匹配，非裸子串）。"""
    _bypass_ssrf(monkeypatch)
    monkeypatch.setattr("app.config.settings.allow_any_url", False, raising=False)
    validate_ingest_url("https://music.163.com/song?id=1")


def test_allowlist_rejects_lookalike_host(monkeypatch):
    """evil163.com 不应命中 163.com（必须点号边界，防 lookalike 绕过）。"""
    _bypass_ssrf(monkeypatch)
    monkeypatch.setattr("app.config.settings.allow_any_url", False, raising=False)
    with pytest.raises(IngestURLError):
        validate_ingest_url("https://evil163.com/x")


# ---- 端点集成：SSRF URL → 400 ----

def test_ingest_endpoint_rejects_ssrf():
    from fastapi.testclient import TestClient

    from app.api.main import app

    client = TestClient(app)
    resp = client.post("/assets/ingest", json={"url": "http://127.0.0.1/secret"})
    assert resp.status_code == 400
    resp_full = client.post("/assets/ingest_full", json={"url": "http://10.0.0.1/x"})
    assert resp_full.status_code == 400
