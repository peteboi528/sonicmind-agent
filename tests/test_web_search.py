"""web_search 单测：fetch_url_content (Tavily Extract) 解析逻辑，不打真网。"""

from __future__ import annotations

import json

from app.sources import web_search


class _FakeResp:
    def __init__(self, payload):
        self._data = json.dumps(payload).encode()

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_fetch_url_content_parses_and_truncates(monkeypatch):
    long_text = "Blonde is critically acclaimed. " * 200
    monkeypatch.setattr(
        web_search.urllib.request,
        "urlopen",
        lambda req, timeout: _FakeResp({"results": [{"raw_content": long_text}]}),
    )
    out = web_search.fetch_url_content("https://www.last.fm/x", api_key="k", max_chars=120)
    assert out.startswith("Blonde is critically")
    assert len(out) <= 122  # 截断 + 省略号
    assert out.endswith("…")


def test_fetch_url_content_prefers_text_field(monkeypatch):
    """新版 Tavily Extract 用 text 字段；优先取 text。"""
    monkeypatch.setattr(
        web_search.urllib.request,
        "urlopen",
        lambda req, timeout: _FakeResp({"results": [{"text": "real body", "raw_content": "ignored"}]}),
    )
    assert web_search.fetch_url_content("https://x", api_key="k") == "real body"


def test_fetch_url_content_returns_empty_on_failure(monkeypatch):
    def _boom(req, timeout):
        raise RuntimeError("blocked by anti-scrape")

    monkeypatch.setattr(web_search.urllib.request, "urlopen", _boom)
    assert web_search.fetch_url_content("https://x", api_key="k") == ""


def test_fetch_url_content_empty_without_key():
    assert web_search.fetch_url_content("https://x", api_key="") == ""
    assert web_search.fetch_url_content("", api_key="k") == ""


def test_fetch_url_content_handles_empty_or_failed_results(monkeypatch):
    monkeypatch.setattr(
        web_search.urllib.request,
        "urlopen",
        lambda req, timeout: _FakeResp({"results": [], "failed_results": [{"url": "https://x"}]}),
    )
    assert web_search.fetch_url_content("https://x", api_key="k") == ""
