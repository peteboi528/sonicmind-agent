"""Discogs client 单测：mock search 结果解析，不依赖真实网络与 token。"""
from __future__ import annotations

from app.sources import discogs_client


def test_unavailable_without_token():
    assert discogs_client.DiscogsClient("").available is False
    assert discogs_client.DiscogsClient("tok").available is True


def test_resolve_release_parses_year_styles(monkeypatch):
    client = discogs_client.DiscogsClient("tok")
    monkeypatch.setattr(client, "_get", lambda path, **p: {"results": [
        {"id": 123, "title": "Frank Ocean — Blonde", "year": 2016,
         "genre": ["Electronic"], "style": ["Alternative R&B", "Neo Soul"], "type": "master"},
    ]} if "search" in path else {})
    hit = client.resolve_release("Blonde", "Frank Ocean")
    assert hit["title"] == "Frank Ocean — Blonde"
    assert hit["year"] == 2016
    assert "Alternative R&B" in hit["styles"]
    assert hit["type"] == "master"


def test_resolve_release_falls_back_to_release_type(monkeypatch):
    client = discogs_client.DiscogsClient("tok")
    calls = []

    def fake_get(path, **p):
        calls.append(p.get("type"))
        # 第一次 master 查询空，第二次 release 命中
        if p.get("type") == "master":
            return {}
        return {"results": [{"id": 9, "title": "X — Y", "year": 2001, "style": ["Rock"]}]}

    monkeypatch.setattr(client, "_get", fake_get)
    hit = client.resolve_release("Y", "X")
    assert hit["year"] == 2001
    assert "master" in calls and "release" in calls


def test_resolve_artist_parses_styles(monkeypatch):
    client = discogs_client.DiscogsClient("tok")
    monkeypatch.setattr(client, "_get", lambda path, **p: {"results": [
        {"id": 7, "title": "SZA", "genre": ["Pop"], "style": ["Contemporary R&B"]},
    ]} if "search" in path else {})
    hit = client.resolve_artist("sza")
    assert hit["name"] == "SZA"
    assert "Contemporary R&B" in hit["styles"]


def test_resolve_returns_none_on_empty(monkeypatch):
    client = discogs_client.DiscogsClient("tok")
    monkeypatch.setattr(client, "_get", lambda path, **p: {})
    assert client.resolve_release("nothing") is None
    assert client.resolve_artist("nobody") is None
