"""MusicBrainz client 单测：mock HTTP 响应测解析逻辑，不依赖真实网络。"""
from __future__ import annotations

import json

from app.sources import musicbrainz_client


class _FakeResp:
    def __init__(self, payload):
        self._data = json.dumps(payload).encode()

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _patch_urlopen(monkeypatch, payload):
    musicbrainz_client._RESPONSE_CACHE.clear()  # 进程缓存跨测试持久，逐测试清空保隔离
    monkeypatch.setattr(
        musicbrainz_client.urllib.request, "urlopen",
        lambda req, timeout: _FakeResp(payload),
    )


def test_search_artist_parses_score_aliases_tags(monkeypatch):
    _patch_urlopen(monkeypatch, {"artists": [
        {"id": "mb1", "name": "Frank Ocean", "score": "100", "type": "Person",
         "country": "US", "disambiguation": "",
         "aliases": [{"name": "Lonny Breaux"}, {"name": "Christopher Edwin Breaux"}],
         "tags": [{"name": "r&b"}, {"name": "neo soul"}]},
        {"id": "mb2", "name": "Frankie Ocean", "score": "40"},
    ]})
    hits = musicbrainz_client.MusicBrainzClient().search_artist("frank ocean", limit=3)
    assert len(hits) == 2
    top = hits[0]
    assert top["name"] == "Frank Ocean"
    assert top["score"] == 100  # MB score 是字符串，需转 int
    assert "Lonny Breaux" in top["aliases"]
    assert "neo soul" in top["tags"]


def test_resolve_artist_returns_highest_score(monkeypatch):
    _patch_urlopen(monkeypatch, {"artists": [
        {"id": "mb2", "name": "Wrong Artist", "score": "30"},
        {"id": "mb1", "name": "SZA", "score": "98", "type": "Person"},
    ]})
    hit = musicbrainz_client.MusicBrainzClient().resolve_artist("sza")
    assert hit["name"] == "SZA"
    assert hit["mbid"] == "mb1"


def test_search_release_group_parses_date_type_artist(monkeypatch):
    _patch_urlopen(monkeypatch, {"release-groups": [
        {"id": "rg1", "title": "Blonde", "score": "95", "primary-type": "Album",
         "first-release-date": {"date": "2016-08-20"},
         "artist-credit": [{"name": "Frank Ocean", "joinphrase": ""}],
         "tags": [{"name": "alternative r&b"}]},
    ]})
    hits = musicbrainz_client.MusicBrainzClient().search_release_group("Blonde", "Frank Ocean")
    assert hits[0]["title"] == "Blonde"
    assert hits[0]["date"] == "2016-08-20"
    assert hits[0]["type"] == "Album"
    assert "Frank Ocean" in hits[0]["artist"]
    assert "alternative r&b" in hits[0]["tags"]


def test_lookup_release_group_parses_url_relations(monkeypatch):
    _patch_urlopen(monkeypatch, {
        "id": "rg1",
        "title": "OK Computer",
        "primary-type": "Album",
        "first-release-date": "1997-06-16",
        "artist-credit": [{"name": "Radiohead"}],
        "tags": [{"name": "alternative rock"}],
        "relations": [
            {"target-type": "url", "type": "review", "url": {"resource": "https://www.bbc.co.uk/music/reviews/wcp2"}},
            {"target-type": "artist", "type": "member of band", "artist": {"name": "Radiohead"}},
        ],
    })
    hit = musicbrainz_client.MusicBrainzClient().lookup_release_group("rg1")
    assert hit["title"] == "OK Computer"
    assert hit["relations"] == [{
        "type": "review",
        "url": "https://www.bbc.co.uk/music/reviews/wcp2",
        "ended": "",
    }]


def test_lookup_artist_parses_bbc_relation(monkeypatch):
    _patch_urlopen(monkeypatch, {
        "id": "artist1",
        "name": "Radiohead",
        "type": "Group",
        "country": "GB",
        "relations": [
            {"target-type": "url", "type": "BBC Music page", "ended": True, "url": {"resource": "https://www.bbc.co.uk/music/artists/artist1"}},
        ],
    })
    hit = musicbrainz_client.MusicBrainzClient().lookup_artist("artist1")
    assert hit["name"] == "Radiohead"
    assert hit["relations"][0]["type"] == "BBC Music page"
    assert hit["relations"][0]["ended"] == "true"


def test_get_returns_empty_on_network_failure(monkeypatch):
    monkeypatch.setattr(
        musicbrainz_client.urllib.request, "urlopen",
        lambda req, timeout: (_ for _ in ()).throw(OSError("network down")),
    )
    data = musicbrainz_client._get("artist/", query="x")
    assert data == {}


def test_empty_or_missing_query_returns_empty():
    client = musicbrainz_client.MusicBrainzClient()
    assert client.search_artist("") == []
    assert client.search_release_group("   ") == []
    assert client.resolve_artist("") is None


def test_resolve_release_group_prefers_exact_title_over_higher_fuzzy_score(monkeypatch):
    # 裸标题 "Blonde"：Bob Dylan 的 "Blonde on Blonde" 模糊分更高，但精确名应胜出。
    _patch_urlopen(monkeypatch, {"release-groups": [
        {"id": "wrong", "title": "Blonde on Blonde", "score": "100",
         "artist-credit": [{"name": "Bob Dylan"}], "primary-type": "Album"},
        {"id": "right", "title": "Blonde", "score": "92",
         "artist-credit": [{"name": "Frank Ocean"}], "primary-type": "Album"},
    ]})
    hit = musicbrainz_client.MusicBrainzClient().resolve_release_group("Blonde")
    assert hit["title"] == "Blonde"
    assert hit["mbid"] == "right"


def test_resolve_release_group_uses_artist_to_disambiguate_same_title(monkeypatch):
    _patch_urlopen(monkeypatch, {"release-groups": [
        {"id": "dylan", "title": "Blonde", "score": "99",
         "artist-credit": [{"name": "Other Guy"}], "primary-type": "Album"},
        {"id": "ocean", "title": "Blonde", "score": "80",
         "artist-credit": [{"name": "Frank Ocean"}], "primary-type": "Album"},
    ]})
    hit = musicbrainz_client.MusicBrainzClient().resolve_release_group("Blonde", "Frank Ocean")
    assert hit["mbid"] == "ocean"


def test_resolve_artist_prefers_exact_name_over_higher_fuzzy_score(monkeypatch):
    _patch_urlopen(monkeypatch, {"artists": [
        {"id": "redhead", "name": "Blonde Redhead", "score": "100", "type": "Group"},
        {"id": "exact", "name": "Blonde", "score": "70", "type": "Group"},
    ]})
    hit = musicbrainz_client.MusicBrainzClient().resolve_artist("Blonde")
    assert hit["name"] == "Blonde"
    assert hit["mbid"] == "exact"
