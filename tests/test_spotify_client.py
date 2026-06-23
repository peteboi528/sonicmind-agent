"""Spotify client 单测：mock HTTP/解析逻辑，不依赖真实网络与凭证。"""
from __future__ import annotations

from app.sources import spotify_client


def test_unavailable_without_credentials():
    assert spotify_client.SpotifyClient("", "").available is False
    assert spotify_client.SpotifyClient("id", "").available is False
    assert spotify_client.SpotifyClient("id", "secret").available is True


def test_search_artist_parses_genres_popularity_image(monkeypatch):
    client = spotify_client.SpotifyClient("id", "secret")
    monkeypatch.setattr(client, "_ensure_token", lambda: "fake-token")
    monkeypatch.setattr(client, "_get", lambda path, **p: {"artists": {"items": [
        {"id": "sp1", "name": "Frank Ocean", "genres": ["r&b", "neo soul"],
         "popularity": 85, "images": [{"url": "img-url"}]},
    ]}})
    hit = client.search_artist("frank ocean")
    assert hit["name"] == "Frank Ocean"
    assert hit["genres"] == ["r&b", "neo soul"]
    assert hit["popularity"] == 85
    assert hit["image"] == "img-url"


def test_search_album_parses_release_date(monkeypatch):
    client = spotify_client.SpotifyClient("id", "secret")
    monkeypatch.setattr(client, "_ensure_token", lambda: "fake-token")
    monkeypatch.setattr(client, "_get", lambda path, **p: {"albums": {"items": [
        {"id": "al1", "name": "Blonde", "artists": [{"name": "Frank Ocean"}],
         "release_date": "2016-08-20", "total_tracks": 17, "images": [{"url": "cover"}]},
    ]}})
    hit = client.search_album("Blonde", "Frank Ocean")
    assert hit["name"] == "Blonde"
    assert hit["release_date"] == "2016-08-20"
    assert hit["total_tracks"] == 17


def test_audio_features_description_averages_and_describes(monkeypatch):
    client = spotify_client.SpotifyClient("id", "secret")
    monkeypatch.setattr(client, "artist_top_track_ids", lambda aid, limit=3: ["t1", "t2"])
    monkeypatch.setattr(client, "audio_features", lambda ids: [
        {"danceability": 0.8, "energy": 0.75, "valence": 0.3, "acousticness": 0.1, "tempo": 120},
        {"danceability": 0.7, "energy": 0.65, "valence": 0.2, "acousticness": 0.2, "tempo": 118},
    ])
    desc = client.audio_features_description("sp1")
    # avg: danceability 0.75>0.7 → 律动感强；energy 0.7 不 >0.7；valence 0.25<0.4 → 情绪偏暗
    assert "律动感强" in desc
    assert "情绪偏暗" in desc
    assert "能量充沛" not in desc
    assert "bpm" in desc


def test_audio_features_description_empty_without_tracks(monkeypatch):
    client = spotify_client.SpotifyClient("id", "secret")
    monkeypatch.setattr(client, "artist_top_track_ids", lambda aid, limit=3: [])
    assert client.audio_features_description("sp1") == ""
