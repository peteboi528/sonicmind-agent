"""网易云歌单搜索：搜情绪/风格歌单，从中提取高质量真实歌曲。

网易云歌曲搜索对模糊的情绪词效果很差（"慵懒 R&B" 返回随机歌），
但歌单搜索很好——因为歌单是真人策划的，歌曲质量有保障。
"""
from __future__ import annotations

import logging
from typing import Any

import requests

from app.models import ExternalTrack

logger = logging.getLogger(__name__)

_NETEASE_SEARCH_URL = "https://music.163.com/api/search/get/web"
_NETEASE_PLAYLIST_URL = "https://music.163.com/api/v6/playlist/detail"
_HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}


def search_netease_playlists(query: str, limit: int = 5) -> list[dict[str, Any]]:
    """搜索网易云歌单，返回 [{id, name, track_count}, ...]。"""
    try:
        resp = requests.get(
            _NETEASE_SEARCH_URL,
            params={"s": query, "type": 1000, "limit": limit},
            headers=_HEADERS,
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        logger.debug("Netease playlist search failed for %r", query, exc_info=True)
        return []

    if not isinstance(data, dict):
        logger.debug("Netease playlist search returned non-dict payload for %r: %r", query, type(data).__name__)
        return []
    result = data.get("result")
    if not isinstance(result, dict):
        logger.debug("Netease playlist search returned invalid result payload for %r: %r", query, type(result).__name__)
        return []
    playlists = result.get("playlists") or []
    if not isinstance(playlists, list):
        logger.debug("Netease playlist search returned invalid playlists payload for %r: %r", query, type(playlists).__name__)
        return []
    return [
        {
            "id": pl["id"],
            "name": pl.get("name", ""),
            "track_count": pl.get("trackCount", 0),
        }
        for pl in playlists
    ]


def get_playlist_tracks(playlist_id: int, limit: int = 30) -> list[ExternalTrack]:
    """从歌单中提取歌曲，返回 ExternalTrack 列表。"""
    try:
        resp = requests.get(
            _NETEASE_PLAYLIST_URL,
            params={"id": playlist_id, "n": limit},
            headers=_HEADERS,
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        logger.debug("Netease playlist detail failed for id=%s", playlist_id, exc_info=True)
        return []

    if not isinstance(data, dict):
        logger.debug("Netease playlist detail returned non-dict payload for id=%s: %r", playlist_id, type(data).__name__)
        return []
    playlist = data.get("playlist")
    if not isinstance(playlist, dict):
        logger.debug("Netease playlist detail returned invalid playlist payload for id=%s: %r", playlist_id, type(playlist).__name__)
        return []
    tracks_raw = playlist.get("tracks") or []
    if not isinstance(tracks_raw, list):
        logger.debug("Netease playlist detail returned invalid tracks payload for id=%s: %r", playlist_id, type(tracks_raw).__name__)
        return []
    result: list[ExternalTrack] = []
    for t in tracks_raw[:limit]:
        song_id = t.get("id")
        if not song_id:
            continue
        name = t.get("name", "")
        artists = "/".join(ar.get("name", "") for ar in t.get("ar", []))
        album_name = ""
        al = t.get("al") or {}
        if al:
            album_name = al.get("name", "")
        cover = al.get("picUrl", "") if al else ""

        result.append(ExternalTrack(
            external_id=str(song_id),
            title=name,
            artist=artists,
            album=album_name or None,
            cover_url=cover or None,
            source="netease",
            playback_url=f"https://music.163.com/song?id={song_id}",
        ))
    return result


def search_and_extract(query: str, max_playlists: int = 3, tracks_per_playlist: int = 15) -> list[ExternalTrack]:
    """一步到位：搜歌单 + 从热门歌单提取歌曲。去重后返回。"""
    playlists = search_netease_playlists(query, limit=max_playlists)
    all_tracks: list[ExternalTrack] = []
    seen: set[str] = set()

    for pl in playlists:
        tracks = get_playlist_tracks(pl["id"], limit=tracks_per_playlist)
        for t in tracks:
            key = f"{t.title.lower()}|{t.artist.lower()}"
            if key not in seen:
                seen.add(key)
                all_tracks.append(t)

    return all_tracks
