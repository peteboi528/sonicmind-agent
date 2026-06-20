"""网易云歌单搜索：搜情绪/风格歌单，从中提取高质量真实歌曲。

网易云歌曲搜索对模糊的情绪词效果很差（"慵懒 R&B" 返回随机歌），
但歌单搜索很好——因为歌单是真人策划的，歌曲质量有保障。
"""
from __future__ import annotations

import logging
import math
from datetime import UTC, datetime
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
            "play_count": int(pl.get("playCount") or 0),
            "creator_name": str((pl.get("creator") or {}).get("nickname") or ""),
            "creator_verified": bool((pl.get("creator") or {}).get("authStatus")),
        }
        for pl in playlists
    ]


def get_playlist_detail(playlist_id: int, limit: int = 30) -> dict[str, Any] | None:
    """获取歌单元数据与曲目，供榜单调用方校验名称和更新时间。"""
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
        return None

    if not isinstance(data, dict):
        logger.debug("Netease playlist detail returned non-dict payload for id=%s: %r", playlist_id, type(data).__name__)
        return None
    playlist = data.get("playlist")
    if not isinstance(playlist, dict):
        logger.debug("Netease playlist detail returned invalid playlist payload for id=%s: %r", playlist_id, type(playlist).__name__)
        return None
    tracks_raw = playlist.get("tracks") or []
    if not isinstance(tracks_raw, list):
        logger.debug("Netease playlist detail returned invalid tracks payload for id=%s: %r", playlist_id, type(tracks_raw).__name__)
        return None
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
    update_time = playlist.get("updateTime")
    updated_at = None
    if isinstance(update_time, (int, float)) and update_time > 0:
        updated_at = datetime.fromtimestamp(update_time / 1000, tz=UTC).isoformat()
    return {
        "id": str(playlist.get("id") or playlist_id),
        "name": str(playlist.get("name") or ""),
        "updated_at": updated_at,
        "track_count": int(playlist.get("trackCount") or len(result)),
        "tracks": result,
    }


def get_playlist_tracks(playlist_id: int, limit: int = 30) -> list[ExternalTrack]:
    """从歌单中提取歌曲，返回 ExternalTrack 列表。"""
    detail = get_playlist_detail(playlist_id, limit=limit)
    return detail["tracks"] if detail else []


def search_and_extract(query: str, max_playlists: int = 3, tracks_per_playlist: int = 15) -> list[ExternalTrack]:
    """一步到位：搜歌单 + 从热门歌单提取歌曲。去重后返回。"""
    # 搜索顺序容易被标题 SEO 操纵。官方编辑歌单优先，其次认证账号和真实播放量，
    # 避免先抽到“跑步/BPM/Type Beat”关键词堆砌歌单。
    playlists = search_netease_playlists(query, limit=max(max_playlists * 3, 8))

    def trust_score(playlist: dict[str, Any]) -> tuple[int, int, float]:
        creator = playlist.get("creator_name", "").lower()
        official = int("云音乐官方" in creator or "netease cloud music" in creator)
        verified = int(bool(playlist.get("creator_verified")))
        popularity = math.log10(max(int(playlist.get("play_count") or 0), 1))
        return official, verified, popularity

    playlists = sorted(playlists, key=trust_score, reverse=True)[:max_playlists]
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
