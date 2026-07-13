"""网易云歌单搜索：搜情绪/风格歌单，从中提取高质量真实歌曲。

网易云歌曲搜索对模糊的情绪词效果很差（"慵懒 R&B" 返回随机歌），
但歌单搜索很好——因为歌单是真人策划的，歌曲质量有保障。
"""

from __future__ import annotations

import logging
import math
import time
from datetime import UTC, datetime
from typing import Any

import requests

from app.graph.tag_rules import extract_tags
from app.models import ExternalTrack
from app.sources.mock_source import MockSource
from app.sources.netease import _search_headers, _throttle

logger = logging.getLogger(__name__)

# 歌单搜索端点：旧的 /api/search/get/web 已失效（返非 JSON 串→解析空→歌单路径长期 0 命中，
# 每日推荐在线曲为空的真因）。cloudsearch/pc 对 type=1000 仍稳定返回 result.playlists。
_NETEASE_SEARCH_URL = "https://music.163.com/api/cloudsearch/pc"
_NETEASE_PLAYLIST_URL = "https://music.163.com/api/v6/playlist/detail"
_HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}

# 限流/网络抖动重试。429/5xx/超时是可重试的（与 http_transport 异步路径一致），
# 而 HTTP 200 的合法空结果不重试。退避避免连打加重限流。
_RETRIES = 1
_BACKOFF = 0.3
_TIMEOUT = 10


def _get_with_retry(url: str, params: dict[str, Any], desc: str) -> dict[str, Any] | None:
    """GET + 重试。返回解析后的 JSON dict；网络/限流失败重试后仍败返回 None。

    与旧实现的关键区别：429/5xx/超时不再被静默吞成空——它们会重试，且最终失败时
    升到 WARNING 让限流可见（旧代码全 debug，限流和"真没结果"无法区分）。
    """
    attempts = _RETRIES + 1
    _throttle()  # 全局节流：复用 netese source 的最小间隔，压住突发打爆限流
    for attempt in range(attempts):
        try:
            resp = requests.get(url, params=params, headers=_search_headers(), timeout=_TIMEOUT)
            status = getattr(resp, "status_code", 200)
            if status == 429 or status >= 500:
                # 限流/服务端错误：可重试。
                raise requests.RequestException(f"retryable status {status}")
            resp.raise_for_status()
            data = resp.json()
            return data if isinstance(data, dict) else None
        except Exception:
            if attempt + 1 < attempts:
                time.sleep(_BACKOFF * (attempt + 1))
                continue
            logger.warning("Netease %s 失败（疑似限流），重试 %d 次后放弃：%r", desc, _RETRIES, params, exc_info=True)
            return None
    return None


def search_netease_playlists(query: str, limit: int = 5) -> list[dict[str, Any]]:
    """搜索网易云歌单，返回 [{id, name, track_count}, ...]。"""
    data = _get_with_retry(_NETEASE_SEARCH_URL, {"s": query, "type": 1000, "limit": limit}, "playlist search")
    if not isinstance(data, dict):
        return []
    result = data.get("result")
    if not isinstance(result, dict):
        logger.debug("Netease playlist search returned invalid result payload for %r: %r", query, type(result).__name__)
        return []
    playlists = result.get("playlists") or []
    if not isinstance(playlists, list):
        logger.debug(
            "Netease playlist search returned invalid playlists payload for %r: %r", query, type(playlists).__name__
        )
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
    data = _get_with_retry(_NETEASE_PLAYLIST_URL, {"id": playlist_id, "n": limit}, "playlist detail")
    if not isinstance(data, dict):
        return None
    playlist = data.get("playlist")
    if not isinstance(playlist, dict):
        logger.debug(
            "Netease playlist detail returned invalid playlist payload for id=%s: %r",
            playlist_id,
            type(playlist).__name__,
        )
        return None
    tracks_raw = playlist.get("tracks") or []
    if not isinstance(tracks_raw, list):
        logger.debug(
            "Netease playlist detail returned invalid tracks payload for id=%s: %r",
            playlist_id,
            type(tracks_raw).__name__,
        )
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

        result.append(
            ExternalTrack(
                external_id=str(song_id),
                title=name,
                artist=artists,
                album=album_name or None,
                cover_url=cover or None,
                source="netease",
                playback_url=f"https://music.163.com/song?id={song_id}",
            )
        )
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


def _offline_playlist_fallback(query: str, limit: int) -> list[ExternalTrack]:
    """离线/弱网兜底：用内置曲库给出保守候选，避免歌单链路整段返空。"""
    source = MockSource()
    tracks = source.search(query, limit=limit)
    if not tracks:
        tags = extract_tags(query)
        tracks = source.get_recommendations(tags["genre"], tags["mood"], limit=limit)
    return [track.model_copy(update={"source": "local"}) for track in tracks[:limit]]


def search_and_extract(query: str, max_playlists: int = 3, tracks_per_playlist: int = 15) -> list[ExternalTrack]:
    """一步到位：搜歌单 + 从热门歌单提取歌曲。去重后返回。"""
    # 搜索顺序容易被标题 SEO 操纵。官方编辑歌单优先，其次认证账号和真实播放量，
    # 避免先抽到“跑步/BPM/Type Beat”关键词堆砌歌单。
    playlists = search_netease_playlists(query, limit=max(max_playlists * 3, 8))
    if not playlists:
        return _offline_playlist_fallback(query, max_playlists * tracks_per_playlist)

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

    return all_tracks or _offline_playlist_fallback(query, max_playlists * tracks_per_playlist)
