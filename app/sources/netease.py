from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
import time
import urllib.parse
import urllib.request
from collections import OrderedDict
from functools import lru_cache
from typing import Any

from app.netease_auth import _cookie_header

logger = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
    "Referer": "https://music.163.com/",
}

# 网易云搜索接口会间歇性限流：同一查询有时返回结果、有时返回空 songs（HTTP 200
# 但 result.songs 为 []）。一次空就放弃会导致大量真实查询掉到 mock fallback，
# 用户看到一堆 netease-fallback 假候选（播不了）。这里多端点轮询 + 带退避重试，
# 把限流抖动滤掉（实测 search/get、cloudsearch/pc 比旧的 search/get/web 稳得多）。
_SEARCH_ENDPOINTS = (
    "https://music.163.com/api/search/get",
    "https://music.163.com/api/cloudsearch/pc",
    "https://music.163.com/api/search/get/web",
)
_ALBUM_SEARCH_ENDPOINTS = (
    "https://music.163.com/api/search/get",
    "https://music.163.com/api/cloudsearch/pc",
)
# _SEARCH_RETRIES 语义对齐 http_transport：表示"重试次数"，实际尝试 = retries + 1。
# 异步路径 source_transport.request 即用 attempts = retries + 1；同步路径下方也照此展开，
# 否则 1 在 range(1) 里只跑一次、退避 sleep 永不触发（旧 bug）。
_SEARCH_RETRIES = 1
_SEARCH_BACKOFF = 0.2
_SEARCH_TIMEOUT = 3.0

# 专辑详情进程内缓存：按 album_id 缓存完整曲目，重复点击同一专辑不再重复打网易云。
# FIFO + 上限，避免长跑实例无限增长；FIFO 而非 LRU 是因为专辑曲目稳定、不需要按热度复用。
_ALBUM_DETAIL_CACHE: OrderedDict[str, dict[str, Any]] = OrderedDict()
_ALBUM_DETAIL_CACHE_LOCK = threading.Lock()
_ALBUM_DETAIL_CACHE_MAX = 64


def clear_album_detail_cache() -> int:
    """清空专辑详情缓存，返回清掉的条目数（供 /cache 主动刷新用）。"""
    with _ALBUM_DETAIL_CACHE_LOCK:
        n = len(_ALBUM_DETAIL_CACHE)
        _ALBUM_DETAIL_CACHE.clear()
        return n


def _fetch_netease_songs(query: str, limit: int, offset: int = 0) -> list[dict[str, Any]]:
    """请求网易云搜索接口，返回原始 songs 列表。多端点轮询 + 重试以抵抗间歇性限流。

    依次尝试多个搜索端点；每个端点请求成功且非空即返回。全部端点本轮都空/失败时
    退避后重试，重试次数用尽仍空才返回 []。

    offset 支持翻页：延续指令"不要重复/再来几首"时，调用层传 offset=已展示数，
    跳过已经给用户看过的那批最热结果，拿更深位次的新歌——否则同一查询永远返回
    同一批 top-N，去重后很快就无新歌可给。
    """
    headers = dict(_HEADERS)
    encoded = urllib.parse.quote(query)
    offset = max(0, int(offset or 0))

    from app.concurrency import run_parallel

    def fetch(base: str) -> list[dict[str, Any]]:
        search_url = f"{base}?s={encoded}&type=1&limit={limit}&offset={offset}"
        try:
            req = urllib.request.Request(search_url, headers=headers)
            with urllib.request.urlopen(req, timeout=_SEARCH_TIMEOUT) as response:
                data = json.loads(response.read().decode())
            return data.get("result", {}).get("songs", []) or []
        except Exception:
            logger.debug("NetEase search failed for %s via %s", query, base, exc_info=True)
            return []

    # attempts = 重试次数 + 1（与 http_transport 异步路径一致）。每轮多端点并行，
    # 任一端点非空即返回；全空（疑似限流）则退避后重试，次数用尽仍空才返回 []。
    attempts = _SEARCH_RETRIES + 1
    for attempt in range(attempts):
        batches = run_parallel(
            [(f"netease:{base}", lambda base=base: fetch(base)) for base in _SEARCH_ENDPOINTS],
            timeout=_SEARCH_TIMEOUT + 0.2,
            default=[],
        )
        for songs in batches:
            if songs:
                return songs
        logger.debug("NetEase search empty (rate-limited?) for %s, attempt %d/%d", query, attempt + 1, attempts)
        if attempt < attempts - 1:
            time.sleep(_SEARCH_BACKOFF * (attempt + 1))
    return []


def search_netease(query: str) -> str | None:
    songs = _fetch_netease_songs(query, limit=1)
    if songs:
        return str(songs[0]["id"])
    return None


@lru_cache(maxsize=128)
def fetch_netease_lyrics(song_id: str) -> list[str]:
    """Fetch verified NetEase lyrics and strip timestamp metadata.

    Empty/failed responses return ``[]``.  The function never synthesizes text.
    """
    song_id = str(song_id or "").strip()
    if not song_id.isdigit():
        return []
    url = f"https://music.163.com/api/song/lyric?id={song_id}&lv=-1&kv=-1&tv=-1"
    try:
        req = urllib.request.Request(url, headers=_HEADERS)
        with urllib.request.urlopen(req, timeout=8) as response:
            data = json.loads(response.read().decode())
    except Exception:
        logger.debug("NetEase lyric fetch failed for %s", song_id, exc_info=True)
        return []
    lyric = ((data.get("lrc") or {}).get("lyric") or "").strip()
    if not lyric:
        return []
    lines: list[str] = []
    for raw_line in lyric.splitlines():
        text = re.sub(r"^(?:\[[^\]]+\])+", "", raw_line).strip()
        if text and not re.match(r"^(作词|作曲|编曲|制作人|混音)[:：]", text):
            lines.append(text)
    return lines[:500]


def search_netease_artist_image(artist: str) -> str | None:
    """搜网易云歌手（type=100）取头像 url（picUrl，回退 img1v1Url）。

    比 Last.fm 的 image 字段可靠得多——Last.fm artist.getInfo 的最大尺寸 #text
    对大量歌手为空，导致歌手卡只显示占位 emoji。网易云歌手页几乎都有 picUrl。
    多端点轮询，首个非空即返回。
    """
    headers = dict(_HEADERS)
    encoded = urllib.parse.quote(artist)
    for base in _SEARCH_ENDPOINTS:
        try:
            search_url = f"{base}?s={encoded}&type=100&limit=1"
            req = urllib.request.Request(search_url, headers=headers)
            with urllib.request.urlopen(req, timeout=8) as response:
                data = json.loads(response.read().decode())
            artists = data.get("result", {}).get("artists", []) or []
            if artists:
                a = artists[0]
                url = a.get("picUrl") or a.get("img1v1Url")
                if url:
                    return url
        except Exception:
            logger.debug("NetEase artist image search failed for %s", artist, exc_info=True)
    return None


def search_netease_album(artist: str, album: str) -> dict[str, Any] | None:
    """Search a NetEase album by artist + album name and return album metadata."""
    artist = (artist or "").strip()
    album = (album or "").strip()
    if not album:
        return None
    query = " ".join(part for part in [artist, album] if part)
    albums = _fetch_netease_albums(query, limit=8)
    if not albums and artist:
        albums = _fetch_netease_albums(album, limit=8)
    if not albums:
        return None

    wanted_album = _normalize_music_name(album)
    wanted_artist = _normalize_music_name(artist)

    def score(item: dict[str, Any]) -> tuple[int, int]:
        item_name = _normalize_music_name(item.get("name", ""))
        item_artist = _normalize_music_name(_album_artist_name(item))
        name_score = 3 if item_name == wanted_album else 2 if wanted_album in item_name or item_name in wanted_album else 0
        artist_score = 2 if wanted_artist and item_artist == wanted_artist else 1 if wanted_artist and wanted_artist in item_artist else 0
        return name_score + artist_score, int(item.get("size") or 0)

    best = max(albums, key=score)
    if score(best)[0] <= 0:
        return None
    return {
        "id": str(best.get("id") or best.get("idStr") or ""),
        "name": (best.get("name") or album).strip(),
        "artist": _album_artist_name(best) or artist,
        "cover": best.get("picUrl") or best.get("blurPicUrl") or "",
        "track_count": int(best.get("size") or 0) or None,
        "raw": best,
    }


def fetch_netease_album_tracks(album_id: str, limit: int = 100) -> dict[str, Any] | None:
    """Fetch an album detail from NetEase, preserving the original album track order.

    结果按 album_id 进程内缓存（FIFO，上限 64）：重复点击同一专辑不再重复打网易云，
    只在首次访问时网络取一次。缓存的是完整曲目，limit 仅在返回时裁剪，故不同 limit
    的调用互不影响。返回的是缓存对象的浅拷贝（tracks 已按 limit 裁剪），调用方改不动缓存。
    """
    album_id = str(album_id or "").strip()
    if not album_id:
        return None
    n = max(0, int(limit or 0))

    with _ALBUM_DETAIL_CACHE_LOCK:
        cached = _ALBUM_DETAIL_CACHE.get(album_id)
        if cached is not None:
            _ALBUM_DETAIL_CACHE.move_to_end(album_id)  # 命中即续命（保持 FIFO 顺序）
    if cached is not None:
        tracks = cached["tracks"][:n] if n > 0 else list(cached["tracks"])
        return {**cached, "tracks": tracks}

    # 缓存未命中：网络取完整专辑详情（不裁剪，完整曲目入缓存）
    try:
        api = f"https://music.163.com/api/v1/album/{album_id}"
        with urllib.request.urlopen(urllib.request.Request(api, headers=_HEADERS), timeout=8) as response:
            data = json.loads(response.read().decode())
    except Exception:
        logger.debug("NetEase album detail fetch failed: %s", album_id, exc_info=True)
        return None
    if not isinstance(data, dict) or data.get("code") not in (None, 200):
        return None
    album = data.get("album") or {}
    songs = data.get("songs") or []
    if not isinstance(songs, list):
        songs = []
    album_name = (album.get("name") or "").strip()
    album_artist = _album_artist_name(album)
    cover = album.get("picUrl") or album.get("blurPicUrl") or ""
    tracks: list[dict[str, Any]] = []
    for song in songs:
        song_id = song.get("id")
        title = (song.get("name") or "").strip()
        if not song_id or not title:
            continue
        al = song.get("al") or song.get("album") or {}
        tracks.append({
            "song_id": str(song_id),
            "title": title,
            "artist": _song_artists(song) or album_artist,
            "album": (al.get("name") or album_name or "").strip() or None,
            "cover": al.get("picUrl") or cover or None,
        })
    full = {
        "id": album_id,
        "name": album_name,
        "artist": album_artist,
        "cover": cover,
        "track_count": int(album.get("size") or len(songs) or len(tracks)) or None,
        "tracks": tracks,
    }
    with _ALBUM_DETAIL_CACHE_LOCK:
        _ALBUM_DETAIL_CACHE[album_id] = full
        _ALBUM_DETAIL_CACHE.move_to_end(album_id)
        while len(_ALBUM_DETAIL_CACHE) > _ALBUM_DETAIL_CACHE_MAX:
            _ALBUM_DETAIL_CACHE.popitem(last=False)
    return {**full, "tracks": full["tracks"][:n] if n > 0 else list(full["tracks"])}


def search_netease_artist_albums(artist: str, limit: int = 6) -> list[dict[str, Any]]:
    """搜某歌手的专辑，返回带真实 album_id 的专辑元数据列表（供歌手页代表专辑）。

    Last.fm 的 artist.getTopAlbums 只给 name/image、没有 id，点进去还得二次
    search_netease_album 按名字猜匹配，容易猜错（同名合集/精选混入）。这里直接
    拿网易云专辑搜索结果，album_id 真实可信，点击直达专辑详情，省掉二次猜测。

    返回 [{id, name, image, artist, track_count}]，本歌手专辑排前（按歌手名匹配
    排序），按归一化名称去重，最多 limit 条。失败/无结果返回 []。
    """
    artist = (artist or "").strip()
    if not artist:
        return []
    # 多取一些再排序去重，确保过滤掉同名合集后仍能凑够 limit。
    albums = _fetch_netease_albums(artist, limit=max(limit * 3, 12))
    if not albums:
        return []
    return _normalize_artist_albums(albums, artist, limit)


async def asearch_netease_artist_albums(artist: str, limit: int = 6) -> list[dict[str, Any]]:
    artist = (artist or "").strip()
    if not artist:
        return []
    from app.sources.http_transport import source_transport

    params = {"s": artist, "type": 10, "limit": max(limit * 3, 12)}

    async def fetch(endpoint: str) -> list[dict[str, Any]]:
        try:
            response = await source_transport.request(
                "netease", "GET", endpoint, params=params, headers=_HEADERS,
                retries=1, concurrency=4,
            )
            return response.json().get("result", {}).get("albums", []) or []
        except asyncio.CancelledError:
            raise
        except Exception:
            return []

    batches = await asyncio.gather(*(fetch(endpoint) for endpoint in _ALBUM_SEARCH_ENDPOINTS))
    albums = next((batch for batch in batches if batch), [])
    return _normalize_artist_albums(albums, artist, limit)


def _normalize_artist_albums(
    albums: list[dict[str, Any]], artist: str, limit: int,
) -> list[dict[str, Any]]:

    wanted_artist = _normalize_music_name(artist)

    def artist_rank(item: dict[str, Any]) -> int:
        item_artist = _normalize_music_name(_album_artist_name(item))
        if not wanted_artist or not item_artist:
            return 1
        if item_artist == wanted_artist:
            return 2
        if wanted_artist in item_artist or item_artist in wanted_artist:
            return 1
        return 0

    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in sorted(albums, key=artist_rank, reverse=True):
        name = (item.get("name") or "").strip()
        album_id = str(item.get("id") or item.get("idStr") or "")
        if not name or not album_id:
            continue
        key = _normalize_music_name(name)
        if key in seen:
            continue
        seen.add(key)
        result.append({
            "id": album_id,
            "name": name,
            "image": item.get("picUrl") or item.get("blurPicUrl") or "",
            "artist": _album_artist_name(item) or artist,
            "track_count": int(item.get("size") or 0) or None,
        })
        if len(result) >= limit:
            break
    return result


def _fetch_netease_albums(query: str, limit: int = 8) -> list[dict[str, Any]]:
    headers = dict(_HEADERS)
    encoded = urllib.parse.quote(query)
    for base in _ALBUM_SEARCH_ENDPOINTS:
        try:
            search_url = f"{base}?s={encoded}&type=10&limit={limit}"
            req = urllib.request.Request(search_url, headers=headers)
            with urllib.request.urlopen(req, timeout=8) as response:
                data = json.loads(response.read().decode())
            albums = data.get("result", {}).get("albums", []) or []
            if isinstance(albums, list) and albums:
                return albums
        except Exception:
            logger.debug("NetEase album search failed for %s via %s", query, base, exc_info=True)
    return []


def search_netease_many(query: str, limit: int = 20, offset: int = 0) -> list[dict[str, Any]]:
    """搜索网易云，返回多条真实曲目元数据（用于推荐/歌单候选）。

    每项 {"song_id","title","artist","album","cover"}。失败返回空列表。
    offset 用于延续指令翻页取新歌（见 _fetch_netease_songs）。
    """
    songs = _fetch_netease_songs(query, limit=limit, offset=offset)

    return _normalize_netease_songs(songs)


async def asearch_netease_many(query: str, limit: int = 20, offset: int = 0) -> list[dict[str, Any]]:
    """Async pooled NetEase search with cancellation, source limits, retries and circuit breaking."""
    from app.sources.http_transport import source_transport

    params = {"s": query, "type": 1, "limit": limit, "offset": max(0, int(offset or 0))}

    async def fetch(endpoint: str) -> list[dict[str, Any]]:
        try:
            response = await source_transport.request(
                "netease", "GET", endpoint, params=params, headers=_HEADERS,
                retries=_SEARCH_RETRIES, concurrency=4,
            )
            payload = response.json()
            return payload.get("result", {}).get("songs", []) or []
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.debug("Async NetEase search failed for %s via %s", query, endpoint, exc_info=True)
            return []

    batches = await asyncio.gather(*(fetch(endpoint) for endpoint in _SEARCH_ENDPOINTS))
    songs = next((batch for batch in batches if batch), [])
    return _normalize_netease_songs(songs)


def _normalize_netease_songs(songs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for song in songs:
        name = (song.get("name") or "").strip()
        if not name:
            continue
        artists = "、".join(
            a.get("name", "").strip()
            for a in (song.get("artists") or song.get("ar") or [])
            if a.get("name")
        )
        album = song.get("album") or song.get("al") or {}
        results.append({
            "song_id": str(song.get("id")),
            "title": name,
            "artist": artists,
            "album": (album.get("name") or "").strip() or None,
            "cover": album.get("picUrl"),
        })
    return results


def _normalize_music_name(value: str) -> str:
    return re.sub(r"[\s\-_:：·•.'\"“”‘’()（）\[\]【】]+", "", (value or "").lower())


def _album_artist_name(album: dict[str, Any]) -> str:
    artist = album.get("artist")
    if isinstance(artist, dict) and artist.get("name"):
        return str(artist.get("name", "")).strip()
    artists = album.get("artists")
    if isinstance(artists, list):
        names = [str(a.get("name", "")).strip() for a in artists if isinstance(a, dict) and a.get("name")]
        return "、".join(names)
    return ""


def _song_artists(song: dict[str, Any]) -> str:
    raw = song.get("ar") or song.get("artists") or []
    if not isinstance(raw, list):
        return ""
    return "、".join(str(a.get("name", "")).strip() for a in raw if isinstance(a, dict) and a.get("name"))


def search_netease_detail(query: str) -> dict[str, Any] | None:
    """Search NetEase and return verified track metadata."""
    song_id = search_netease(query)
    if not song_id:
        return None
    return fetch_netease_song_detail(song_id)


def fetch_netease_title(url: str, song_id: str | None = None) -> str | None:
    if song_id:
        detail = fetch_netease_song_detail(song_id)
        if detail:
            title = detail.get("title") or ""
            artist = detail.get("artist") or ""
            return f"{title} - {artist}" if artist else title

    page_url = f"https://music.163.com/song?id={song_id}" if song_id else url
    try:
        req = urllib.request.Request(page_url, headers=_HEADERS)
        with urllib.request.urlopen(req, timeout=10) as response:
            html = response.read().decode("utf-8", errors="ignore")[:20000]
        match = re.search(r"<title[^>]*>(.+?)</title>", html, re.IGNORECASE | re.DOTALL)
        if match:
            title = match.group(1).strip()
            for suffix in [" - 单曲 - 网易云音乐", " - 网易云音乐"]:
                title = title.replace(suffix, "")
            return title.strip()
    except Exception:
        logger.debug("NetEase title fetch failed", exc_info=True)
    return None


def fetch_netease_song_detail(song_id: str) -> dict[str, Any] | None:
    try:
        api = f"https://music.163.com/api/song/detail/?id={song_id}&ids=[{song_id}]"
        with urllib.request.urlopen(urllib.request.Request(api, headers=_HEADERS), timeout=8) as response:
            data = json.loads(response.read().decode())
        songs = data.get("songs") or []
        if not songs:
            return None
        song = songs[0]
        name = (song.get("name") or "").strip()
        if not name:
            return None
        artists = "、".join(
            artist.get("name", "").strip()
            for artist in (song.get("artists") or song.get("ar") or [])
            if artist.get("name")
        )
        album = song.get("album") or song.get("al") or {}
        return {
            "song_id": song_id,
            "title": name,
            "artist": artists,
            "album": (album.get("name") or "").strip() or None,
            "cover": album.get("picUrl"),
            "raw": song,
        }
    except Exception:
        logger.debug("NetEase song detail fetch failed: %s", song_id, exc_info=True)
        return None


def get_netease_audio_url(song_id: str, cookie: str = "") -> str | None:
    headers = dict(_HEADERS)
    cookie_header = _cookie_header(cookie) if cookie else ""
    if cookie_header:
        headers["Cookie"] = cookie_header

    apis = []
    if cookie_header:
        apis.append(
            "https://music.163.com/api/song/enhance/player/url/v1"
            f"?ids=[{song_id}]&level=exhigh&encodeType=aac"
        )
    apis.append(
        "https://music.163.com/api/song/enhance/player/url"
        f"?id={song_id}&ids=[{song_id}]&br=320000"
    )

    for api in apis:
        try:
            req = urllib.request.Request(api, headers=headers)
            with urllib.request.urlopen(req, timeout=8) as response:
                data = json.loads(response.read().decode())
            items = data.get("data", [])
            if items and items[0].get("url"):
                return items[0]["url"].replace("http://", "https://", 1)
        except Exception:
            logger.debug("NetEase audio URL fetch failed: %s", song_id, exc_info=True)
    return None
