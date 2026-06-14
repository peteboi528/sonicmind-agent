from __future__ import annotations

import json
import logging
import re
import time
import urllib.parse
import urllib.request
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
_SEARCH_RETRIES = 2
_SEARCH_BACKOFF = 0.5


def _fetch_netease_songs(query: str, limit: int) -> list[dict[str, Any]]:
    """请求网易云搜索接口，返回原始 songs 列表。多端点轮询 + 重试以抵抗间歇性限流。

    依次尝试多个搜索端点；每个端点请求成功且非空即返回。全部端点本轮都空/失败时
    退避后重试，重试次数用尽仍空才返回 []。
    """
    headers = dict(_HEADERS)
    encoded = urllib.parse.quote(query)

    for attempt in range(_SEARCH_RETRIES):
        for base in _SEARCH_ENDPOINTS:
            search_url = f"{base}?s={encoded}&type=1&limit={limit}&offset=0"
            try:
                req = urllib.request.Request(search_url, headers=headers)
                with urllib.request.urlopen(req, timeout=8) as response:
                    data = json.loads(response.read().decode())
                songs = data.get("result", {}).get("songs", []) or []
                if songs:
                    return songs
            except Exception:
                logger.debug("NetEase search failed for %s via %s, attempt %d", query, base, attempt + 1, exc_info=True)
        logger.debug("NetEase search empty (rate-limited?) for %s, attempt %d", query, attempt + 1)
        if attempt < _SEARCH_RETRIES - 1:
            time.sleep(_SEARCH_BACKOFF * (attempt + 1))
    return []


def search_netease(query: str) -> str | None:
    songs = _fetch_netease_songs(query, limit=1)
    if songs:
        return str(songs[0]["id"])
    return None


def search_netease_many(query: str, limit: int = 20) -> list[dict[str, Any]]:
    """搜索网易云，返回多条真实曲目元数据（用于推荐/歌单候选）。

    每项 {"song_id","title","artist","album","cover"}。失败返回空列表。
    """
    songs = _fetch_netease_songs(query, limit=limit)

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
