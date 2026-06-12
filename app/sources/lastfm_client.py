"""Last.fm API 客户端：用品味档案的 top_artists/top_genres 做音乐发现。

只需要一个免费 API Key（https://www.last.fm/api/account/create），不需要 OAuth。
核心能力：artist.getSimilar → 找同风格艺人 → artist.getTopTracks → 拿代表曲目 → 网易云验证。
"""
from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)

_BASE_URL = "https://ws.audioscrobbler.com/2.0/"
_HEADERS = {"User-Agent": "MusicAgent/1.0"}


class LastfmClient:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    def _get(self, method: str, **params: Any) -> dict:
        """统一 GET 请求 Last.fm API。"""
        params.update({"method": method, "api_key": self.api_key, "format": "json"})
        url = f"{_BASE_URL}?{urllib.parse.urlencode(params)}"
        try:
            req = urllib.request.Request(url, headers=_HEADERS)
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read().decode())
        except Exception:
            logger.debug("Last.fm API call failed: method=%s params=%s", method, params, exc_info=True)
            return {}

    def get_similar_artists(self, artist: str, limit: int = 10) -> list[str]:
        """获取相似艺人名称列表。"""
        data = self._get("artist.getSimilar", artist=artist, limit=limit, autocorrect=1)
        artists_raw = (data.get("similarartists") or {}).get("artist") or []
        if isinstance(artists_raw, dict):  # 单条结果会变成 dict 而非 list
            artists_raw = [artists_raw]
        return [a.get("name", "") for a in artists_raw if a.get("name")]

    def get_artist_top_tracks(self, artist: str, limit: int = 10) -> list[dict[str, str]]:
        """获取艺人热门曲目。返回 [{"title": ..., "artist": ...}, ...]。"""
        data = self._get("artist.getTopTracks", artist=artist, limit=limit, autocorrect=1)
        tracks_raw = (data.get("toptracks") or {}).get("track") or []
        if isinstance(tracks_raw, dict):
            tracks_raw = [tracks_raw]
        result: list[dict[str, str]] = []
        for t in tracks_raw:
            name = t.get("name", "").strip()
            artist_name = t.get("artist", {}).get("name", "").strip() if isinstance(t.get("artist"), dict) else ""
            if name:
                result.append({"title": name, "artist": artist_name or artist})
        return result

    def get_tag_top_tracks(self, tag: str, limit: int = 20) -> list[dict[str, str]]:
        """获取标签/风格的热门曲目。返回 [{"title": ..., "artist": ...}, ...]。"""
        data = self._get("tag.getTopTracks", tag=tag, limit=limit)
        tracks_raw = (data.get("tracks") or {}).get("track") or []
        if isinstance(tracks_raw, dict):
            tracks_raw = [tracks_raw]
        result: list[dict[str, str]] = []
        for t in tracks_raw:
            name = t.get("name", "").strip()
            artist_name = t.get("artist", {}).get("name", "").strip() if isinstance(t.get("artist"), dict) else ""
            if name:
                result.append({"title": name, "artist": artist_name})
        return result

    def get_chart_top_tracks(self, limit: int = 20) -> list[dict[str, str]]:
        """获取全球热门榜单。返回 [{"title": ..., "artist": ...}, ...]。"""
        data = self._get("chart.getTopTracks", limit=limit)
        tracks_raw = (data.get("tracks") or {}).get("track") or []
        if isinstance(tracks_raw, dict):
            tracks_raw = [tracks_raw]
        result: list[dict[str, str]] = []
        for t in tracks_raw:
            name = t.get("name", "").strip()
            artist_name = t.get("artist", {}).get("name", "").strip() if isinstance(t.get("artist"), dict) else ""
            if name:
                result.append({"title": name, "artist": artist_name})
        return result

    def get_similar_tracks(self, artist: str, track: str, limit: int = 10) -> list[dict[str, str]]:
        """获取相似曲目。"""
        data = self._get("track.getSimilar", artist=artist, track=track, limit=limit, autocorrect=1)
        tracks_raw = (data.get("similartracks") or {}).get("track") or []
        if isinstance(tracks_raw, dict):
            tracks_raw = [tracks_raw]
        result: list[dict[str, str]] = []
        for t in tracks_raw:
            name = t.get("name", "").strip()
            artist_name = t.get("artist", {}).get("name", "").strip() if isinstance(t.get("artist"), dict) else ""
            if name:
                result.append({"title": name, "artist": artist_name})
        return result

    def get_artist_info(self, artist: str) -> dict[str, Any]:
        """获取歌手资料：头像、简介摘要、标签。返回 { name, image, bio, tags[] }。"""
        data = self._get("artist.getInfo", artist=artist, autocorrect=1)
        raw = data.get("artist") or {}
        # 头像：取最大尺寸
        images = raw.get("image") or []
        image_url = ""
        if isinstance(images, list) and images:
            image_url = images[-1].get("#text", "") if images else ""
        # 简介
        bio = (raw.get("bio") or {}).get("summary", "") or ""
        # 清理 Last.fm 里的 HTML 标签
        import re
        bio = re.sub(r"<[^>]+>", "", bio).strip()
        # 标签
        tags_raw = (raw.get("tags") or {}).get("tag") or []
        if isinstance(tags_raw, dict):
            tags_raw = [tags_raw]
        tags = [t.get("name", "") for t in tags_raw if t.get("name")]
        return {
            "name": raw.get("name", artist),
            "image": image_url,
            "bio": bio,
            "tags": tags,
        }

    def get_artist_top_albums(self, artist: str, limit: int = 6) -> list[dict[str, str]]:
        """获取歌手代表专辑。返回 [{ name, image, playcount }, ...]。"""
        data = self._get("artist.getTopAlbums", artist=artist, limit=limit, autocorrect=1)
        albums_raw = (data.get("topalbums") or {}).get("album") or []
        if isinstance(albums_raw, dict):
            albums_raw = [albums_raw]
        result: list[dict[str, str]] = []
        for a in albums_raw:
            name = a.get("name", "").strip()
            if not name:
                continue
            images = a.get("image") or []
            image_url = images[-1].get("#text", "") if isinstance(images, list) and images else ""
            result.append({"name": name, "image": image_url})
        return result
