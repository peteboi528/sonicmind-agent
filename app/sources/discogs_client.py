"""Discogs 客户端：权威发行(year/genres/styles/tracklist)。

需 Personal Access Token(https://www.discogs.com/settings/developers)。
Discogs 的独特价值是细粒度 styles(比 genres 更准)和权威发行年份/版本。
token 缺失或调用失败时返回空，知识链路降级。
"""
from __future__ import annotations

import json
import logging
import re
import urllib.parse
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.discogs.com"
_HEADERS = {"User-Agent": "MusicAgent/1.0"}
_TIMEOUT = 6


def _norm(value: str) -> str:
    return re.sub(r"[^a-z0-9一-鿿]+", "", (value or "").lower())


def _best_discogs_match(results: list[dict], title: str, artist: str) -> dict:
    """从 Discogs 候选里挑最匹配的：title 形如 "Artist - Album"。
    优先同时含查询标题与 artist 的候选；其次只含标题的；都没有才回落 results[0]。"""
    nt, na = _norm(title), _norm(artist)
    both = [r for r in results if nt and nt in _norm(r.get("title", "")) and (not na or na in _norm(r.get("title", "")))]
    if both:
        return both[0]
    title_only = [r for r in results if nt and nt in _norm(r.get("title", ""))]
    if title_only:
        return title_only[0]
    return results[0]


class DiscogsClient:
    def __init__(self, token: str = "") -> None:
        self.token = token

    @property
    def available(self) -> bool:
        return bool(self.token)

    def _get(self, path: str, **params: Any) -> dict:
        if not self.available:
            return {}
        query = {"token": self.token, **params}
        url = f"{_BASE_URL}/{path.lstrip('/')}?{urllib.parse.urlencode(query)}"
        try:
            req = urllib.request.Request(url, headers=_HEADERS)
            with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
                return json.loads(resp.read().decode())
        except Exception:
            logger.debug("Discogs API call failed: path=%s params=%s", path, params, exc_info=True)
            return {}

    def search(self, query: str, rtype: str = "master", limit: int = 1) -> list[dict]:
        """database/search。rtype: master/release/artist。"""
        data = self._get("database/search", q=query, type=rtype, limit=limit)
        return data.get("results") or []

    def resolve_release(self, title: str, artist: str = "") -> dict | None:
        """专辑发行消歧：优先 master（规范版），回落 release。返回 {id,title,year,genres,styles}。

        Discogs 的 title 形如 "Frank Ocean - Blonde"。裸标题查询时 results[0] 常是模糊匹配
        的错专辑，故多取几条候选，优先选 title 里同时包含查询标题与 artist 的那条。
        """
        q = f"{title} {artist}".strip() if artist else title
        results = self.search(q, rtype="master", limit=5) or self.search(q, rtype="release", limit=5)
        if not results:
            return None
        r = _best_discogs_match(results, title, artist)
        return {
            "id": str(r.get("id", "")),
            "title": r.get("title", ""),
            "year": r.get("year", 0),
            "genres": r.get("genre") or [],
            "styles": r.get("style") or [],
            "type": r.get("type", ""),
        }

    def resolve_artist(self, name: str) -> dict | None:
        results = self.search(name, rtype="artist", limit=1)
        if not results:
            return None
        r = results[0]
        return {
            "id": str(r.get("id", "")),
            "name": r.get("title", ""),
            "genres": r.get("genre") or [],
            "styles": r.get("style") or [],
        }
