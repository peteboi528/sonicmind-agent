from __future__ import annotations

import json
import logging
import re
import urllib.parse
import urllib.request
from html import unescape
from typing import Any

logger = logging.getLogger(__name__)

_SEARCH_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://www.bilibili.com/",
    "Cookie": "buvid3=infoc;",
}


def extract_bilibili_id(url: str) -> tuple[str, str] | None:
    match = re.search(r"(BV[a-zA-Z0-9]+)", url)
    if match:
        return ("bvid", match.group(1))
    match = re.search(r"av(\d+)", url, re.IGNORECASE)
    if match:
        return ("aid", match.group(1))
    return None


def search_bilibili_video(query: str) -> str | None:
    """Search Bilibili video and return bvid."""
    detail = search_bilibili_detail(query)
    return detail.get("bvid") if detail else None


def fetch_bilibili_title(url: str) -> str | None:
    try:
        req = urllib.request.Request(url, headers=_SEARCH_HEADERS)
        with urllib.request.urlopen(req, timeout=10) as response:
            html = response.read().decode("utf-8", errors="ignore")[:20000]
        match = re.search(r"<title[^>]*>(.+?)</title>", html, re.IGNORECASE | re.DOTALL)
        if match:
            title = match.group(1).strip()
            for suffix in ["_哔哩哔哩_bilibili", " - 哔哩哔哩"]:
                title = title.replace(suffix, "")
            return unescape(title.strip())
    except Exception:
        logger.debug("Bilibili title fetch failed", exc_info=True)
    return None


def search_bilibili_detail(query: str) -> dict[str, Any] | None:
    """Search Bilibili and return real video title/author metadata."""
    results = search_bilibili_many(query, limit=1)
    return results[0] if results else None


def search_bilibili_many(query: str, limit: int = 5) -> list[dict[str, Any]]:
    """Search Bilibili and return multiple video results.

    Returns list of {"bvid", "title", "author", "description"} dicts.
    """
    search_url = (
        "https://api.bilibili.com/x/web-interface/search/type"
        f"?search_type=video&keyword={urllib.parse.quote(query)}"
    )
    try:
        req = urllib.request.Request(search_url, headers=_SEARCH_HEADERS)
        with urllib.request.urlopen(req, timeout=8) as response:
            data = json.loads(response.read().decode())
        return _parse_bilibili_results(data, limit)
    except Exception:
        logger.debug("Bilibili search request failed", exc_info=True)
        return []


async def asearch_bilibili_many(query: str, limit: int = 5) -> list[dict[str, Any]]:
    from app.sources.http_transport import source_transport

    try:
        response = await source_transport.request(
            "bilibili", "GET", "https://api.bilibili.com/x/web-interface/search/type",
            params={"search_type": "video", "keyword": query}, headers=_SEARCH_HEADERS,
            retries=1, concurrency=3,
        )
        return _parse_bilibili_results(response.json(), limit)
    except Exception:
        logger.debug("Async Bilibili search request failed", exc_info=True)
        return []


def _parse_bilibili_results(data: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    if data.get("code") != 0:
        return []
    items: list[dict[str, Any]] = []
    for item in (data.get("data", {}).get("result", []) or [])[:limit]:
        bvid = item.get("bvid")
        title = item.get("title")
        if bvid and title:
            items.append({
                "bvid": bvid,
                "title": unescape(re.sub(r"</?em[^>]*>", "", title).strip()),
                "author": (item.get("author") or "").strip(),
                "description": unescape(re.sub(r"</?em[^>]*>", "", (item.get("description") or "")).strip()),
            })
    return items
