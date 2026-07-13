from __future__ import annotations

import json
import logging
import re
import urllib.parse
import urllib.request

logger = logging.getLogger(__name__)


def extract_youtube_id(url: str) -> str | None:
    patterns = [
        r"(?:v=|/embed/|youtu\.be/)([a-zA-Z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def fetch_youtube_title(url: str) -> str | None:
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
    try:
        oembed = f"https://www.youtube.com/oembed?url={urllib.parse.quote(url, safe='')}&format=json"
        req = urllib.request.Request(oembed, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as response:
            return json.loads(response.read().decode()).get("title")
    except Exception:
        logger.debug("YouTube title fetch failed", exc_info=True)
        return None


def search_youtube_video(query: str) -> str | None:
    results = search_youtube_many(query, limit=1)
    return results[0]["video_id"] if results else None


def search_youtube_many(query: str, limit: int = 3) -> list[dict[str, str]]:
    """Search YouTube and return multiple video results.

    Returns list of {"video_id", "title"} dicts.
    """
    search_url = f"https://www.youtube.com/results?search_query={urllib.parse.quote_plus(query)}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8",
    }
    try:
        req = urllib.request.Request(search_url, headers=headers)
        with urllib.request.urlopen(req, timeout=8) as response:
            html = response.read().decode("utf-8")
    except Exception:
        logger.debug("YouTube search request failed", exc_info=True)
        return []

    return _parse_youtube_search_html(html, limit)


async def asearch_youtube_many(query: str, limit: int = 3) -> list[dict[str, str]]:
    from app.sources.http_transport import source_transport

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8",
    }
    try:
        response = await source_transport.request(
            "youtube",
            "GET",
            "https://www.youtube.com/results",
            params={"search_query": query},
            headers=headers,
            retries=1,
            concurrency=3,
        )
        return _parse_youtube_search_html(response.text, limit)
    except Exception:
        logger.debug("Async YouTube search request failed", exc_info=True)
        return []


async def afetch_youtube_title(video_id: str) -> str:
    from app.sources.http_transport import source_transport

    url = f"https://www.youtube.com/watch?v={video_id}"
    try:
        response = await source_transport.request(
            "youtube",
            "GET",
            "https://www.youtube.com/oembed",
            params={"url": url, "format": "json"},
            retries=1,
            concurrency=3,
        )
        return str(response.json().get("title") or "")
    except Exception:
        return ""


def _parse_youtube_search_html(html: str, limit: int) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []

    match = re.search(r"var ytInitialData\s*=\s*(\{.+?\});\s*</script>", html, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1))
            tabs = (
                data.get("contents", {})
                .get("twoColumnSearchResultsRenderer", {})
                .get("primaryContents", {})
                .get("sectionListRenderer", {})
                .get("contents", [])
            )
            for tab in tabs:
                for item in tab.get("itemSectionRenderer", {}).get("contents", []):
                    renderer = item.get("videoRenderer", {})
                    video_id = renderer.get("videoId")
                    title_runs = renderer.get("title", {}).get("runs", [])
                    title = title_runs[0].get("text", "") if title_runs else ""
                    if video_id:
                        items.append({"video_id": video_id, "title": title})
                    if len(items) >= limit:
                        return items
        except (KeyError, IndexError, TypeError, json.JSONDecodeError):
            logger.debug("YouTube initial data parse failed", exc_info=True)

    # Regex fallback
    seen_ids: set[str] = set()
    for vid in re.findall(r'"videoId":"([a-zA-Z0-9_-]{11})"', html):
        if vid not in seen_ids:
            seen_ids.add(vid)
            items.append({"video_id": vid, "title": ""})
            if len(items) >= limit:
                break
    return items
