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
        return None

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
                    video_id = item.get("videoRenderer", {}).get("videoId")
                    if video_id:
                        return video_id
        except (KeyError, IndexError, TypeError, json.JSONDecodeError):
            logger.debug("YouTube initial data parse failed", exc_info=True)

    ids = re.findall(r'"videoId":"([a-zA-Z0-9_-]{11})"', html)
    return ids[0] if ids else None
