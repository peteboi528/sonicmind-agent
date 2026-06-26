"""通用网页搜索：Tavily API（主） + DuckDuckGo Instant Answer（兜底）。

Tavily 专为 AI agent 设计，返回高质量摘要文本，适合直接注入 LLM prompt。
未配置 TAVILY_API_KEY 时自动降级到 DuckDuckGo（纯 urllib，无需 key）。
"""
from __future__ import annotations

import json
import logging
import re
import urllib.parse
import urllib.request

logger = logging.getLogger(__name__)


def search_web_info(
    query: str,
    max_results: int = 5,
    api_key: str = "",
    timeout: float | None = None,
) -> list[dict[str, str]]:
    """搜索网页，返回结构化摘要列表。

    Returns list of {"title": ..., "content": ..., "url": ...} dicts.
    """
    timeout = max(0.5, float(timeout or 15.0))
    if api_key:
        results = _search_tavily(query, max_results, api_key, timeout=timeout)
        if results:
            return results
        logger.info("Tavily returned no results, falling back to DuckDuckGo")
    return _search_duckduckgo(query, max_results, timeout=timeout)


async def asearch_web_info(query: str, max_results: int = 5, api_key: str = "") -> list[dict[str, str]]:
    if api_key:
        results = await _asearch_tavily(query, max_results, api_key)
        if results:
            return results
    return await _asearch_duckduckgo(query, max_results)


def fetch_url_content(url: str, api_key: str = "", timeout: float | None = None, max_chars: int = 2000) -> str:
    """按 URL 取网页正文（Tavily Extract）。

    与 search_web_info（关键词搜索）互补——后者只返回摘要片段，本函数给定具体乐评
    URL（如 MusicBrainz relations 里的 last.fm/Discogs/Genius 链接）取回正文，供合成
    LLM 写专业乐评。失败/反爬/超时一律返回 ""（与 search_web_info 同样的零异常降级）。
    未配置 TAVILY_API_KEY 时直接返回空——没有可靠的免 key 单 URL 抓取兜底。
    """
    url = (url or "").strip()
    if not url or not api_key:
        return ""
    timeout = max(0.5, float(timeout or 15.0))
    payload = json.dumps({"urls": [url]}).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    try:
        req = urllib.request.Request(
            "https://api.tavily.com/extract", data=payload, headers=headers, method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except Exception:
        logger.debug("Tavily extract failed for url=%s", url, exc_info=True)
        return ""
    results = data.get("results") or []
    if not results:
        return ""
    item = results[0] if isinstance(results[0], dict) else {}
    if item.get("text") is not None:
        # Tavily Extract 成功命中字段为 "text"（新版）；旧版用 raw_content/content。
        text = str(item.get("text") or item.get("raw_content") or item.get("content") or "")
    else:
        text = str(item.get("raw_content") or item.get("content") or "")
    return _clean_extracted_text(text, max_chars)


def _clean_extracted_text(text: str, max_chars: int) -> str:
    """压缩抓回的正文：去多余空白，丢掉常见导航噪音后截断到 max_chars。"""
    text = (text or "").strip()
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0] + "…"
    return text.strip()


async def _asearch_tavily(query: str, max_results: int, api_key: str) -> list[dict[str, str]]:
    from app.sources.http_transport import source_transport

    try:
        response = await source_transport.request(
            "tavily", "POST", "https://api.tavily.com/search",
            json={
                "query": query, "max_results": max_results,
                "search_depth": "basic", "include_answer": False,
            },
            headers={"Authorization": f"Bearer {api_key}"}, retries=0, concurrency=3,
        )
        return _parse_tavily_results(response.json(), max_results)
    except Exception:
        logger.debug("Async Tavily search failed for query=%s", query, exc_info=True)
        return []


async def _asearch_duckduckgo(query: str, max_results: int) -> list[dict[str, str]]:
    from app.sources.http_transport import source_transport

    try:
        response = await source_transport.request(
            "duckduckgo", "GET", "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
            headers={"User-Agent": "Mozilla/5.0"}, retries=1, concurrency=3,
        )
        return _parse_duckduckgo_results(response.json(), max_results)
    except Exception:
        logger.debug("Async DuckDuckGo search failed for query=%s", query, exc_info=True)
        return []


def _search_tavily(query: str, max_results: int, api_key: str, timeout: float = 15.0) -> list[dict[str, str]]:
    """Tavily Search API — 高质量结构化搜索结果。"""
    url = "https://api.tavily.com/search"
    payload = json.dumps({
        "query": query,
        "max_results": max_results,
        "search_depth": "basic",
        "include_answer": False,
    }).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    try:
        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
        return _parse_tavily_results(data, max_results)
    except Exception:
        logger.debug("Tavily search failed for query=%s", query, exc_info=True)
        return []


def _search_duckduckgo(query: str, max_results: int, timeout: float = 10.0) -> list[dict[str, str]]:
    """DuckDuckGo Instant Answer API — 纯 urllib 兜底，无需 key。

    质量不如 Tavily，但零配置可用。
    """
    url = (
        "https://api.duckduckgo.com/"
        f"?q={urllib.parse.quote(query)}&format=json&no_html=1&skip_disambig=1"
    )
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
        return _parse_duckduckgo_results(data, max_results)
    except Exception:
        logger.debug("DuckDuckGo search failed for query=%s", query, exc_info=True)
        return []


def _parse_tavily_results(data: dict, max_results: int) -> list[dict[str, str]]:
    return [
        {"title": item.get("title", ""), "content": item.get("content", ""), "url": item.get("url", "")}
        for item in data.get("results", [])[:max_results]
    ]


def _parse_duckduckgo_results(data: dict, max_results: int) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    abstract = data.get("Abstract", "")
    if abstract:
        results.append({
            "title": data.get("Heading", ""), "content": abstract,
            "url": data.get("AbstractURL", ""),
        })
    for topic in data.get("RelatedTopics", [])[:max_results]:
        if isinstance(topic, dict) and (topic.get("Text") or topic.get("text")):
            text = topic.get("Text") or topic.get("text") or ""
            results.append({"title": text[:80], "content": text, "url": topic.get("FirstURL", "")})
        if len(results) >= max_results:
            break
    return results
