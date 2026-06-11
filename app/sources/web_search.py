"""通用网页搜索：Tavily API（主） + DuckDuckGo Instant Answer（兜底）。

Tavily 专为 AI agent 设计，返回高质量摘要文本，适合直接注入 LLM prompt。
未配置 TAVILY_API_KEY 时自动降级到 DuckDuckGo（纯 urllib，无需 key）。
"""
from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)


def search_web_info(query: str, max_results: int = 5, api_key: str = "") -> list[dict[str, str]]:
    """搜索网页，返回结构化摘要列表。

    Returns list of {"title": ..., "content": ..., "url": ...} dicts.
    """
    if api_key:
        results = _search_tavily(query, max_results, api_key)
        if results:
            return results
        logger.info("Tavily returned no results, falling back to DuckDuckGo")
    return _search_duckduckgo(query, max_results)


def _search_tavily(query: str, max_results: int, api_key: str) -> list[dict[str, str]]:
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
        with urllib.request.urlopen(req, timeout=15) as response:
            data = json.loads(response.read().decode("utf-8"))
        results: list[dict[str, str]] = []
        for item in data.get("results", [])[:max_results]:
            results.append({
                "title": item.get("title", ""),
                "content": item.get("content", ""),
                "url": item.get("url", ""),
            })
        return results
    except Exception:
        logger.debug("Tavily search failed for query=%s", query, exc_info=True)
        return []


def _search_duckduckgo(query: str, max_results: int) -> list[dict[str, str]]:
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
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))
        results: list[dict[str, str]] = []

        # Abstract（直接答案）
        abstract = data.get("Abstract", "")
        if abstract:
            results.append({
                "title": data.get("Heading", ""),
                "content": abstract,
                "url": data.get("AbstractURL", ""),
            })

        # Related topics
        for topic in data.get("RelatedTopics", [])[:max_results]:
            if isinstance(topic, dict) and topic.get("text"):
                results.append({
                    "title": topic.get("Text", "")[:80],
                    "content": topic.get("text", ""),
                    "url": topic.get("FirstURL", ""),
                })
            if len(results) >= max_results:
                break

        return results
    except Exception:
        logger.debug("DuckDuckGo search failed for query=%s", query, exc_info=True)
        return []
