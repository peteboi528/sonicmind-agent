from __future__ import annotations

import random
import uuid
from typing import Any

from app.llm.protocol import LLMResponse, ToolCall
from app.llm.tools import (
    TOOL_ANALYZE,
    TOOL_FETCH_METADATA,
    TOOL_IMPORT_NETEASE_PLAYLIST,
    TOOL_MEMORY_UPDATE,
    TOOL_PLAYLIST,
    TOOL_RECOMMEND,
    TOOL_REPORT,
    TOOL_RETRIEVE,
    TOOL_SEARCH,
    TOOL_SIMILAR_CROSS,
    TOOL_SIMILAR_INTRA,
    TOOL_TASTE,
    TOOL_WEB_MUSIC_SEARCH,
)

REASON_TEMPLATES = [
    "因为你喜欢{genre}风格，这首歌的{mood}氛围很适合{time}听。",
    "根据你最近的收听习惯，这首{genre}曲目的节奏和能量感很匹配你的偏好。",
    "这首歌融合了{mood}的情绪和{genre}的元素，和你的品味非常契合。",
    "你之前喜欢的音乐中有类似的{mood}氛围，推荐你试试这首。",
    "这是一首{genre}风格的佳作，{mood}的基调很适合现在的时段。",
    "基于你的品味档案，这首歌在旋律和情绪上都很对味。",
]

SEARCH_TEMPLATES = [
    "为你找到了{count}首相关内容，其中包含{genre}风格和{mood}氛围的作品。",
    "根据你的搜索，以下是最匹配的{count}首推荐。",
]

DAILY_SUMMARY_TEMPLATES = [
    "今天为你精选了{count}首歌曲，涵盖{genres}等风格，希望能陪伴你度过美好的一天。",
    "根据你的品味和当前时段，今日推荐以{mood}为主基调，共{count}首精选。",
]

CHAT_TEMPLATES = [
    "好的，让我根据你的偏好来推荐。你喜欢{genre}风格，我找到了一些很适合的内容。",
    "明白了！根据你的收听历史，我推荐以下几首{mood}氛围的音乐。",
]


# 工具决策规则（仅用于 MockLLM 第一轮）
_TOOL_RULES: list[tuple[list[str], str]] = [
    (["网易云歌单", "netease playlist", "playlist?id", "导入歌单"], TOOL_IMPORT_NETEASE_PLAYLIST),
    (["联网", "真实", "最新", "网易云", "bilibili", "b站", "线上"], TOOL_WEB_MUSIC_SEARCH),
    (["metadata", "元数据", "标题", "补全信息"], TOOL_FETCH_METADATA),
    (["similar video", "similar asset", "类似视频", "相似视频"], TOOL_SIMILAR_CROSS),
    (["similar segment", "类似片段", "相似片段", "similar moment"], TOOL_SIMILAR_INTRA),
    (["search", "find songs", "搜索", "找歌"], TOOL_SEARCH),
    (["playlist", "歌单", "合集"], TOOL_PLAYLIST),
    (["taste", "品味", "风格分析", "分析我"], TOOL_TASTE),
    (["recommend", "suggest", "推荐", "建议", "适合", "挑", "跑步"], TOOL_RECOMMEND),
    (["analyze", "分析素材", "index"], TOOL_ANALYZE),
    (["report", "摘要", "报告"], TOOL_REPORT),
    (["remember", "preference", "记住", "我喜欢"], TOOL_MEMORY_UPDATE),
    (["what happens", "时间点", "minute mark", "片段"], TOOL_RETRIEVE),
]


class MockLLM:
    def __init__(self) -> None:
        self._tool_call_history: dict[str, list[str]] = {}

    def generate(self, prompt: str, system: str | None = None, temperature: float = 0.7) -> str:
        # 同时检查 system，让意图分类器在 mock 下也能命中（system 含"意图"）
        haystack = f"{system or ''}\n{prompt}".lower()
        if "推荐理由" in prompt or "reason" in haystack:
            return self._reason(prompt)
        if "搜索" in prompt or "search" in haystack:
            return self._search_summary(prompt)
        if "每日" in prompt or "daily" in haystack:
            return self._daily_summary(prompt)
        if "意图" in haystack or "intent" in haystack or "actiontype" in haystack:
            return self._intent(prompt)
        return self._chat(prompt)

    def chat(self, messages: list[dict[str, Any]], temperature: float = 0.7) -> str:
        last_user = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "user"), ""
        )
        return self.generate(last_user)

    def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        temperature: float = 0.3,
        tool_choice: str = "auto",
    ) -> LLMResponse:
        """Mock 实现：第一轮根据关键词选 1 个 tool 调用，第二轮收到工具结果后给最终答案。"""
        # 找到最后一条 user 消息（原始 query）
        last_user = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "user"), ""
        )
        # 检查对话里已经发生的工具调用，避免重复
        called_tools = [
            tc.get("function", {}).get("name", "")
            for m in messages
            if m.get("role") == "assistant"
            for tc in (m.get("tool_calls") or [])
        ]

        tool_names = {t["function"]["name"] for t in tools}
        next_tool = self._pick_tool(last_user, called_tools, tool_names)

        if next_tool is None:
            return LLMResponse(
                content=self._final_answer(last_user, called_tools),
                finish_reason="stop",
            )

        return LLMResponse(
            tool_calls=[
                ToolCall(
                    id=f"call_{uuid.uuid4().hex[:8]}",
                    name=next_tool,
                    arguments=self._mock_args(next_tool, last_user),
                )
            ],
            finish_reason="tool_calls",
        )

    def _pick_tool(self, query: str, already_called: list[str], available: set[str]) -> str | None:
        lowered = query.lower()
        for keywords, tool in _TOOL_RULES:
            if any(kw in lowered for kw in keywords) and tool in available and tool not in already_called:
                return tool
        if not already_called and TOOL_RECOMMEND in available:
            return TOOL_RECOMMEND
        return None

    def _mock_args(self, tool: str, query: str) -> dict[str, Any]:
        if tool in {TOOL_RECOMMEND, TOOL_SEARCH, TOOL_RETRIEVE, TOOL_WEB_MUSIC_SEARCH}:
            return {"query": query, "top_k": 5}
        if tool == TOOL_PLAYLIST:
            return {"instruction": query}
        if tool == TOOL_MEMORY_UPDATE:
            return {"event": query}
        if tool == TOOL_FETCH_METADATA:
            return {"url": query, "use_network": False}
        if tool == TOOL_IMPORT_NETEASE_PLAYLIST:
            return {"playlist_ref": query, "limit": 20}
        return {}

    def _final_answer(self, query: str, called: list[str]) -> str:
        if not called:
            return f"已处理你的请求：{query}"
        if TOOL_RECOMMEND in called:
            return random.choice(CHAT_TEMPLATES).format(genre="流行", mood="轻松")
        if TOOL_IMPORT_NETEASE_PLAYLIST in called or TOOL_WEB_MUSIC_SEARCH in called:
            return f"我已经根据真实输入工具的 observation 整合了结果：{query}"
        return f"已根据 {len(called)} 个工具的结果整合答案：{query}"

    def _intent(self, prompt: str) -> str:
        return '{"actions": ["recommend"], "confidence": 0.8, "reason": "mock intent"}'

    def _reason(self, prompt: str) -> str:
        template = random.choice(REASON_TEMPLATES)
        return template.format(
            genre=_extract_or_default(prompt, "流行"),
            mood=_extract_or_default(prompt, "轻松"),
            time="现在",
        )

    def _search_summary(self, prompt: str) -> str:
        template = random.choice(SEARCH_TEMPLATES)
        return template.format(count=5, genre="流行", mood="欢快")

    def _daily_summary(self, prompt: str) -> str:
        template = random.choice(DAILY_SUMMARY_TEMPLATES)
        return template.format(count=25, genres="流行、电子、民谣", mood="轻松愉快")

    def _chat(self, prompt: str) -> str:
        template = random.choice(CHAT_TEMPLATES)
        return template.format(genre="流行", mood="轻松")


def _extract_or_default(text: str, default: str) -> str:
    keywords = ["流行", "摇滚", "电子", "古典", "爵士", "民谣", "说唱", "R&B"]
    for kw in keywords:
        if kw in text:
            return kw
    return default
