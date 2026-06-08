"""Agent 工具集定义（OpenAI tools/function calling 格式）。

每个工具对应 ReAct 循环里 LLM 可以主动调用的一个动作。
schema 严格遵守 OpenAI tools 格式，兼容 Qwen / DeepSeek / Claude (via adapter)。
"""

from __future__ import annotations

from typing import Any

# Tool 名称常量（避免散落字符串）
TOOL_RECOMMEND = "recommend_music"
TOOL_SEARCH = "search_music"
TOOL_PLAYLIST = "generate_playlist"
TOOL_TASTE = "summarize_taste"
TOOL_SIMILAR_CROSS = "find_similar_assets"
TOOL_SIMILAR_INTRA = "find_similar_segments"
TOOL_RETRIEVE = "retrieve_evidence"
TOOL_ANALYZE = "analyze_media"
TOOL_REPORT = "generate_report"
TOOL_MEMORY_UPDATE = "update_user_memory"

ALL_TOOL_NAMES = {
    TOOL_RECOMMEND, TOOL_SEARCH, TOOL_PLAYLIST, TOOL_TASTE,
    TOOL_SIMILAR_CROSS, TOOL_SIMILAR_INTRA, TOOL_RETRIEVE,
    TOOL_ANALYZE, TOOL_REPORT, TOOL_MEMORY_UPDATE,
}


def _tool(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


AGENT_TOOLS: list[dict[str, Any]] = [
    _tool(
        TOOL_RECOMMEND,
        "根据用户的品味档案和当前需求推荐音乐。适用于：用户想要新歌推荐、根据心情/时段推荐、每日推荐。",
        {
            "query": {
                "type": "string",
                "description": "用户的推荐请求原文，例如 '推荐适合工作时听的歌'",
            },
            "top_k": {
                "type": "integer",
                "description": "返回数量，默认 5",
                "default": 5,
            },
        },
        ["query"],
    ),
    _tool(
        TOOL_SEARCH,
        "在本地音乐库和外部曲库中搜索匹配的歌曲。适用于：用户找特定歌曲、特定歌手、特定关键词。",
        {
            "query": {
                "type": "string",
                "description": "搜索关键词，可以是歌名、歌手、风格",
            },
            "include_external": {
                "type": "boolean",
                "description": "是否搜索外部曲库",
                "default": True,
            },
        },
        ["query"],
    ),
    _tool(
        TOOL_PLAYLIST,
        "根据用户指令生成一个主题歌单。适用于：用户明确说'做一个歌单'、'帮我整理'等。",
        {
            "instruction": {
                "type": "string",
                "description": "歌单主题或场景，例如 '深夜专注用'、'跑步歌单'",
            },
        },
        ["instruction"],
    ),
    _tool(
        TOOL_TASTE,
        "总结用户的音乐品味档案。适用于：用户想了解自己的偏好、风格画像。",
        {},
        [],
    ),
    _tool(
        TOOL_SIMILAR_CROSS,
        "在媒体库中查找与指定素材相似的其他素材。需要有 asset_id 上下文。",
        {
            "top_k": {"type": "integer", "default": 5},
        },
        [],
    ),
    _tool(
        TOOL_SIMILAR_INTRA,
        "在同一素材中查找与某片段相似的其他片段。需要有 asset_id 上下文。",
        {
            "top_k": {"type": "integer", "default": 5},
        },
        [],
    ),
    _tool(
        TOOL_RETRIEVE,
        "在已分析的素材中检索与 query 相关的片段（RAG 证据）。需要有 asset_id 上下文。",
        {
            "query": {"type": "string", "description": "检索关键词"},
            "top_k": {"type": "integer", "default": 5},
        },
        ["query"],
    ),
    _tool(
        TOOL_ANALYZE,
        "对指定 asset 执行媒体分析，生成 segments。需要有 asset_id 上下文。",
        {},
        [],
    ),
    _tool(
        TOOL_REPORT,
        "生成指定 asset 的综合报告。需要有 asset_id 上下文。",
        {},
        [],
    ),
    _tool(
        TOOL_MEMORY_UPDATE,
        "记录用户的偏好或显式表达的需求。适用于：用户说 '我喜欢X'、'记住Y'。",
        {
            "event": {
                "type": "string",
                "description": "用户原话或抽取的偏好文本",
            },
        },
        ["event"],
    ),
]
