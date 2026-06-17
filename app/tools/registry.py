from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    args_schema: dict[str, Any]
    required: tuple[str, ...] = ()
    llm_visible: bool = True
    graph_handler: str = ""
    aliases: tuple[str, ...] = ()

    def openai_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.args_schema,
                    "required": list(self.required),
                },
            },
        }


TOOL_REGISTRY: dict[str, ToolSpec] = {
    "recommend": ToolSpec(
        name="recommend",
        aliases=("recommend_music",),
        description="根据用户的品味档案和当前需求推荐音乐。适用于：用户想要新歌推荐、根据心情/时段推荐、每日推荐。默认应先调用 web_music_search 获取真实线上候选，再用本工具排序和解释。",
        args_schema={
            "query": {"type": "string", "description": "用户的推荐请求原文，例如 '推荐适合工作时听的歌'"},
            "top_k": {"type": "integer", "description": "返回数量，默认 5", "default": 5},
        },
        required=("query",),
    ),
    "search": ToolSpec(
        name="search",
        aliases=("search_music",),
        description="搜索本地音乐库，并可补充离线 fallback。适用于：用户找特定歌曲、特定歌手、特定关键词。默认推荐先调用 web_music_search；本工具用于补充本地记忆和库内命中。",
        args_schema={
            "query": {"type": "string", "description": "搜索关键词，可以是歌名、歌手、风格"},
            "include_external": {"type": "boolean", "description": "是否搜索外部曲库", "default": True},
        },
        required=("query",),
    ),
    "artist_albums": ToolSpec(
        name="artist_albums",
        aliases=("recommend_albums",),
        description="获取某歌手的真实专辑清单（网易云专辑搜索，带真实 album_id，可整张播放）。适用于：用户想听/推荐/列举某歌手的专辑、唱片、大碟、discography。不走单曲搜索。",
        args_schema={
            "query": {"type": "string", "description": "歌手名或含歌手名的专辑请求，例如 'The Weeknd 的专辑'"},
        },
        required=("query",),
    ),
    "playlist": ToolSpec(
        name="playlist",
        aliases=("generate_playlist",),
        description="根据用户指令生成一个主题歌单。适用于：用户明确说'做一个歌单'、'帮我整理'等。生成前应尽量已有 web_music_search 或导入结果作为真实候选；若用户给的是网易云歌单链接，应先调用 import_netease_playlist。",
        args_schema={
            "instruction": {"type": "string", "description": "歌单主题或场景，例如 '深夜专注用'、'跑步歌单'"},
            "target_count": {"type": "integer", "description": "用户要求的歌单曲目数，例如 50；未指定时由系统从 instruction 推断"},
        },
        required=("instruction",),
    ),
    "taste": ToolSpec(
        name="taste",
        aliases=("summarize_taste",),
        description="总结用户的音乐品味档案。适用于：用户想了解自己的偏好、风格画像。",
        args_schema={},
    ),
    "taste_experiment": ToolSpec(
        name="taste_experiment",
        aliases=("generate_taste_experiment",),
        description="生成个人音乐品味实验：safe/stretch/bold 三档候选，用播放、跳过、收藏、不喜欢等反馈验证用户探索边界。",
        args_schema={
            "prompt": {"type": "string", "description": "用户的探索请求，例如 '推荐点不一样的'"},
            "total": {"type": "integer", "description": "实验总曲目数，默认 12", "default": 12},
        },
        required=("prompt",),
    ),
    "similar_cross": ToolSpec(
        name="similar_cross",
        aliases=("find_similar_assets",),
        description="在媒体库中查找与指定素材相似的其他素材。需要有 asset_id 上下文。",
        args_schema={"top_k": {"type": "integer", "default": 5}},
    ),
    "similar_intra": ToolSpec(
        name="similar_intra",
        aliases=("find_similar_segments",),
        description="在同一素材中查找与某片段相似的其他片段。需要有 asset_id 上下文。",
        args_schema={"top_k": {"type": "integer", "default": 5}},
    ),
    "retrieve": ToolSpec(
        name="retrieve",
        aliases=("retrieve_evidence",),
        description="在已分析的素材中检索与 query 相关的片段（RAG 证据）。需要有 asset_id 上下文。",
        args_schema={
            "query": {"type": "string", "description": "检索关键词"},
            "top_k": {"type": "integer", "default": 5},
        },
        required=("query",),
    ),
    "analyze": ToolSpec(
        name="analyze",
        aliases=("analyze_media",),
        description="对指定 asset 执行媒体分析，生成 segments。需要有 asset_id 上下文。",
        args_schema={},
    ),
    "report": ToolSpec(
        name="report",
        aliases=("generate_report",),
        description="生成指定 asset 的综合报告。需要有 asset_id 上下文。",
        args_schema={},
    ),
    "memory_update": ToolSpec(
        name="memory_update",
        aliases=("update_user_memory",),
        description="记录用户的偏好或显式表达的需求。适用于：用户说 '我喜欢X'、'记住Y'。",
        args_schema={"event": {"type": "string", "description": "用户原话或抽取的偏好文本"}},
        required=("event",),
    ),
    "web_music_search": ToolSpec(
        name="web_music_search",
        aliases=("search_web_music",),
        description="联网搜索真实音乐或视频候选。推荐、搜索、歌单任务默认优先调用本工具。结果不可预知，调用后应评估数量和质量；不足时换关键词继续搜索，足够时再推荐或生成歌单。",
        args_schema={
            "query": {"type": "string", "description": "搜索关键词，可以包含歌曲、歌手、场景或平台"},
            "top_k": {"type": "integer", "description": "返回数量，默认 5", "default": 5},
        },
        required=("query",),
    ),
    "fetch_metadata": ToolSpec(
        name="fetch_metadata",
        aliases=("fetch_track_metadata",),
        description="抓取真实曲目或素材元数据。适用于：已有 asset_id 或 URL，但标题、歌手、专辑、封面等信息不完整。",
        args_schema={
            "asset_id": {"type": "string", "description": "已有素材 id，可选"},
            "url": {"type": "string", "description": "待抓取的音乐或视频 URL，可选"},
            "use_network": {"type": "boolean", "description": "是否允许联网抓取", "default": True},
        },
    ),
    "import_netease_playlist": ToolSpec(
        name="import_netease_playlist",
        aliases=("import",),
        description="导入网易云歌单。适用于：用户给出网易云歌单链接或 id，希望把歌单作为后续推荐、筛选、建歌单的真实输入。",
        args_schema={
            "playlist_ref": {"type": "string", "description": "网易云歌单链接或歌单 id"},
            "limit": {"type": "integer", "description": "最多导入数量，默认 100", "default": 100},
        },
        required=("playlist_ref",),
    ),
    "journey": ToolSpec(
        name="journey",
        description="为用户生成多阶段音乐旅程。",
        args_schema={},
        llm_visible=False,
    ),
    "video_search": ToolSpec(
        name="video_search",
        description="搜索 B 站和 YouTube 上的 MV、现场和演唱会视频。",
        args_schema={},
        llm_visible=False,
    ),
    "web_info_search": ToolSpec(
        name="web_info_search",
        description="用搜索引擎查找歌手、乐队和作品的背景资料。",
        args_schema={},
        llm_visible=False,
    ),
}


_ALIAS_TO_CANONICAL: dict[str, str] = {}
for _canonical, _spec in TOOL_REGISTRY.items():
    _ALIAS_TO_CANONICAL[_canonical] = _canonical
    for _alias in _spec.aliases:
        _ALIAS_TO_CANONICAL[_alias] = _canonical


ALL_TOOL_NAMES = set(_ALIAS_TO_CANONICAL)


def normalize_tool_name(name: str) -> str | None:
    return _ALIAS_TO_CANONICAL.get(name)


def get_handler(name: str) -> str | None:
    canonical = normalize_tool_name(name)
    if canonical is None:
        return None
    spec = TOOL_REGISTRY[canonical]
    return spec.graph_handler or spec.name


def to_openai_tools() -> list[dict[str, Any]]:
    return [spec.openai_tool() for spec in TOOL_REGISTRY.values() if spec.llm_visible]
