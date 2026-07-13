from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model

from app.tools.contracts import ToolContext, ToolResult, ToolRisk

ToolHandler = Callable[[dict[str, Any], ToolContext], ToolResult | Awaitable[ToolResult]]


def _python_type(schema: dict[str, Any]) -> Any:
    kind = schema.get("type")
    if "enum" in schema:
        return Literal.__getitem__(tuple(schema["enum"]))
    if kind == "integer":
        return int
    if kind == "number":
        return float
    if kind == "boolean":
        return bool
    if kind == "array":
        return list[_python_type(schema.get("items", {}))]
    if kind == "object":
        return dict[str, Any]
    return str


def _args_model(name: str, schema: dict[str, Any], required: tuple[str, ...]) -> type[BaseModel]:
    fields: dict[str, tuple[Any, Any]] = {}
    for key, item in schema.items():
        annotation = _python_type(item)
        default = ... if key in required else item.get("default", None)
        fields[key] = (annotation, Field(default=default, description=item.get("description")))
    model = create_model(f"{''.join(part.title() for part in name.split('_'))}Args", **fields)
    model.model_config = ConfigDict(extra="forbid")
    return model


@dataclass
class ToolSpec:
    name: str
    description: str
    args_schema: dict[str, Any]
    required: tuple[str, ...] = ()
    llm_visible: bool = True
    graph_handler: str = ""
    aliases: tuple[str, ...] = ()
    risk: ToolRisk = ToolRisk.READ
    timeout_seconds: float = 8.0
    max_retries: int = 1
    max_concurrency: int = 20
    idempotent: bool = True
    source: str = "internal"
    handler: ToolHandler | None = None
    async_handler: ToolHandler | None = None
    args_model: type[BaseModel] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.args_model = _args_model(self.name, self.args_schema, self.required)

    def openai_tool(self) -> dict[str, Any]:
        parameters = self.args_model.model_json_schema()
        parameters.pop("title", None)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": parameters,
            },
        }


TOOL_REGISTRY: dict[str, ToolSpec] = {
    "recommend": ToolSpec(
        name="recommend",
        aliases=("recommend_music",),
        description="根据用户的品味档案和当前需求推荐音乐。适用于：用户想要新歌推荐、根据心情/时段推荐、每日推荐。默认应先调用 web_music_search 获取真实线上候选，再用本工具排序和解释。",
        args_schema={
            "query": {"type": "string", "description": "用户的推荐请求原文，例如 '推荐适合工作时听的歌'"},
            "search_query": {
                "type": "string",
                "description": "供搜索/召回使用的改写查询，例如从原文中提炼出的场景、实体或正向检索词",
            },
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
    "similar_artists": ToolSpec(
        name="similar_artists",
        aliases=("recommend_similar_artists",),
        description="根据一个种子歌手，在可追溯曲库中寻找曲风和情绪标签相近的其他歌手。",
        args_schema={
            "artist": {"type": "string", "description": "种子歌手名"},
            "top_k": {"type": "integer", "default": 6},
        },
        required=("artist",),
        source="local_library",
    ),
    "playlist": ToolSpec(
        name="playlist",
        aliases=("generate_playlist",),
        description="根据用户指令生成一个主题歌单。适用于：用户明确说'做一个歌单'、'帮我整理'等。生成前应尽量已有 web_music_search 或导入结果作为真实候选；若用户给的是网易云歌单链接，应先调用 import_netease_playlist。",
        args_schema={
            "instruction": {"type": "string", "description": "歌单主题或场景，例如 '深夜专注用'、'跑步歌单'"},
            "target_count": {
                "type": "integer",
                "description": "用户要求的歌单曲目数，例如 50；未指定时由系统从 instruction 推断",
            },
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
    "playlist_repair": ToolSpec(
        name="playlist_repair",
        description="诊断上一轮歌单或推荐结果里的重复、脏数据、风格跳跃、能量断层、语言混杂和目标偏移，并给出修复建议。",
        args_schema={
            "instruction": {"type": "string", "description": "用户对想修复效果的补充描述，例如“更适合深夜”"},
            "target": {"type": "string", "description": "显式指定要修复的对象，例如“上一轮歌单”"},
        },
    ),
    "taste_shift_detector": ToolSpec(
        name="taste_shift_detector",
        description="分析用户近期与历史听歌偏好的变化，识别新上升的曲风、情绪和艺人。",
        args_schema={
            "window_recent_days": {"type": "integer", "default": 30},
            "window_baseline_days": {"type": "integer", "default": 90},
        },
    ),
    "music_fact_check": ToolSpec(
        name="music_fact_check",
        description="核验音乐事实陈述，只在有可追溯资料支持时标记 verified，证据不足时明确 uncertain。",
        args_schema={
            "query": {"type": "string", "description": "用户当前要核验的音乐问题或陈述"},
            "claims_text": {"type": "string", "description": "可选：明确给出待核验陈述"},
        },
        required=("query",),
    ),
    "recommend_explainer": ToolSpec(
        name="recommend_explainer",
        description="解释最近一轮推荐为什么会出现这些歌，基于真实推荐结果、用户口味和场景证据生成 grounded 说明。",
        args_schema={
            "query": {"type": "string", "description": "用户想解释的推荐轮次或补充说明"},
        },
    ),
    "resolve_music_entity": ToolSpec(
        name="resolve_music_entity",
        description="在严格时间预算内解析音乐实体（艺人/专辑/歌曲/流派），不确定时返回候选而非强行猜测。",
        args_schema={
            "query": {"type": "string", "description": "用户原始音乐知识问题"},
            "intent": {"type": "string", "description": "知识类意图"},
        },
        required=("query",),
        llm_visible=False,
        timeout_seconds=3.0,
        max_retries=0,
    ),
    "music_metadata_lookup": ToolSpec(
        name="music_metadata_lookup",
        description="并行查询音乐实体基础资料、代表作品和平台元数据，单源失败不影响整体。",
        args_schema={"query": {"type": "string", "description": "用户原始音乐知识问题"}},
        required=("query",),
        llm_visible=False,
        timeout_seconds=3.0,
        max_retries=0,
    ),
    "review_search": ToolSpec(
        name="review_search",
        description="在严格预算内搜索乐评/评价摘要，最多少量 query 和来源，不抓取全文。",
        args_schema={"query": {"type": "string", "description": "用户原始音乐知识问题"}},
        required=("query",),
        llm_visible=False,
        timeout_seconds=8.0,
        max_retries=0,
    ),
    "web_knowledge_search": ToolSpec(
        name="web_knowledge_search",
        description=(
            "强搜索 provider：为知识类问题（album/artist/review/compare/concert/fact_check）取结构化 "
            "claims + sources + citations。web provider 不可用时降级 DeepSeek 先验（claim 标未联网核实），"
            "全空时回退 legacy review_search。是知识链路取代 review_search 的主检索工具。"
        ),
        args_schema={
            "query": {"type": "string", "description": "用户原始音乐知识问题"},
            "intent": {"type": "string", "description": "知识类意图"},
        },
        required=("query",),
        llm_visible=False,
        # parametric 直答是首选 provider，一次长答案生成可达 ~25-35s（非流式，等整段算完）。
        # 必须 ≥ web_knowledge_timeout_seconds(40s)，否则工具墙会比 provider 预算先到，杀掉正在生成的答案。
        timeout_seconds=40.0,
        max_retries=0,
    ),
    "build_music_dossier": ToolSpec(
        name="build_music_dossier",
        description="把实体、元数据和乐评证据合成为结构化音乐档案；含正文抓取(Tavily Extract/Discogs API)+LLM 合成，超时或证据不足时输出 partial dossier。",
        args_schema={"query": {"type": "string", "description": "用户原始音乐知识问题"}},
        required=("query",),
        llm_visible=False,
        # web_knowledge 工具失败时，dossier 侧兜底会再生成一次直答（~20-30s）；放宽到 35s 让兜底能跑完。
        timeout_seconds=35.0,
        max_retries=0,
    ),
    "sample_relation_search": ToolSpec(
        name="sample_relation_search",
        description="检索歌曲采样/插值/翻唱/源曲关系，优先 WhoSampled/Genius/Discogs/Wikipedia 等可引用来源。",
        args_schema={"query": {"type": "string", "description": "用户原始采样查询"}},
        required=("query",),
        llm_visible=False,
        timeout_seconds=4.0,
        max_retries=0,
    ),
    "locate_sample_sources": ToolSpec(
        name="locate_sample_sources",
        description="把采样关系里的源曲定位到可播放平台候选，返回 SongCard。",
        args_schema={"query": {"type": "string", "description": "用户原始采样查询"}},
        required=("query",),
        llm_visible=False,
        timeout_seconds=4.0,
        max_retries=0,
    ),
    "build_sample_dossier": ToolSpec(
        name="build_sample_dossier",
        description="把采样证据、关系和源曲卡片合成为可解释的采样溯源结果。",
        args_schema={"query": {"type": "string", "description": "用户原始采样查询"}},
        required=("query",),
        llm_visible=False,
        timeout_seconds=3.0,
        max_retries=0,
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
    "feedback": ToolSpec(
        name="feedback",
        aliases=("record_feedback", "rate_track"),
        description="记录用户对真实曲目或歌手的反馈：like/dislike/skip/played。定位不到真实曲目时只记录文本偏好，不伪造曲目。",
        args_schema={
            "action": {"type": "string", "enum": ["like", "dislike", "skip", "played"]},
            "title": {"type": "string", "description": "歌名，可选"},
            "artist": {"type": "string", "description": "歌手，可选"},
            "reason": {"type": "string", "description": "用户给出的原因，可选"},
        },
        required=("action",),
    ),
    "listening_history": ToolSpec(
        name="listening_history",
        aliases=("my_history",),
        description="查询用户近期、近一周或近一月的听歌历史，并按曲目或歌手聚合。",
        args_schema={
            "window": {"type": "string", "enum": ["recent", "week", "month"], "default": "recent"},
            "group_by": {"type": "string", "enum": ["track", "artist"], "default": "track"},
            "top_k": {"type": "integer", "default": 10},
        },
    ),
    "list_my_playlists": ToolSpec(
        name="list_my_playlists",
        aliases=("my_playlists",),
        description="列出当前登录用户在网易云的歌单；未登录时返回扫码登录提示。",
        args_schema={},
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
        args_schema={"query": {"type": "string", "description": "歌曲、歌手或视频关键词"}},
        required=("query",),
        llm_visible=True,
    ),
    "web_info_search": ToolSpec(
        name="web_info_search",
        description="用搜索引擎查找歌手、乐队和作品的背景资料。",
        args_schema={"query": {"type": "string", "description": "要查证的歌手、乐队或作品"}},
        required=("query",),
        llm_visible=True,
    ),
    "find_on_platform": ToolSpec(
        name="find_on_platform",
        aliases=("locate_track",),
        description="把一首已知歌曲定位到网易云、YouTube 或 B站；只返回经平台搜索命中的结果。",
        args_schema={
            "title": {"type": "string"},
            "artist": {"type": "string"},
            "platform": {"type": "string", "enum": ["netease", "youtube", "bilibili"]},
        },
        required=("title", "platform"),
    ),
    "lyrics": ToolSpec(
        name="lyrics",
        description="查询网易云真实歌词。先解析真实 song_id；获取失败时返回空，绝不编造。",
        args_schema={
            "title": {"type": "string"},
            "artist": {"type": "string"},
            "song_id": {"type": "string"},
        },
    ),
    "audio_features": ToolSpec(
        name="audio_features",
        description="读取曲库中已有的 BPM、能量等真实音频特征；没有测量值时明确返回未知。",
        args_schema={
            "asset_id": {"type": "string"},
            "title": {"type": "string"},
            "artist": {"type": "string"},
        },
    ),
    "save_to_playlist": ToolSpec(
        name="save_to_playlist",
        description="预览把曲目加入网易云歌单的写操作。必须 confirm=true 才可执行；当前无可靠写接口时会安全拒绝。",
        args_schema={
            "playlist_id": {"type": "string"},
            "track_ids": {"type": "array", "items": {"type": "string"}},
            "confirm": {"type": "boolean", "default": False},
        },
        required=("playlist_id", "track_ids"),
    ),
    "favorite_track": ToolSpec(
        name="favorite_track",
        description="预览收藏网易云歌曲的写操作。必须 confirm=true；无可靠写接口时安全拒绝。",
        args_schema={
            "track_id": {"type": "string"},
            "confirm": {"type": "boolean", "default": False},
        },
        required=("track_id",),
    ),
    "concert_events": ToolSpec(
        name="concert_events",
        description="查找歌手公开演出或巡演信息，返回可追溯事件卡和网页来源；不会臆测用户位置。",
        args_schema={
            "artist": {"type": "string"},
            "city": {"type": "string"},
        },
        required=("artist",),
    ),
}

for _name in ("feedback", "memory_update"):
    TOOL_REGISTRY[_name].risk = ToolRisk.LOCAL_WRITE
    TOOL_REGISTRY[_name].idempotent = False
    TOOL_REGISTRY[_name].max_retries = 0
for _name in ("save_to_playlist", "favorite_track"):
    TOOL_REGISTRY[_name].risk = ToolRisk.EXTERNAL_WRITE
    TOOL_REGISTRY[_name].idempotent = False
    TOOL_REGISTRY[_name].max_retries = 0

# 差异化超时：默认 8.0s 对"内部要做多次网络+LLM"的重工具不够。journey 内部
# run_parallel(timeout=15.0)，被 8s 墙杀掉会让所有阶段空候选；recommend/playlist
# 走 LLM 发现 + 多轮网易云搜索 + 重排，也常 >8s。按真实内部耗时放宽。
_HEAVY_TOOL_TIMEOUTS = {
    "journey": 18.0,  # 内部 5 阶段并行歌单搜索 timeout=15.0，留余量
    "recommend": 30.0,  # LLM 发现 + 多轮网易云搜索 + 重排（国内访问偶尔 >20s）
    "playlist": 30.0,  # 同 recommend，且常要更多候选
    "web_music_search": 16.0,  # 多端点搜索 + 验证
    "taste_experiment": 20.0,  # 三档候选生成，LLM + 搜索
    "import_netease_playlist": 30.0,  # 批量导入大歌单
}
for _name, _timeout in _HEAVY_TOOL_TIMEOUTS.items():
    TOOL_REGISTRY[_name].timeout_seconds = _timeout


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


def get_tool_spec(name: str) -> ToolSpec | None:
    canonical = normalize_tool_name(name)
    return TOOL_REGISTRY.get(canonical) if canonical else None


def bind_tool_handler(name: str, handler: ToolHandler) -> None:
    spec = get_tool_spec(name)
    if spec is None:
        raise KeyError(name)
    spec.handler = handler


def bind_async_tool_handler(name: str, handler: ToolHandler) -> None:
    spec = get_tool_spec(name)
    if spec is None:
        raise KeyError(name)
    spec.async_handler = handler


def to_openai_tools() -> list[dict[str, Any]]:
    return [spec.openai_tool() for spec in TOOL_REGISTRY.values() if spec.llm_visible]
