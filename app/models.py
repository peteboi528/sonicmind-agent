from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, PrivateAttr, field_validator


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def today_str() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d")


class AssetStatus(StrEnum):
    INGESTED = "ingested"
    ANALYZED = "analyzed"
    FAILED = "failed"


class Modality(StrEnum):
    TEXT = "text"
    VISION = "vision"
    AUDIO = "audio"
    SUMMARY = "summary"


class Asset(BaseModel):
    asset_id: str
    source_url: str
    title: str
    duration_seconds: int
    local_path: str | None = None
    status: AssetStatus = AssetStatus.INGESTED
    tags_fingerprint: list[str] = Field(default_factory=list)
    media_type: Literal["audio", "video"] = "audio"
    genre: list[str] = Field(default_factory=list)
    mood: list[str] = Field(default_factory=list)
    tempo_bpm: int | None = None
    energy_level: float | None = None
    artist: str | None = None
    album: str | None = None
    cover_url: str | None = None
    source: Literal["local", "external"] = "local"
    external_id: str | None = None
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)


class Segment(BaseModel):
    segment_id: str
    asset_id: str
    start_seconds: int
    end_seconds: int
    transcript: str
    keyframe_path: str | None = None
    visual_tags: list[str] = Field(default_factory=list)
    audio_tags: list[str] = Field(default_factory=list)
    scene_summary: str

    @property
    def timestamp(self) -> str:
        return f"{format_time(self.start_seconds)}-{format_time(self.end_seconds)}"

    def searchable_text(self) -> str:
        parts = [
            self.transcript,
            self.scene_summary,
            " ".join(self.visual_tags),
            " ".join(self.audio_tags),
        ]
        return " ".join(part for part in parts if part)


class RagEvidence(BaseModel):
    segment_id: str
    timestamp: str
    content: str
    modality: Modality
    keyframe_path: str | None = None
    similarity: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryEntry(BaseModel):
    text: str
    frequency: int = 1
    last_used: str = Field(default_factory=utc_now_iso)
    source: str = "user_event"


class EpisodicMemory(BaseModel):
    """情景记忆：某一次具体交互的快照（"3 周前说过想要慵懒爵士"）。

    与 structured_preferences（语义记忆：稳定口味）区分：
    - kind="episodic"：一次性事件，带 embedding 供跨会语义召回。
    - embedding 为空时（无 sentence-transformers）退化为不可语义召回，
      但仍按时间衰减作为近期事件保留，零依赖路径不破坏。
    """

    text: str
    kind: str = "episodic"  # episodic | semantic
    embedding: list[float] = Field(default_factory=list)
    source: str = "turn"
    timestamp: str = Field(default_factory=utc_now_iso)


class ListeningEvent(BaseModel):
    asset_id: str
    timestamp: str = Field(default_factory=utc_now_iso)
    duration_listened: int = 0
    completed: bool = False
    context: str | None = None


class TasteProfile(BaseModel):
    top_genres: list[tuple[str, float]] = Field(default_factory=list)
    top_moods: list[tuple[str, float]] = Field(default_factory=list)
    top_artists: list[tuple[str, float]] = Field(default_factory=list)
    preferred_energy: float = 0.5
    preferred_tempo_range: list[int] = Field(default_factory=lambda: [80, 140])
    discovery_openness: float = 0.3


class UserMemory(BaseModel):
    user_id: str
    preferences: list[str] = Field(default_factory=list)
    structured_preferences: list[MemoryEntry] = Field(default_factory=list)
    common_goals: list[str] = Field(default_factory=list)
    confirmed_segments: list[str] = Field(default_factory=list)
    project_notes: list[str] = Field(default_factory=list)
    listening_history: list[ListeningEvent] = Field(default_factory=list)
    ratings: list[RatingEntry] = Field(default_factory=list)
    dislikes: list[str] = Field(default_factory=list)
    exclusion_rules: list[str] = Field(default_factory=list)  # 用户明确排除的风格/类型，如 ["抖音热歌", "中文孟菲斯说唱"]
    taste_profile: TasteProfile | None = None
    # P1-G 记忆升级：情景记忆（跨会语义召回）+ 巩固画像（一句话稳定口味）。
    episodic_memory: list[EpisodicMemory] = Field(default_factory=list)
    consolidated_profile: str = ""  # 每 N 轮由 LLM 把零散偏好巩固成一句话画像
    turns_since_consolidation: int = 0
    daily_rec_last_generated: str | None = None
    updated_at: str = Field(default_factory=utc_now_iso)


class AgentAnswer(BaseModel):
    answer: str
    evidences: list[RagEvidence]
    recommended_segments: list[Segment] = Field(default_factory=list)
    recommended_tracks: list[TrackRef] = Field(default_factory=list)
    prompt_versions: dict[str, str] = Field(default_factory=dict)
    runtime_metrics: dict[str, float | int] = Field(default_factory=dict)
    memory_updated: bool = False
    agent_trace: list[str] = Field(default_factory=list)
    pending_goal: str | None = None
    goal_progress: list[str] = Field(default_factory=list)
    # 标记本轮是否走了降级路径（LLM 失败 → 关键词/模板兜底），便于排查"对话僵硬"。
    fallback_reason: str | None = None
    _compound_cards: list[dict[str, Any]] = PrivateAttr(default_factory=list)


class IngestRequest(BaseModel):
    url: str
    force_refresh: bool = False


class RetrieveRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=12)


class AskRequest(BaseModel):
    user_id: str = "demo-user"
    question: str
    top_k: int = Field(default=5, ge=1, le=12)


class MemoryUpdateRequest(BaseModel):
    user_id: str = "demo-user"
    event: str
    asset_id: str | None = None
    segment_id: str | None = None


class RecommendRequest(BaseModel):
    user_id: str = "demo-user"
    goal: str = "Find cinematic moments for a trailer."
    top_k: int = Field(default=3, ge=1, le=8)


class SimilarAssetResult(BaseModel):
    asset_id: str
    title: str
    score: float
    shared_tags: list[str]


class SimilarSegmentResult(BaseModel):
    segment: Segment
    score: float
    matching_modalities: list[str]


class ReActStep(BaseModel):
    thought: str
    action: str
    observation: str


class AgentGoal(BaseModel):
    goal: str
    steps_done: list[str] = Field(default_factory=list)
    steps_pending: list[str] = Field(default_factory=list)
    status: Literal["active", "completed", "blocked"] = "active"
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)


class DialogueState(BaseModel):
    """轻量多轮对话状态：用于"再来几首/换一批/类似这个"这类延续请求。

    与 UserMemory（长期偏好）分开存储在 dialogue/{user_id}.json，只记最近一轮的
    意图与实体，话题切换时清空。load_context 读取，plan_intent 在 LLM 未抽到实体
    的延续指令上程序化继承，finalize 写回。
    """
    user_id: str
    last_intent: str = "chat"
    last_query: str = ""
    entities: list[str] = Field(default_factory=list)
    genre_tags: list[str] = Field(default_factory=list)
    mood_tags: list[str] = Field(default_factory=list)
    scenario_tags: list[str] = Field(default_factory=list)
    turn_count: int = 0
    shown_tracks: list[dict[str, str]] = Field(default_factory=list)
    """每轮已展示给用户的歌曲摘要 [{"title":..., "artist":..., "source":..., "source_id":...}]。
    延续指令（多来几首/换一批）时用于去重，避免重复推荐同一首歌。
    """
    updated_at: str = Field(default_factory=utc_now_iso)


class RetrievalPlan(BaseModel):
    """结构化检索执行计划（对齐 SoulTuner MusicQueryPlan.retrieval_plan 思想）。

    分工：LLM 只负责判意图 + 抽实体名（entities）；genre/mood/scenario 等标签
    由确定性规则（app/graph/tag_rules.py）填充，降低幻觉与成本。
    """
    use_local: bool = False        # 检索本地库 / 候选资源库
    use_vector: bool = False       # 启用语义向量检索（sentence-transformers / TF 降级）
    use_web: bool = False          # 联网搜索真实平台候选
    entities: list[str] = Field(default_factory=list)        # LLM 抽取的实体（歌手/歌名）
    genre_filter: list[str] = Field(default_factory=list)    # 规则填充
    mood_filter: list[str] = Field(default_factory=list)     # 规则填充
    scenario_filter: list[str] = Field(default_factory=list) # 规则填充
    # LLM 合成的自包含正向检索词：把对话历史 + 本轮约束（含"不要中文"这类否定）
    # 改写成可直接喂搜索 API 的正向 query（否定尽量转正向，如"不要中文"→"英文 欧美"）。
    # 为空时检索层降级回 _extract_search_query 关键词切词路径（mock/无 key 不破）。
    search_query: str = ""
    language_filter: str = ""      # 语言偏好 zh/en/ja/ko/...，非空时对候选做安全后过滤


class AgentPlan(BaseModel):
    # capability 意图：对照 app.intents.INTENT_REGISTRY 校验。
    # 用 str + validator 而非 Literal，新增意图只需改 registry，不会因漏改
    # 这里的 Literal 触发 Pydantic 500（历史上 discuss 就是这样炸的）。
    # 未知意图统一降级为 chat，保证主流程不崩。
    intent: str = "chat"
    strategy: Literal["online_first", "library_first", "memory_only", "no_search"] = "online_first"
    tools_needed: list[str] = Field(default_factory=list)
    target_count: int | None = None
    online_required: bool = True
    reasoning_summary: str = ""
    retrieval_plan: RetrievalPlan = Field(default_factory=RetrievalPlan)
    _excluded_tracks: list[dict[str, str]] = PrivateAttr(default_factory=list)

    @field_validator("intent", mode="before")
    @classmethod
    def _coerce_intent(cls, v: object) -> str:
        from app.intents import is_valid_intent
        s = str(v or "chat")
        return s if is_valid_intent(s) else "chat"


class QueryPlanPayload(BaseModel):
    intent: str
    entities: list[str] = Field(default_factory=list)
    use_local: bool = True
    use_vector: bool = False
    use_web: bool = True
    search_query: str = ""
    language: str = ""
    target_count: int | None = None
    reasoning: str = ""

    @field_validator("intent", mode="before")
    @classmethod
    def _validate_intent(cls, v: object) -> str:
        from app.intents import is_valid_intent

        s = str(v or "").strip()
        if not is_valid_intent(s):
            raise ValueError(f"invalid intent: {s}")
        return s

    @field_validator("entities", mode="before")
    @classmethod
    def _coerce_entities(cls, v: object) -> list[str]:
        if v is None:
            return []
        if not isinstance(v, list):
            raise ValueError("entities must be a list")
        return [str(item).strip() for item in v if str(item).strip()]

    @field_validator("search_query", "language", "reasoning", mode="before")
    @classmethod
    def _coerce_text(cls, v: object) -> str:
        return str(v or "").strip()

    @field_validator("target_count", mode="before")
    @classmethod
    def _coerce_target_count(cls, v: object) -> int | None:
        if v in {None, "", 0, "0"}:
            return None
        if isinstance(v, (int, float)):
            return int(v)
        try:
            return int(str(v).strip())
        except Exception as exc:
            raise ValueError("target_count must be int-like") from exc


class ResourceTrack(BaseModel):
    title: str
    artist: str = ""
    source: str = "unknown"
    source_id: str = ""
    genre: list[str] = Field(default_factory=list)
    mood: list[str] = Field(default_factory=list)
    playback_url: str | None = None
    verified: bool = False
    last_seen: str = Field(default_factory=utc_now_iso)
    exposure_count: int = 0


class RankingBreakdown(BaseModel):
    title: str
    source: str
    score: float
    reason: str
    components: dict[str, float] = Field(default_factory=dict)


class TrackRef(BaseModel):
    title: str
    artist: str = ""
    source: str = "local"
    source_id: str = ""
    genre: list[str] = Field(default_factory=list)
    mood: list[str] = Field(default_factory=list)
    score: float | None = None
    components: dict[str, float] = Field(default_factory=dict)


class TasteExperimentFeedback(BaseModel):
    completed: int = 0
    skipped: int = 0
    liked: int = 0
    disliked: int = 0
    saved: int = 0
    rated: int = 0
    too_safe: int = 0
    too_far: int = 0
    scores: list[float] = Field(default_factory=list)
    last_signal: str = ""


class TasteExperimentTrack(BaseModel):
    track: TrackRef
    bucket: Literal["safe", "stretch", "bold"]
    reason: str = ""
    expected_signal: str = ""
    components: dict[str, float] = Field(default_factory=dict)
    feedback: TasteExperimentFeedback = Field(default_factory=TasteExperimentFeedback)


class TasteExperimentSegment(BaseModel):
    name: Literal["safe", "stretch", "bold"]
    label: str
    description: str
    tracks: list[TasteExperimentTrack] = Field(default_factory=list)


class TasteExperimentReport(BaseModel):
    summary: str = ""
    bucket_stats: dict[str, dict[str, float | int]] = Field(default_factory=dict)
    hypothesis_result: str = ""
    next_recommendation_strategy: str = ""


class TasteExperiment(BaseModel):
    experiment_id: str
    user_id: str
    hypothesis: str
    status: Literal["collecting", "ready", "reported"] = "collecting"
    prompt: str = ""
    segments: list[TasteExperimentSegment] = Field(default_factory=list)
    result_summary: str = ""
    report: TasteExperimentReport | None = None
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)


class StreamEvent(BaseModel):
    type: Literal[
        "plan", "thinking", "tool_start", "tool_result", "candidates",
        "song_card", "album_card", "eval", "final", "guard", "error",
    ]
    content: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)


class FeedbackRequest(BaseModel):
    user_id: str = "demo-user"
    segment_id: str
    accepted: bool


class TasteExperimentRequest(BaseModel):
    user_id: str = "demo-user"
    prompt: str = "探索我的口味"
    total: int = Field(default=12, ge=3, le=30)


class TasteExperimentFeedbackRequest(BaseModel):
    user_id: str = "demo-user"
    experiment_id: str
    track_key: str
    signal: Literal[
        "completed", "skipped", "liked", "disliked", "saved",
        "rated", "too_safe", "too_far",
    ]
    score: float | None = Field(default=None, ge=0.0, le=10.0)


class TasteExperimentReportRequest(BaseModel):
    user_id: str = "demo-user"
    experiment_id: str


class TasteExperimentRegenerateRequest(BaseModel):
    user_id: str = "demo-user"
    experiment_id: str
    bucket: Literal["safe", "stretch", "bold"]


class DislikeRequest(BaseModel):
    user_id: str = "demo-user"
    title: str = ""
    artist: str = ""
    source: str = ""
    source_id: str = ""
    reason: str = ""


class JourneyRequest(BaseModel):
    user_id: str = "demo-user"
    instruction: str


class SimilarRequest(BaseModel):
    top_k: int = Field(default=5, ge=1, le=20)


class ReactRequest(BaseModel):
    user_id: str = "demo-user"
    query: str
    top_k: int = Field(default=5, ge=1, le=12)


class ExternalTrack(BaseModel):
    external_id: str
    title: str
    artist: str
    album: str | None = None
    genre: list[str] = Field(default_factory=list)
    mood: list[str] = Field(default_factory=list)
    tempo_bpm: int | None = None
    energy_level: float | None = None
    cover_url: str | None = None
    preview_url: str | None = None
    playback_url: str | None = None
    source: str = "mock"
    # 候选类型（七分类）：
    #   track        单曲（保留）
    #   official_mv  官方 MV / 现场（可绑定单曲，保留）
    #   lyrics_video 动态歌词/歌词版（过滤）
    #   playlist     歌单/榜单（过滤）
    #   compilation  合集/连播/串烧/精选集（过滤）
    #   long_mix     长混音/连续播放/DJ mix（过滤）
    #   unknown      无明确信号（兜底保留）
    candidate_kind: Literal[
        "track", "official_mv", "lyrics_video", "playlist",
        "compilation", "long_mix", "unknown",
    ] = "track"


TrackOrigin = Literal["local", "netease", "bilibili", "youtube", "mock", "llm_guess"]


class TrackEntity(BaseModel):
    """候选池契约：只有进入候选池的 TrackEntity 才允许出现在最终答案里。

    verified=True 表示 title/artist 经过真实来源回查（如网易云 song detail API）；
    verified=False（尤其 origin='llm_guess'）表示由 LLM 生成、未经核实，
    Answer Guard 默认不允许其出现在面向用户的文本中。
    """
    title: str
    artist: str = ""
    source: str = "unknown"
    source_id: str = ""
    verified: bool = False
    origin: TrackOrigin = "llm_guess"
    evidence_ref: str | None = None

    def display_key(self) -> str:
        """用于 Answer Guard 去重/白名单比对的归一化键。"""
        return f"{self.title.strip().lower()}|{self.artist.strip().lower()}"


class RecommendedTrack(BaseModel):
    asset: Asset | ExternalTrack
    score: float
    reason: str
    category: Literal["familiar", "discovery", "mood_match"] = "familiar"
    components: dict[str, float] = Field(default_factory=dict)


class DailyRecommendation(BaseModel):
    user_id: str
    date: str = Field(default_factory=today_str)
    tracks: list[RecommendedTrack] = Field(default_factory=list)
    generated_at: str = Field(default_factory=utc_now_iso)
    reason_summary: str = ""
    evidences: list[RagEvidence] = Field(default_factory=list)
    agent_trace: list[str] = Field(default_factory=list)


class ListenRequest(BaseModel):
    user_id: str = "demo-user"
    asset_id: str
    duration: int = 0
    completed: bool = False
    context: str | None = None


class RatingRequest(BaseModel):
    user_id: str = "demo-user"
    asset_id: str
    score: float = Field(ge=0.0, le=10.0)


class RatingEntry(BaseModel):
    asset_id: str
    score: float
    title: str = ""
    artist: str = ""
    genre: list[str] = Field(default_factory=list)
    mood: list[str] = Field(default_factory=list)
    timestamp: str = Field(default_factory=utc_now_iso)


class SearchRequest(BaseModel):
    user_id: str = "demo-user"
    query: str
    include_external: bool = True
    top_k: int = Field(default=20, ge=1, le=50)


class SearchResponse(BaseModel):
    local: list[Asset] = Field(default_factory=list)
    external: list[ExternalTrack] = Field(default_factory=list)
    summary: str = ""
    evidences: list[RagEvidence] = Field(default_factory=list)
    agent_trace: list[str] = Field(default_factory=list)


class ChatMessage(BaseModel):
    role: str  # "user" | "assistant" | "system"
    content: str


class ChatRequest(BaseModel):
    user_id: str = "demo-user"
    message: str
    history: list[ChatMessage] = Field(default_factory=list)


class DailyRequest(BaseModel):
    user_id: str = "demo-user"
    time_of_day: str | None = None


class Playlist(BaseModel):
    playlist_id: str
    user_id: str
    name: str
    description: str = ""
    tracks: list[Asset | ExternalTrack] = Field(default_factory=list)
    generated_by: str = "llm"
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)


class PlaylistRequest(BaseModel):
    user_id: str = "demo-user"
    instruction: str


class EnrichRequest(BaseModel):
    use_network: bool = False


class EnrichResponse(BaseModel):
    asset: Asset
    enriched: bool = False
    mode: str = "offline"
    note: str = ""


# ── Discover / Browse ──

class BrowseRequest(BaseModel):
    user_id: str = "demo-user"
    category: str  # "genre" | "mood" | "scene"
    value: str     # "摇滚" | "放松" | "深夜"
    limit: int = Field(default=12, ge=1, le=30)
    seed: int = Field(default=0, ge=0, le=50)  # 换一批：按 seed 轮换关键词/歌单数，让同一分类能取到不同曲目


class TrendingRequest(BaseModel):
    user_id: str = "demo-user"
    limit: int = Field(default=12, ge=1, le=30)


class ArtistInfoRequest(BaseModel):
    artist: str


class ArtistAlbum(BaseModel):
    id: str = ""
    name: str
    image: str = ""
    artist: str = ""
    track_count: int | None = None


class AlbumTracksRequest(BaseModel):
    artist: str
    album: str
    album_id: str | None = None
    limit: int = Field(default=100, ge=1, le=100)


class AlbumTracksResponse(BaseModel):
    album: ArtistAlbum
    tracks: list[ExternalTrack] = Field(default_factory=list)
    summary: str = ""


class SavedAlbum(BaseModel):
    """用户收藏的专辑：保存专辑元数据 + 完整曲目，便于从「我的库」直接整张播放，
    无需重新搜索/取网易云。与 Playlist 同构存储（collection=saved_albums，
    key=f"{user_id}_{album_id}"）。只有带真实 album_id 的专辑（网易云源）可收藏，
    故回放可靠。"""
    album_id: str
    user_id: str = "demo-user"
    name: str
    artist: str = ""
    image: str = ""
    track_count: int | None = None
    tracks: list[ExternalTrack] = Field(default_factory=list)
    saved_at: str = Field(default_factory=utc_now_iso)


class SaveAlbumRequest(BaseModel):
    user_id: str = "demo-user"
    album_id: str
    name: str
    artist: str = ""
    image: str = ""
    track_count: int | None = None
    tracks: list[ExternalTrack] = Field(default_factory=list)


class ArtistInfoResponse(BaseModel):
    name: str
    image: str = ""
    bio: str = ""
    tags: list[str] = Field(default_factory=list)
    top_albums: list[ArtistAlbum] = Field(default_factory=list)
    top_tracks: list[ExternalTrack] = Field(default_factory=list)


def format_time(seconds: int) -> str:
    minutes, sec = divmod(max(0, seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"
