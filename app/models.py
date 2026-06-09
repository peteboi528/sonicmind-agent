from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def today_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


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


class ListeningEvent(BaseModel):
    asset_id: str
    timestamp: str = Field(default_factory=utc_now_iso)
    duration_listened: int = 0
    completed: bool = False
    context: str | None = None


class TasteProfile(BaseModel):
    top_genres: list[tuple[str, float]] = Field(default_factory=list)
    top_moods: list[tuple[str, float]] = Field(default_factory=list)
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
    ratings: list["RatingEntry"] = Field(default_factory=list)
    dislikes: list[str] = Field(default_factory=list)
    taste_profile: TasteProfile | None = None
    daily_rec_last_generated: str | None = None
    updated_at: str = Field(default_factory=utc_now_iso)


class AgentAnswer(BaseModel):
    answer: str
    evidences: list[RagEvidence]
    recommended_segments: list[Segment] = Field(default_factory=list)
    memory_updated: bool = False
    agent_trace: list[str] = Field(default_factory=list)
    pending_goal: str | None = None
    goal_progress: list[str] = Field(default_factory=list)


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


class AgentPlan(BaseModel):
    # capability 意图：直接对齐 graph/nodes.py 的真实执行分支
    intent: Literal["recommend", "search", "playlist", "taste", "import", "journey", "chat"] = "chat"
    strategy: Literal["online_first", "library_first", "memory_only", "no_search"] = "online_first"
    tools_needed: list[str] = Field(default_factory=list)
    target_count: int | None = None
    online_required: bool = True
    reasoning_summary: str = ""
    retrieval_plan: RetrievalPlan = Field(default_factory=RetrievalPlan)


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


class StreamEvent(BaseModel):
    type: Literal[
        "plan", "thinking", "tool_start", "tool_result", "candidates",
        "song_card", "eval", "final", "guard", "error",
    ]
    content: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)


class FeedbackRequest(BaseModel):
    user_id: str = "demo-user"
    segment_id: str
    accepted: bool


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


def format_time(seconds: int) -> str:
    minutes, sec = divmod(max(0, seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"
