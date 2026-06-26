"""候选质量闸门 Candidate Quality Gate。

把教程/合集/歌单/DJ串烧/节目/新闻/vlog 等非歌曲(或低质)候选挡在推荐、歌单、搜索结果
与资源库之外。比单纯黑名单更强：规则高精度拦截 + sentence-transformers 语义原型分类
(治关键词没覆盖的「南宁Dj阿聪/全旋律说唱」之类) + query-aware 例外(用户明确要 DJ/串烧
时才放行 mix) + source 兜底。

判定三态 accept/maybe/reject：accept 直接入结果；maybe 仅在 allow_maybe 时入；reject 一律挡。
embedding 不可用时安全降级到「规则 + source」，离线测试确定。
"""
from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.retrieval.embeddings import embeddings_available, semantic_scores

# ── 规则层：高精度拦截明显垃圾 ──────────────────────────────────────────
# 纯非歌曲/低质片段：教程/解说/合集/歌单/节目/新闻/vlog/广播剧/高潮片段。一律拒。
HARD_REJECT_PATTERNS = (
    "教程", "教学", "编曲", "编曲技巧", "怎么做", "如何制作", "合集", "全集", "精选集",
    "歌单", "playlist", "reaction", "访谈", "解说", "节目", "电台", "混剪",
    "抖音热播", "抖音热门", "高潮版", "片段版",
    "广播剧", "原声带", "同人", "警示录", "车祸", "事故", "实录", "监控", "录像",
)
# DJ / 串烧 / mix 模式（默认拒；query 明确要 mix/DJ/车载/慢摇 时才放行）。
# 注意：车载/串烧/慢摇/全网最火/低音炮这些放这里(query-aware)，不放 HARD_REJECT，
# 否则用户明确要「车载DJ串烧」时会被 hard_reject 先拦、永远拿不到。
MIX_PATTERNS = (
    "dj", "串烧", "慢摇", "车载", "低音炮", "全网最火", "club mix", "mixset",
    "remix set", "continuous mix", "dj mix",
)
# 句子/新闻/节目型标题的强标点——歌曲几乎不用。
_SENTENCE_PUNCT = ("。", "！", "？", "【", "】")
# query 里表示「允许 mix/DJ」的词。
_MIX_ALLOW_TERMS = ("dj", "串烧", "慢摇", "车载", "mix", "remix set", "club set", "舞曲")
# query 里表示「要 track-level 歌曲」的词（决定是否强校验）。
_TRACK_LEVEL_TERMS = ("歌", "歌曲", "单曲", "歌单", "跑步", "学习", "推荐", "playlist", "song", "track", "深夜", "安静")

# 网易云用户上传的「氛围/翻唱/助眠」假歌签名：标题形如「曲名 - <氛围|男声|女声|助眠…>」。
# 这种描述后缀（"雨爱 - R&B氛围男声"）是用户上传 mood 合辑/翻唱的固定写法，艺人栏往往是
# 网名（小仓鼠要早睡）。官方曲目几乎不用这种后缀（官方 remix/live 不在此列），故高精度硬拒。
# 必须在破折号之后的描述段里命中，避免误伤标题本身就含「氛围/深夜」的真歌。
_MOOD_DESCRIPTOR_RE = re.compile(
    r"\s[-－—]\s*\S{0,12}?(氛围|男声|女声|助眠|催眠|睡眠|八音盒|吉他版|钢琴版|加长版|"
    r"治愈系|降噪版|纯人声|slowed|sped\s*up|\b8d\b|reverb\b|piano\s*cover|guitar\s*cover)",
    re.IGNORECASE,
)

# ── 语义原型：embedding 不可用时整段跳过，确定可降级 ──
QUALITY_PROTOTYPES: dict[str, list[str]] = {
    "song_track": [
        "a single song by a music artist",
        "official song track with title and artist",
        "一首由歌手演唱或制作的正式歌曲",
        "专辑中的一首正式曲目",
    ],
    "dj_mix": [
        "DJ mix set with many songs combined",
        "continuous dance mix or club mix",
        "DJ串烧歌曲合集",
        "车载DJ舞曲串烧",
    ],
    "playlist_collection": [
        "a playlist collection of many songs",
        "music compilation playlist",
        "歌曲合集歌单全集",
        "热门歌曲合集",
    ],
    "tutorial_content": [
        "music production tutorial",
        "how to make R&B music",
        "编曲教程音乐制作教学",
        "如何制作一首歌的教程",
    ],
    "program_video": [
        "music commentary video",
        "reaction video or interview",
        "音乐解说视频",
        "访谈节目不是歌曲",
    ],
}


class CandidateQuality(BaseModel):
    """单个候选的质量判定结果。"""
    status: Literal["accept", "maybe", "reject"] = "maybe"
    entity_type: Literal[
        "track", "album", "playlist", "video", "tutorial", "dj_mix", "program", "unknown",
    ] = "unknown"
    track_score: float = 0.0
    junk_score: float = 0.0
    confidence: float = 0.0
    reasons: list[str] = Field(default_factory=list)


class HygieneReport(BaseModel):
    """一批候选的清洗报告——让 trace/文案能解释「为什么结果变少」。"""
    requested_count: int = 0
    raw_count: int = 0
    accepted_count: int = 0
    maybe_count: int = 0
    rejected_count: int = 0
    rejected_examples: list[str] = Field(default_factory=list)
    reasons: dict[str, int] = Field(default_factory=dict)

    def removed_total(self) -> int:
        return self.raw_count - self.accepted_count


def candidate_text(track: Any) -> str:
    """归一化候选为可判别文本：title + artist + album + tags。"""
    parts = [
        str(getattr(track, "title", "") or ""),
        str(getattr(track, "artist", "") or ""),
        str(getattr(track, "album", "") or ""),
    ]
    for attr in ("genre", "mood"):
        vals = getattr(track, attr, None) or []
        if isinstance(vals, (list, tuple)):
            parts.extend(str(v) for v in vals)
    return " ".join(p for p in parts if p).strip()


def query_allows_mix(query: str) -> bool:
    """用户是否明确要 DJ/串烧/mix——只有这时 mix 类才放行。"""
    q = (query or "").lower()
    return any(term in q for term in _MIX_ALLOW_TERMS)


def query_requires_track_level(query: str) -> bool:
    """用户是否需要 track-level 歌曲（决定是否对 maybe/视频源更严）。"""
    if query_allows_mix(query):
        return False
    q = (query or "").lower()
    return any(term in q for term in _TRACK_LEVEL_TERMS)


def semantic_prototype_scores(text: str) -> dict[str, float]:
    """用 embedding 把候选文本对齐到各质量原型，返回 {label: max_score}。

    embedding 不可用返回 {}——上层据此跳过语义判断、走规则+source，保证离线确定。
    注意 multilingual embedding 分数常整体偏高，调用方必须用「类别间相对 margin」而非绝对阈值。
    """
    if not text or not embeddings_available():
        return {}
    labels: list[str] = []
    proto_texts: list[str] = []
    for label, items in QUALITY_PROTOTYPES.items():
        for item in items:
            labels.append(label)
            proto_texts.append(item)
    scores = semantic_scores(text, proto_texts)
    if scores is None:
        return {}
    out = {label: 0.0 for label in QUALITY_PROTOTYPES}
    for label, score in zip(labels, scores, strict=False):
        out[label] = max(out[label], float(score))
    return out


def classify_candidate(track: Any, query: str = "") -> CandidateQuality:
    """判定单个候选是不是「该进推荐/歌单的歌曲」。

    顺序：基础字段 → 句子标点 → 硬拒关键词 → DJ/mix(query-aware) → 语义原型margin → source 兜底。
    """
    title = str(getattr(track, "title", "") or "").strip()
    artist = str(getattr(track, "artist", "") or "").strip()
    source = str(getattr(track, "source", "") or "").lower()
    kind = str(getattr(track, "candidate_kind", "") or "").strip().lower()

    # 1. 基础字段
    if not title:
        return CandidateQuality(status="reject", entity_type="unknown", confidence=1.0, reasons=["missing_title"])
    if not artist and source in {"bilibili", "youtube"}:
        return CandidateQuality(status="reject", entity_type="video", confidence=0.8, reasons=["missing_artist_for_video_source"])

    # candidate_kind 七分类里明确非单曲的实体直接拒。
    if kind in {"playlist", "compilation", "long_mix", "lyrics_video"}:
        return CandidateQuality(status="reject", entity_type="playlist", junk_score=1.0, confidence=0.9, reasons=[f"candidate_kind:{kind}"])

    text = candidate_text(track)
    lower = text.lower()

    # 2. 句子/新闻/节目型标题（。！？【）——歌曲几乎不用。
    if any(p in title for p in _SENTENCE_PUNCT):
        return CandidateQuality(status="reject", entity_type="program", junk_score=1.0, confidence=0.9, reasons=["sentence_punct_title"])

    # 2b. 网易云用户上传的氛围/翻唱/助眠假歌：「曲名 - <氛围|男声|女声|助眠…>」后缀。
    #     挡在 source 兜底（netease+artist=accept）之前，否则这类脏候选会被无条件放行。
    if _MOOD_DESCRIPTOR_RE.search(title):
        return CandidateQuality(status="reject", entity_type="playlist", junk_score=1.0, confidence=0.9, reasons=["mood_descriptor_title"])
    # bilibili 句子标题（含逗号的长句 vlog/新闻）。
    if source == "bilibili" and ("，" in title or "！" in title):
        return CandidateQuality(status="reject", entity_type="program", junk_score=0.95, confidence=0.85, reasons=["bilibili_sentence_title"])

    # 3. 硬拒关键词
    if any(p in lower for p in HARD_REJECT_PATTERNS):
        etype = "tutorial" if any(w in lower for w in ("教程", "教学", "编曲")) else "playlist"
        return CandidateQuality(status="reject", entity_type=etype, junk_score=1.0, confidence=0.9, reasons=["hard_reject_pattern"])

    # 4. DJ / mix（query-aware：用户没明确要 mix 就拒）
    is_mix = any(p in lower for p in MIX_PATTERNS)
    if is_mix and not query_allows_mix(query):
        return CandidateQuality(status="reject", entity_type="dj_mix", junk_score=0.95, confidence=0.9, reasons=["mix_not_allowed_by_query"])

    # 5. 语义原型 margin：未命中关键词但语义像 junk 的，按 source 区别对待——
    #    视频源(bilibili/youtube)天然多 junk，语义判 junk 即拒；
    #    可信源(netease/spotify/local 带 artist)可能是真歌被 embedding 误判(如 Firework)，
    #    不因语义硬拒，回落 source 兜底，防误杀。
    scores = semantic_prototype_scores(text)
    if scores:
        track_score = scores.get("song_track", 0.0)
        junk = {k: v for k, v in scores.items() if k != "song_track"}
        best_junk_label, best_junk_score = ("", 0.0)
        if junk:
            best_junk_label, best_junk_score = max(junk.items(), key=lambda kv: kv[1])
        semantic_junk = best_junk_score > track_score + 0.08
        if semantic_junk and source in {"bilibili", "youtube"}:
            return CandidateQuality(status="reject", entity_type=best_junk_label or "unknown", track_score=track_score, junk_score=best_junk_score, confidence=0.7, reasons=[f"semantic_junk:{best_junk_label}"])
        # 用户明确要 mix 时，语义判 dj_mix 降为 maybe（出口 allow_maybe=True 时可入）。
        if semantic_junk and best_junk_label == "dj_mix" and query_allows_mix(query):
            return CandidateQuality(status="maybe", entity_type="dj_mix", track_score=track_score, junk_score=best_junk_score, confidence=0.6, reasons=["semantic_mix_allowed"])
        # 可信源不因语义硬拒；继续回落到 source 兜底。

    # 6. source 兜底：网云带艺人 → accept；视频源 → maybe；其余 maybe。
    if source == "netease" and artist:
        return CandidateQuality(status="accept", entity_type="track", confidence=0.6, reasons=["netease_with_artist"])
    if source in {"bilibili", "youtube"}:
        return CandidateQuality(status="maybe", entity_type="video", confidence=0.4, reasons=["video_source_uncertain"])
    if source in {"spotify", "lastfm"} and artist:
        return CandidateQuality(status="accept", entity_type="track", confidence=0.6, reasons=[f"{source}_with_artist"])
    if source in {"local"} and artist:
        return CandidateQuality(status="accept", entity_type="track", confidence=0.6, reasons=["local_with_artist"])
    return CandidateQuality(status="maybe", entity_type="unknown", confidence=0.3, reasons=["fallback_maybe"])


def filter_music_tracks(
    tracks: list[Any],
    query: str = "",
    *,
    allow_maybe: bool = False,
    target_count: int | None = None,
) -> tuple[list[Any], HygieneReport]:
    """统一出口过滤：只留 accept（allow_maybe=True 时也收 maybe）。

    返回 (accepted, report)。report 记录 accepted/rejected/reasons/rejected_examples，
    供 trace 与「诚实数量文案」使用。
    """
    accepted: list[Any] = []
    maybe_count = 0
    rejected_examples: list[str] = []
    reasons: dict[str, int] = {}

    for track in (tracks or []):
        quality = classify_candidate(track, query)
        for reason in quality.reasons:
            reasons[reason] = reasons.get(reason, 0) + 1
        if quality.status == "accept" or (allow_maybe and quality.status == "maybe"):
            accepted.append(track)
            if quality.status == "maybe":
                maybe_count += 1
        else:
            if len(rejected_examples) < 5:
                t = str(getattr(track, "title", "") or "")
                a = str(getattr(track, "artist", "") or "")
                rejected_examples.append(f"{t} - {a}".strip(" -"))

    report = HygieneReport(
        requested_count=int(target_count or 0),
        raw_count=len(tracks or []),
        accepted_count=len(accepted),
        maybe_count=maybe_count,
        rejected_count=len(tracks or []) - len(accepted),
        rejected_examples=rejected_examples,
        reasons=reasons,
    )
    return accepted, report
