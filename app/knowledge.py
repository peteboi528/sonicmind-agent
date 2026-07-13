from __future__ import annotations

import logging
import re
import time
from datetime import UTC, datetime, timedelta
from typing import Any

from pydantic import BaseModel, Field

from app.concurrency import run_parallel
from app.config import settings
from app.llm.structured import extract_json_dict
from app.models import (
    CareerPhase,
    EvidenceConsistencyReport,
    KnowledgeEvidencePack,
    LibraryMatch,
    MusicCitation,
    MusicDossier,
    MusicEntity,
    ReviewOpinion,
    SampleDossier,
    SampleEvidence,
    SampleRelation,
    TrackRef,
    utc_now_iso,
)
from app.sources import web_search as web_search_source

logger = logging.getLogger(__name__)

KNOWLEDGE_INTENTS = {"album_deep_dive", "artist_deep_dive", "review_summary", "music_compare", "sample_lookup"}
KNOWLEDGE_TOOLS = {
    "resolve_music_entity",
    "music_metadata_lookup",
    "review_search",
    "build_music_dossier",
    "sample_relation_search",
    "locate_sample_sources",
    "build_sample_dossier",
}

TIER_A_SOURCES = {
    "pitchfork",
    "pitchforkmedia",
    "allmusic",
    "theguardian",
    "guardian",
    "rollingstone",
    "nme",
    "bbc",
    "time",
    "ew",
    "chicagotribune",
    "residentadvisor",
    "stereogum",
    "musicbrainz",
}
TIER_B_SOURCES = {
    "wikipedia",
    "lastfm",
    "last",
    "albumoftheyear",
    "rateyourmusic",
    "musicboard",
    "discogs",
    "genius",
    "whosampled",
}


class KnowledgeCacheItem(BaseModel):
    key: str
    dossier: MusicDossier
    created_at: str = Field(default_factory=utc_now_iso)
    kind: str = "dossier"


def is_knowledge_intent(intent: str) -> bool:
    return intent in KNOWLEDGE_INTENTS


def is_knowledge_tool(tool: str) -> bool:
    return tool in KNOWLEDGE_TOOLS


def knowledge_deadline() -> float:
    return time.monotonic() + max(1.0, settings.knowledge_turn_budget_seconds)


def remaining_seconds(deadline_at: float | None) -> float | None:
    if not deadline_at:
        return None
    return max(0.0, deadline_at - time.monotonic())


def cache_key(entity: MusicEntity, related: list[MusicEntity] | None = None, intent: str = "") -> str:
    artist = _norm(entity.artist)
    name = _norm(entity.name)
    source = _norm(entity.source or "unknown")
    compare_suffix = ""
    if intent == "music_compare" and related:
        compare_suffix = "|" + "|".join(
            re.sub(
                r"[^a-z0-9_\-]+",
                "_",
                f"{item.type}:{_norm(item.name)}:{_norm(item.artist)}:{_norm(item.source or 'unknown')}",
            )
            for item in related
        )
    return re.sub(r"[^a-z0-9_\-:|]+", "_", f"{entity.type}:{name}:{artist}:{source}{compare_suffix}")[:220]


def read_cached_dossier(
    agent: Any,
    entity: MusicEntity,
    *,
    related: list[MusicEntity] | None = None,
    intent: str = "",
) -> MusicDossier | None:
    try:
        item = agent.store.read_model("knowledge_cache", cache_key(entity, related, intent), KnowledgeCacheItem)
    except Exception:
        return None
    if item is None:
        return None
    created = _parse_iso(item.created_at)
    if created and datetime.now(UTC) - created <= timedelta(hours=24):
        return item.dossier
    return None


def write_cached_dossier(agent: Any, dossier: MusicDossier, *, intent: str = "") -> None:
    if dossier.partial:
        return
    try:
        related = dossier.related_entities[:1] if intent == "music_compare" else []
        key = cache_key(dossier.entity, related, intent)
        # library_matches 是 per-user 的，缓存按实体共享、不存用户维度——置空再写，
        # 命中时由 build_dossier 用当前 user_id 重算，避免跨用户串库。
        cacheable = dossier.model_copy(update={"library_matches": []})
        agent.store.write_model("knowledge_cache", key, KnowledgeCacheItem(key=key, dossier=cacheable))
    except Exception:
        return


def resolve_music_entities(query: str, intent: str, plan: dict[str, Any] | None = None) -> list[MusicEntity]:
    query = (query or "").strip()
    plan = plan or {}
    if intent == "music_compare":
        names = _compare_names(query)
        if not names:
            retrieval = plan.get("retrieval_plan") or {}
            planned_entities = [str(e).strip() for e in (retrieval.get("entities") or []) if str(e).strip()]
            names = planned_entities[:2]
        if names:
            entity_type = _infer_entity_type(query, intent)
            return [
                MusicEntity(type=entity_type, name=name, artist=_infer_artist(query, name, entity_type), source="query")
                for name in names
                if name
            ][:2]
    structured = _structured_entity_from_query(query, intent)
    if structured:
        return [structured]
    entity_type = _infer_entity_type(query, intent)
    explicit = _explicit_artist_entity_from_query(query, entity_type, intent)
    if explicit:
        return [explicit]
    retrieval = plan.get("retrieval_plan") or {}
    planned_entities = [str(e).strip() for e in (retrieval.get("entities") or []) if str(e).strip()]
    names = _compare_names(query) if intent == "music_compare" else []
    if not names and planned_entities:
        names = planned_entities[: 2 if intent == "music_compare" else 1]
    if not names:
        names = [_guess_entity_name(query)]
    return [
        MusicEntity(type=entity_type, name=name, artist=_infer_artist(query, name, entity_type), source="query")
        for name in names
        if name
    ][: 2 if intent == "music_compare" else 1]


# 实体名噪声剥离（卡片 + 自然语言共用）：去开口动词前缀 + 去引用短语尾巴，
# 治「介绍X 这个专辑」「讲一下Y」被解析成 name='介绍X'/artist='这个专辑'。
_ENTITY_NOISE_LEAD = re.compile(
    r"^\s*(介绍一下|介绍下|介绍|讲一下|讲一讲|讲讲|聊聊看|聊聊|聊一聊|说说|谈谈|"
    r"分析一下|分析|了解|科普|解读|为什么经典|为什么这么经典|为什么|评价如何|评价|"
    r"乐评怎么说|请|帮我|搜索|查一下|查下|"
    r"我指的是|我说的是|我是说|我是指|指的是|我是想问|我想问的是|我想问)\s*",
    re.I,
)
_ENTITY_NOISE_TAIL = re.compile(
    r"\s*(这个专辑|这张专辑|这张唱片|这张|这个|这首歌|这首|专辑|唱片|大碟)\s*$",
    re.I,
)
_ENTITY_NOISE_WHOLE = re.compile(
    r"^\s*(这个专辑|这张专辑|这张唱片|这张|这个|这首歌|这首|专辑|唱片|大碟|album|release)\s*$",
    re.I,
)


def _strip_entity_noise(text: str) -> str:
    text = _ENTITY_NOISE_LEAD.sub("", text or "")
    text = _ENTITY_NOISE_TAIL.sub("", text)
    return text.strip(" ？?，,。.:：")


def _is_noise_only(text: str) -> bool:
    return bool(_ENTITY_NOISE_WHOLE.match((text or "").strip()))


def _structured_entity_from_query(query: str, intent: str) -> MusicEntity | None:
    """Parse compact field-style UI input such as:

    album
    Blonde on Blonde
    West Norwood Cassette Library

    The knowledge agent must preserve the provided artist for disambiguation;
    otherwise same-title albums can be canonicalized to a famous but wrong work.
    会用 _strip_entity_noise 去掉开口动词（介绍/讲一下…）和引用短语（这个专辑/这张…），
    否则 UI 拼出的卡片会把「介绍X 这个专辑」解析成 name='介绍X'、artist='这个专辑'。
    """
    lines = [re.sub(r"^[\s:：\-•]+|[\s:：]+$", "", line).strip() for line in (query or "").splitlines() if line.strip()]
    if len(lines) < 2:
        return None

    first = lines[0].lower().strip()
    kind_aliases = {
        "album": "album",
        "专辑": "album",
        "唱片": "album",
        "release": "album",
        "track": "track",
        "song": "track",
        "歌曲": "track",
        "单曲": "track",
        "artist": "artist",
        "艺人": "artist",
        "歌手": "artist",
        "乐队": "artist",
    }
    entity_type = kind_aliases.get(first)
    if entity_type is None and len(lines) == 2 and intent in {"album_deep_dive", "review_summary", "sample_lookup"}:
        inferred = "track" if intent == "sample_lookup" else "album"
        name = _strip_entity_noise(re.sub(r"^(?:title|name|标题|名称)\s*[:：]\s*", "", lines[0], flags=re.I).strip())
        artist = _strip_entity_noise(re.sub(r"^(?:artist|艺人|歌手|乐队)\s*[:：]\s*", "", lines[1], flags=re.I).strip())
        if name and artist and name.lower() != artist.lower() and not _is_noise_only(artist):
            return MusicEntity(
                type=inferred,
                name=_canonical_music_name(name),
                artist=artist,
                source="query",
            )
    if entity_type is None:
        return None

    name = _canonical_music_name(_strip_entity_noise(lines[1]))
    artist = ""
    if entity_type in {"album", "track"} and len(lines) >= 3:
        candidate = _strip_entity_noise(
            re.sub(r"^(?:artist|艺人|歌手|乐队)\s*[:：]\s*", "", lines[2], flags=re.I).strip()
        )
        if candidate and not _is_noise_only(candidate):
            artist = candidate
    # If a UI includes field labels on following lines, strip the most common ones.
    name = re.sub(r"^(?:title|name|标题|名称)\s*[:：]\s*", "", name, flags=re.I).strip()
    if not name:
        return None
    return MusicEntity(type=entity_type, name=name, artist=artist, source="query")


def _explicit_artist_entity_from_query(query: str, entity_type: str, intent: str) -> MusicEntity | None:
    if intent == "music_compare" or entity_type not in {"album", "track"}:
        return None
    text = re.sub(r"[《》“”\"']", " ", query or "")
    text = re.sub(
        r"(讲一下|讲一讲|讲讲|聊聊|聊一聊|说说|谈谈|介绍一下|介绍|分析一下|分析|了解|科普|解读|为什么经典|为什么|乐评怎么说|评价如何|评价|请|帮我|搜索|查一下|这张专辑|这首歌|"
        r"我指的是|我说的是|我是说|我是指|指的是|我是想问|我想问的是|我想问)",
        " ",
        text,
        flags=re.I,
    )
    text = re.sub(r"\s+", " ", text).strip(" ？?，,。.")
    patterns = [
        r"^(?P<artist>.+?)\s*(?:的)\s*(?:专辑\s*)?(?P<name>.+?)(?:\s*(?:专辑|album|乐评|评价))?$",
        r"^(?P<artist>.+?)\s*(?:-|–|—|:|：)\s*(?P<name>.+?)(?:\s*(?:专辑|album|乐评|评价))?$",
        r"^(?P<artist>.+?)\s*(?:'s|’s)\s*(?P<name>.+?)(?:\s*(?:album|review))?$",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I)
        if not match:
            continue
        artist = match.group("artist").strip(" ？?，,。.")
        name = match.group("name").strip(" ？?，,。.")
        name = re.sub(r"^(?:专辑|album)\s+", "", name, flags=re.I).strip()
        artist = re.sub(r"^(?:album|track|song|专辑|歌曲)\s+", "", artist, flags=re.I).strip()
        if artist and name and artist.lower() != name.lower():
            return MusicEntity(
                type=entity_type,
                name=_canonical_music_name(name),
                artist=artist,
                source="query",
            )
    return None


def _apply_release_hit(entity: MusicEntity, hit: dict[str, Any]) -> None:
    """把单个 MB release-group 候选回填到 entity（权威名/艺人/MBID）。"""
    if hit.get("title"):
        entity.name = hit["title"]
    if hit.get("artist") and not entity.artist:
        entity.artist = hit["artist"]
    if hit.get("mbid"):
        entity.external_ids.setdefault("musicbrainz", hit["mbid"])
    if not entity.query_origin:
        entity.query_origin = "musicbrainz"


def _canonicalize_release_entity(client: Any, entity: MusicEntity) -> None:
    """album/track 消歧：基于 MB 候选判定 resolved/ambiguous/unresolved。

    歧义判定聚焦真正的 bug 场景——**精确同名 + 不同艺人 + 用户未给消歧艺人**：
    例：裸查「Blonde」时 MB 同时返回 Frank Ocean《Blonde》与另一艺人《Blonde》，
    此时无法可靠选择，标记 ambiguous，交由 build_dossier 返回消歧提示而非硬编一个。
    用户已给艺人、或只有一个精确同名作品时，正常 resolved。
    """
    hits = client.search_release_group(entity.name, entity.artist, limit=10)
    if not hits:
        entity.ambiguity = "unresolved"
        return
    title_key = _match_norm(entity.name)
    exact_title = [h for h in hits if _match_norm(h.get("title", "")) == title_key]
    artist_key = _match_norm(entity.artist)

    if entity.artist:
        with_artist = [
            h
            for h in exact_title
            if artist_key
            and (
                artist_key in _match_norm(h.get("artist", "")) or _match_norm(h.get("artist", "")).endswith(artist_key)
            )
        ]
        if with_artist:
            best = max(with_artist, key=lambda h: h.get("score", 0))
            _apply_release_hit(entity, best)
            entity.ambiguity = "resolved"
            entity.confidence = min(1.0, best.get("score", 0) / 100.0)
            entity.candidates = hits[:5]
            return
        # 给了艺人但没有「精确标题+该艺人」命中：回落精确标题里 score 最高，置信降档。
        if exact_title:
            best = max(exact_title, key=lambda h: h.get("score", 0))
            _apply_release_hit(entity, best)
            entity.ambiguity = "resolved" if best.get("score", 0) >= 60 else "unresolved"
            entity.confidence = min(1.0, best.get("score", 0) / 100.0)
            entity.candidates = hits[:5]
            return

    # 未给艺人：精确同名候选里出现 ≥2 个不同艺人 → 歧义过大
    if exact_title:
        distinct_artists = {_match_norm(h.get("artist", "")) for h in exact_title if h.get("artist")}
        best = max(exact_title, key=lambda h: h.get("score", 0))
        _apply_release_hit(entity, best)
        entity.candidates = exact_title[:5]
        if len(distinct_artists) >= 2:
            entity.ambiguity = "ambiguous"
            entity.confidence = min(0.5, best.get("score", 0) / 100.0)
        else:
            entity.ambiguity = "resolved"
            entity.confidence = min(1.0, best.get("score", 0) / 100.0)
        return

    # 仅模糊命中（标题不完全相同）：按「与查询标题的贴近度 + score」选最佳，避免裸标题
    # 被带偏到更长的同名衍生作（裸查「Blonde」不该落到「Blonde on Blonde」）。保守 unresolved，
    # 但仍回填最佳、不拒答。贴近度：精确同名 > 标题长度最接近 > score 最高。
    def _closeness(h: dict) -> tuple:
        norm = _match_norm(h.get("title", ""))
        if norm == title_key:
            return (0, 0, -int(h.get("score", 0)))
        return (1, abs(len(norm) - len(title_key)), -int(h.get("score", 0)))

    best = min(hits, key=_closeness)
    _apply_release_hit(entity, best)
    entity.ambiguity = "unresolved"
    entity.confidence = min(0.5, best.get("score", 0) / 100.0)
    entity.candidates = hits[:5]


def _canonicalize_artist_entity(client: Any, entity: MusicEntity) -> None:
    """artist 消歧：同名艺人较少见，有精确名命中即视为 resolved。"""
    hits = client.search_artist(entity.name, limit=5)
    if not hits:
        entity.ambiguity = "unresolved"
        return
    name_key = _match_norm(entity.name)
    exact = [h for h in hits if _match_norm(h.get("name", "")) == name_key]
    pool = exact or hits
    best = max(pool, key=lambda h: h.get("score", 0))
    if best.get("name"):
        entity.name = best["name"]
        if not entity.query_origin:
            entity.query_origin = "musicbrainz"
    if best.get("mbid"):
        entity.external_ids.setdefault("musicbrainz", best["mbid"])
    entity.candidates = hits[:5]
    entity.confidence = min(1.0, best.get("score", 0) / 100.0)
    entity.ambiguity = "resolved" if best.get("score", 0) >= 60 else "unresolved"


def canonicalize_entities(entities: list[MusicEntity], deadline_at: float | None = None) -> list[MusicEntity]:
    """消歧阶段：用 MusicBrainz 把裸名/裸标题钉成权威 (name, artist, type)，并给出歧义状态。

    这是知识链路的「消歧」职责所在——在 resolve 阶段一次性钉准实体并判定歧义，下游
    metadata/review 全部继承，避免各源(MB/Spotify/Discogs)各自对裸标题模糊匹配出三个
    不同的同名作品（Blonde 实测被解析成 Frank Ocean / Bob Dylan《Blonde on Blonde》/
    West Norwood 三个）。album 类型且缺 artist 时收益最大。

    失败/超时/关闭 MB 时原样返回（ambiguity 保持默认 unresolved），绝不报错——
    保持 offline 测试与降级契约：默认 unresolved 即「未跑消歧、维持旧行为」。
    """
    if not getattr(settings, "enable_musicbrainz", True):
        return entities
    remaining = remaining_seconds(deadline_at)
    if remaining is not None and remaining < 1.0:
        return entities
    try:
        from app.sources.musicbrainz_client import MusicBrainzClient

        client = MusicBrainzClient()
        for entity in entities:
            if entity.type == "artist":
                _canonicalize_artist_entity(client, entity)
            elif entity.type in {"album", "track"}:
                _canonicalize_release_entity(client, entity)
    except Exception:
        logger.debug("canonicalize_entities 失败，按原实体降级", exc_info=True)
    return entities


def _match_norm(value: str) -> str:
    """实体名「包含」比对用的归一化：小写 + 仅保留字母数字与 CJK，去掉一切标点空白。

    与 MusicBrainz 客户端的 _norm 同构，确保 canonicalize 阶段与 evidence 校验阶段
    用同一套归一化口径判定「标题/艺人是否实质出现」。
    """
    return re.sub(r"[^a-z0-9一-鿿]+", "", (value or "").lower())


# ── 证据归属校验（Phase 0 止血核心）────────────────────────────────────────────
# 治同名实体错配：乐评/曲目/资料常把多个同名作品混进同一次检索结果。下面这套打分在
# 合成 dossier 前把「明显归属错误」的条目剔除，并在证据互相冲突时阻止合成完整答案。
_PROSE_CITATION_KINDS = {"review", "encyclopedia", "user_comment"}


def citation_entity_score(citation: MusicCitation, entity: MusicEntity) -> float:
    """单条 citation 对 canonical entity 的归属得分（0~1）。

    关键设计：**结构化源**（metadata/platform：MB/Spotify/Discogs/网易云）是按该实体
    检索来的，按构造即归属，默认高分保留；**散文类源**（review/encyclopedia/user_comment）
    才是同名混拼的高发地，靠文本匹配判定——album/track 已知艺人时，艺人是否出现是消歧
    关键（只命中裸标题 'blonde' 区分不了 Frank Ocean 与 Bob Dylan，给弱分）。
    """
    if citation.kind in {"metadata", "platform"}:
        return 0.8
    title_text = " ".join([citation.title or "", citation.excerpt or ""])
    text = _match_norm(title_text)
    name_key = _match_norm(entity.name)
    name_hit = bool(name_key) and name_key in text
    artist = (entity.artist or "").strip()
    artist_key = _match_norm(artist)
    artist_hit = bool(artist_key) and artist_key in text
    if entity.type in {"album", "track"}:
        if artist:
            if artist_hit and name_hit:
                return 1.0
            if artist_hit:
                return 0.6
            if name_hit:
                # 标题足够特异（长/多词，如 My Beautiful Dark Twisted Fantasy）时，只命中标题
                # 也是强归属——这类标题同名概率极低；短标题（Blonde）才保守给弱分防同名异作品。
                return 0.6 if len(name_key) >= 12 else 0.2
            return 0.0
        return 0.6 if name_hit else 0.0
    # artist 类型实体：拉丁艺人名要避免 Drake 命中 Nick Drake 这类子串误伤。
    if _is_latin_name(entity.name):
        return 1.0 if _latin_name_in_text(entity.name, title_text) else 0.0
    return 1.0 if (artist_hit or name_hit) else 0.0


def _is_latin_name(name: str) -> bool:
    text = (name or "").strip()
    return bool(text) and bool(re.fullmatch(r"[A-Za-z0-9 .&'_-]+", text))


def _latin_name_in_text(name: str, text: str) -> bool:
    escaped = re.escape((name or "").strip())
    if not escaped:
        return False
    raw = text or ""
    for match in re.finditer(rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])", raw, flags=re.I):
        prefix = raw[: match.start()].rstrip()
        prev = re.search(r"([A-Za-z0-9']+)$", prefix)
        prev_token = (prev.group(1) if prev else "").lower()
        if prev_token and prev_token not in {
            "review",
            "reviews",
            "album",
            "albums",
            "artist",
            "artists",
            "biography",
            "discography",
            "guide",
            "feature",
            "essay",
            "the",
            "a",
            "an",
            "of",
            "for",
            "by",
            "vs",
            "and",
        }:
            continue
        return True
    return False


def _track_matches_artist(track: TrackRef, artist: str) -> str | bool:
    """曲目是否归属该艺人（同名专辑防混入别家曲目）。曲目艺人未知时保守保留。"""
    track_artist = _match_norm(getattr(track, "artist", "") or "")
    if not track_artist:
        return True
    target = _match_norm(artist)
    return bool(target) and (target in track_artist or track_artist in target)


def validate_evidence_consistency(
    entity: MusicEntity,
    metadata_citations: list[MusicCitation],
    review_citations: list[MusicCitation],
    tracks: list[TrackRef],
) -> EvidenceConsistencyReport:
    """合成前一致性校验：剔除明显归属错误的 citation/曲目，并报告证据是否仍可靠。

    返回 kept_citations/kept_tracks（已过滤）与 ok/problems/confidence。
    ok=False 表示已知艺人却没有任何资料命中艺人（全部偏题/疑似同名异作品），上层应抑制
    完整 summary 以防把错误实体的资料拼成答案。
    """
    all_citations = [*metadata_citations, *review_citations]
    kept: list[MusicCitation] = []
    dropped = 0
    for citation in all_citations:
        score = citation_entity_score(citation, entity)
        if score <= 0.0:
            dropped += 1
            continue
        # 已知艺人的 album/track：散文类来源必须提到目标艺人才保留，否则大概率是同名异作品
        # （例：查 Frank Ocean《Blonde》却抓到 Bob Dylan《Blonde on Blonde》的乐评）。
        if (
            entity.type in {"album", "track"}
            and entity.artist
            and citation.kind in _PROSE_CITATION_KINDS
            and score < 0.5
        ):
            dropped += 1
            continue
        kept.append(citation)

    problems: list[str] = []
    on_target = [c for c in kept if citation_entity_score(c, entity) >= 0.5]

    kept_tracks = tracks
    artist = (entity.artist or "").strip()
    if entity.type in {"album", "track"} and artist:
        kept_tracks = [t for t in tracks if _track_matches_artist(t, artist)]
        dropped_tracks = len(tracks) - len(kept_tracks)
        if dropped_tracks:
            problems.append(f"剔除 {dropped_tracks} 首疑似归属其他艺人的曲目")
    if dropped:
        problems.append(f"剔除 {dropped} 条与目标实体无关的资料")

    confidence = (len(on_target) / len(all_citations)) if all_citations else 0.0
    ok = True
    if all_citations and not on_target:
        problems.append("所有资料都未明确指向目标实体")
        ok = False
    if (
        entity.type in {"album", "track"}
        and artist
        and all_citations
        and not any(citation_entity_score(c, entity) >= 0.5 for c in kept)
    ):
        ok = False
    return EvidenceConsistencyReport(
        ok=ok,
        problems=problems,
        confidence=confidence,
        kept_citations=kept,
        kept_tracks=kept_tracks,
    )


def validate_compare_evidence_consistency(
    entities: list[MusicEntity],
    metadata_citations: list[MusicCitation],
    review_citations: list[MusicCitation],
    tracks: list[TrackRef],
) -> EvidenceConsistencyReport:
    """Compare 模式的一致性校验：允许资料命中任一比较对象，避免第二个实体被误杀。"""
    pair = entities[:2]
    all_citations = [*metadata_citations, *review_citations]
    kept: list[MusicCitation] = []
    dropped = 0
    for citation in all_citations:
        score = max((citation_entity_score(citation, entity) for entity in pair), default=0.0)
        if score <= 0.0:
            dropped += 1
            continue
        kept.append(citation)

    problems: list[str] = []
    hits: dict[str, int] = {entity.name: 0 for entity in pair}
    for citation in kept:
        for entity in pair:
            if citation_entity_score(citation, entity) >= 0.5:
                hits[entity.name] += 1

    if dropped:
        problems.append(f"剔除 {dropped} 条与比较对象无关的资料")

    missing = [name for name, count in hits.items() if count <= 0]
    if len(missing) == len(pair) and all_citations:
        problems.append("所有资料都未明确指向这两个比较对象")
    elif missing:
        problems.append("未拿到明确指向 " + " / ".join(missing) + " 的资料")

    confidence = (sum(1 for count in hits.values() if count > 0) / len(pair)) if pair else 0.0
    return EvidenceConsistencyReport(
        ok=not (all_citations and len(missing) == len(pair)),
        problems=problems,
        confidence=confidence,
        kept_citations=kept,
        kept_tracks=tracks,
    )


def _ambiguous_summary(entity: MusicEntity) -> str:
    """同名歧义时的消歧提示：列出候选，请用户补艺人名，绝不凭猜测拼完整答案。"""
    names: list[str] = []
    for cand in (entity.candidates or [])[:4]:
        title = cand.get("title") or cand.get("name") or ""
        cand_artist = cand.get("artist") or ""
        if title:
            names.append(f"《{title}》" + (f" - {cand_artist}" if cand_artist else ""))
    listing = "；".join(names) if names else "（未能取到候选列表）"
    return (
        f"「{entity.name}」存在多个同名作品，我不想凭猜测拼出一份答案。"
        f"可能的指代：{listing}。请补上艺人名（例如「{entity.name} 是谁的」），"
        f"我再给你完整、可靠的解读。"
    )


_YEAR_RE = re.compile(r"\b(19[5-9]\d|20[0-3]\d)\b")


def _extract_years(text: str) -> list[int]:
    """从资料文本里抽 4 位年份（1950–2039），用于推断职业生涯时间跨度。"""
    return [int(m) for m in _YEAR_RE.findall(text or "")]


def _build_career_timeline(
    entity: MusicEntity,
    albums: list[dict[str, Any]],
    tracks: list[TrackRef],
    meta_text: str,
    style_tags: list[str],
) -> list[CareerPhase]:
    """为 artist_deep_dive 构建职业生涯脉络（确定性、不臆造）。

    数据现实：网易云歌手专辑接口不返回发行年份，精确到每张专辑的分期来源不足。因此只产出
    「可追溯证据支持」的阶段——bio 里若出现 ≥2 个不同年份，给出时间跨度；始终基于真实专辑/
    曲目给「代表作品」+「入门路线」。绝不编造年份或风格演变（防幻觉铁律）。
    """
    album_names: list[str] = []
    for album in albums:
        name = str(album.get("name") or album.get("title") or "").strip()
        if name and name not in album_names:
            album_names.append(name)

    phases: list[CareerPhase] = []
    years = sorted({y for y in _extract_years(meta_text) if 1950 <= y <= 2039})
    has_span = len(years) >= 2 and years[-1] > years[0]
    if has_span:
        phases.append(
            CareerPhase(
                period=f"{years[0]}–{years[-1]}",
                phase_name="时间跨度",
                sound_change="；".join(style_tags[:2]) if style_tags else "",
                career_context=(
                    f"可追溯资料的时间跨度约 {years[0]}–{years[-1]} 年。受限于来源，"
                    f"无法精确把每张专辑钉到具体阶段，下面按代表作品组织。"
                ),
            )
        )
    # 始终给出「代表作品」阶段（仅基于真实拥有的专辑；不足时回落曲目）
    releases = album_names or [t.title for t in tracks if t.title]
    phases.append(
        CareerPhase(
            period="代表作品",
            phase_name="代表作品",
            key_releases=releases[:6],
            career_context=(
                "来源未提供明确发行年份，按可追溯的代表作品组织，不臆造分期。"
                if not has_span
                else "职业生涯中可追溯的代表专辑。"
            ),
        )
    )
    return phases


_CAREER_YEAR_RE = re.compile(r"(19[5-9]\d|20[0-3]\d)")  # 1950–2039
_CAREER_TITLE_RE = re.compile(r"《([^》]{1,40})》")


def _clean_career_context(sentence: str) -> str:
    """把含年份的句子轻清洗成短上下文：去年份、《》去括号、压空白、截 60 字。"""
    ctx = _CAREER_YEAR_RE.sub("", sentence)
    ctx = _CAREER_TITLE_RE.sub(lambda m: m.group(1), ctx)  # 《X》→X
    ctx = ctx.replace("年", " ")
    ctx = re.sub(r"\s+", " ", ctx).strip(" ，,、。：:·-—")
    return ctx[:60]


def _extract_career_phases_from_text(text: str) -> list[CareerPhase]:
    """从 DeepSeek 直答正文里抽取「年份→专辑/代表作」的职业时间线。

    直答正文通常自带「2015年《Beauty Behind the Madness》……」这类年份+作品的时间脉络。实体解析
    空（无专辑年表）时，用它替换 _build_career_timeline 的无年份空壳阶段，让 artist_deep_dive 仍有
    可读时间线，而不是「来源未提供明确发行年份」。确定性解析、不臆造：只取正文真实出现的年份+《》
    作品，句内按出现位置左→右把作品归给最近的年份（正确处理「2020年《A》与2022年《B》」）。
    没抽到任何「年份+作品」的正文返回 []，由调用方保留原 career_phases。
    """
    by_year: dict[int, dict[str, Any]] = {}
    for sentence in re.split(r"[。！？\n]+", text or ""):
        sentence = sentence.strip()
        if not sentence:
            continue
        tokens: list[tuple[int, str, str]] = []
        for m in _CAREER_YEAR_RE.finditer(sentence):
            tokens.append((m.start(), "year", m.group(1)))
        for m in _CAREER_TITLE_RE.finditer(sentence):
            tokens.append((m.start(), "title", m.group(1).strip()))
        if not any(t[1] == "year" for t in tokens):
            continue  # 该句没年份：不产阶段（避免把听法段曲目误当 era）
        tokens.sort(key=lambda t: t[0])
        cur_year: int | None = None
        for _, kind, value in tokens:
            if kind == "year":
                cur_year = int(value)
            elif kind == "title" and cur_year is not None:
                bucket = by_year.setdefault(cur_year, {"releases": [], "context": ""})
                if value and value not in bucket["releases"]:
                    bucket["releases"].append(value)
                if not bucket["context"]:
                    bucket["context"] = _clean_career_context(sentence)
    phases = [
        CareerPhase(
            period=str(year),
            phase_name=str(year),
            key_releases=by_year[year]["releases"][:5],
            career_context=by_year[year]["context"],
        )
        for year in sorted(by_year)
        if by_year[year]["releases"]
    ]
    return phases[:8]


def lookup_metadata(agent: Any, entities: list[MusicEntity], deadline_at: float | None = None) -> dict[str, Any]:
    remaining = remaining_seconds(deadline_at)
    if remaining is not None and remaining < 1.0:
        return {
            "metadata": [],
            "citations": [],
            "tracks": [],
            "albums": [],
            "skipped_due_to_deadline": ["music_metadata_lookup"],
        }
    # 元数据现在是单波并行（MB/Spotify/Discogs/netease/web 同时发），墙钟≈最慢单源，
    # 不再是各源之和——所以这一波该拿到比旧串行模型(3s)更宽的预算，否则慢源(Spotify
    # OAuth/Discogs/web)永远跑不完，只剩最快的 netease 活下来(实测"来源 netease only")。
    # 取 source_timeout 与「剩余预算留给乐评的余量」中较大者，但不超过剩余预算。
    floor = settings.knowledge_source_timeout_seconds
    if remaining is not None:
        # 给后续 review_search 留 review_timeout 余量；但元数据是 bonus 内容，封顶 metadata_timeout
        # （旧实现会用 ~20s，难抓专辑时把 dossier/synth 挤没）。封顶略低于 review，不拖长 stage2。
        budget_for_meta = min(
            max(floor, remaining - settings.knowledge_review_timeout_seconds - 0.5),
            settings.knowledge_metadata_timeout_seconds,
        )
        timeout = min(budget_for_meta, remaining - 0.3)
    else:
        timeout = floor
    timeout = max(0.5, timeout)
    tasks: list[tuple[str, Any]] = []
    for entity in entities:
        tasks.append((f"metadata:{entity.name}", lambda e=entity: _metadata_for_entity(agent, e, timeout=timeout)))
    batches = run_parallel(tasks, timeout=max(0.2, timeout), default={})
    metadata: list[dict[str, Any]] = []
    citations: list[MusicCitation] = []
    tracks: list[TrackRef] = []
    albums: list[dict[str, Any]] = []
    for batch in batches:
        if not isinstance(batch, dict):
            continue
        metadata.extend(batch.get("metadata") or [])
        citations.extend(batch.get("citations") or [])
        tracks.extend(batch.get("tracks") or [])
        albums.extend(batch.get("albums") or [])
    return {
        "metadata": metadata,
        "citations": _limit_citations(citations),
        "tracks": tracks[:12],
        "albums": albums[:8],
    }


def _review_queries_for_entity(entity: MusicEntity) -> list[str]:
    base = " ".join(part for part in [entity.artist, entity.name] if part).strip() or entity.name
    if entity.type == "artist":
        return [
            f"{base} AllMusic biography",
            f"{base} artist profile review",
            f"{base} Pitchfork review",
            f"{base} style influence interview",
            f"{base} critical reception",
        ]
    return [
        f"{base} Pitchfork review",
        f"{base} AllMusic review",
        f"{base} Guardian review",
        f"{base} critical reception",
        f"{base} 乐评 专辑 评价",
    ]


def search_reviews(
    entities: list[MusicEntity],
    deadline_at: float | None = None,
    *,
    intent: str = "",
    query: str = "",
) -> dict[str, Any]:
    remaining = remaining_seconds(deadline_at)
    if remaining is not None and remaining < 1.0:
        return {"citations": [], "opinions": [], "skipped_due_to_deadline": ["review_search"]}
    timeout = min(settings.knowledge_review_timeout_seconds, remaining or settings.knowledge_review_timeout_seconds)
    queries: list[str] = []
    if intent == "music_compare" and len(entities) >= 2:
        left, right = entities[:2]
        pair = f"{left.name} {right.name}".strip()
        queries.extend(
            [
                f"{pair} comparison style",
                _review_queries_for_entity(left)[0],
                _review_queries_for_entity(right)[0],
            ]
        )
    else:
        for entity in entities:
            queries.extend(_review_queries_for_entity(entity))
    if query.strip():
        queries.append(query.strip())
    deduped = []
    seen: set[str] = set()
    for q in queries:
        key = q.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(q)
    queries = deduped[: settings.knowledge_max_search_queries]
    # Leave a small margin for parsing/ToolRuntime bookkeeping.  Otherwise the
    # outer wait_for may cancel the whole handler at exactly the same wall-clock
    # boundary and report a misleading "skipped due to deadline" even though the
    # search was actually attempted.
    batch_timeout = max(0.5, timeout - 0.3)
    request_timeout = max(0.5, batch_timeout)
    tasks = [
        (
            f"review:{q}",
            lambda q=q: web_search_source.search_web_info(
                q,
                max_results=max(2, settings.knowledge_max_review_sources),
                api_key=settings.tavily_api_key,
                timeout=request_timeout,
            ),
        )
        for q in queries
    ]
    batches = run_parallel(tasks, timeout=batch_timeout, default=[])
    citations: list[MusicCitation] = []
    for batch in batches:
        if not isinstance(batch, list):
            continue
        for item in batch:
            title = (item.get("title") or "").strip()
            url = (item.get("url") or "").strip()
            excerpt = (item.get("content") or "").strip()
            if not title and not url:
                continue
            source = _source_from_url(url) or "web"
            citations.append(
                MusicCitation(
                    source=source,
                    title=title,
                    url=url,
                    kind="review",
                    excerpt=excerpt[:500],
                    confidence=_source_confidence(source, url),
                )
            )
    citations = sorted(
        _dedupe_citations(citations),
        key=lambda item: item.confidence,
        reverse=True,
    )[: settings.knowledge_max_review_sources]
    opinions = [
        ReviewOpinion(
            source=c.source,
            sentiment=_sentiment_from_text(c.excerpt),
            aspects=_aspects_from_text(c.excerpt),
            summary=c.excerpt[:180],
            citation_id=i,
        )
        for i, c in enumerate(citations)
    ]
    return {"citations": citations, "opinions": opinions}


def search_sample_relations(
    entities: list[MusicEntity], query: str, deadline_at: float | None = None
) -> dict[str, Any]:
    target = _target_track_from_entities(entities, query)
    remaining = remaining_seconds(deadline_at)
    if remaining is not None and remaining < 1.0:
        return {
            "target": target.model_dump(mode="json"),
            "evidence": [],
            "skipped_due_to_deadline": ["sample_relation_search"],
        }
    timeout = min(settings.knowledge_review_timeout_seconds, remaining or settings.knowledge_review_timeout_seconds)
    base = " ".join(part for part in [target.artist, target.title] if part).strip() or target.title
    queries = [
        f"{base} WhoSampled",
        f"{base} sampled what song",
        f"{base} Genius sample interpolation",
        f"{base} Discogs sample credits",
    ][: settings.knowledge_max_search_queries]
    tasks = [
        (
            f"sample:{q}",
            lambda q=q: web_search_source.search_web_info(
                q,
                max_results=3,
                api_key=settings.tavily_api_key,
            ),
        )
        for q in queries
    ]
    batches = run_parallel(tasks, timeout=max(0.2, timeout), default=[])
    evidence: list[SampleEvidence] = []
    for batch in batches:
        if not isinstance(batch, list):
            continue
        for item in batch:
            title = (item.get("title") or "").strip()
            url = (item.get("url") or "").strip()
            excerpt = (item.get("content") or "").strip()
            if not title and not url:
                continue
            source = _source_from_url(url) or "web"
            evidence.append(
                SampleEvidence(
                    source=source,
                    title=title,
                    url=url,
                    excerpt=excerpt[:500],
                    confidence=_sample_source_confidence(source, url),
                    source_tier=_source_tier(source, url),
                )
            )
    evidence = _dedupe_sample_evidence([*_canonical_sample_evidence(target), *evidence])[:6]
    return {"target": target.model_dump(mode="json"), "evidence": [e.model_dump(mode="json") for e in evidence]}


def locate_sample_sources(
    agent: Any, target: TrackRef, evidence: list[SampleEvidence], deadline_at: float | None = None
) -> dict[str, Any]:
    remaining = remaining_seconds(deadline_at)
    if remaining is not None and remaining < 1.0:
        return {"relations": [], "source_cards": [], "skipped_due_to_deadline": ["locate_sample_sources"]}
    relations = _relations_from_evidence(target, evidence)
    cards: list[dict[str, Any]] = []
    for relation in relations[:4]:
        q = " ".join(part for part in [relation.source_track.title, relation.source_track.artist] if part).strip()
        found = []
        if q:
            try:
                found = agent.search_web_music(q, top_k=3, relevance_query=q)
            except Exception:
                found = []
        if found:
            from app.answer import song_card

            cards.append(song_card(found[0]))
        else:
            cards.append(
                {
                    "title": relation.source_track.title,
                    "artist": relation.source_track.artist,
                    "source": relation.source_track.source or "sample_source",
                    "source_id": relation.source_track.source_id,
                }
            )
    return {
        "relations": [r.model_dump(mode="json") for r in relations],
        "source_cards": cards,
    }


def build_sample_dossier(
    target: TrackRef,
    evidence: list[SampleEvidence],
    relations: list[SampleRelation],
    source_cards: list[dict[str, Any]],
    skipped: list[str] | None = None,
) -> SampleDossier:
    skipped = skipped or []
    partial_reasons: list[str] = []
    if skipped:
        partial_reasons.append("部分采样工具因时间预算不足被跳过：" + "、".join(sorted(set(skipped))))
    if not evidence:
        partial_reasons.append("没有找到可核实采样关系来源")
    if evidence and not relations:
        partial_reasons.append("找到相关资料，但没有解析出明确源曲")
    dossier = SampleDossier(
        target=target,
        relations=relations,
        source_track_cards=source_cards,
        citations=evidence[:6],
        partial=bool(partial_reasons),
        degraded_reason="；".join(partial_reasons) if partial_reasons else None,
    )
    return dossier


def sample_dossier_answer(dossier: SampleDossier) -> str:
    target = f"《{dossier.target.title}》" if dossier.target.title else "这首歌"
    if not dossier.relations:
        reason = dossier.degraded_reason or "没有找到可核实采样关系来源"
        return f"{target}：{reason}，我不会硬编源曲。"
    lines = [f"{target} 的采样/源曲线索："]
    if dossier.partial and dossier.degraded_reason:
        lines.append(f"资料状态：{dossier.degraded_reason}。")
    for idx, rel in enumerate(dossier.relations, start=1):
        source = f"《{rel.source_track.title}》"
        if rel.source_track.artist:
            source += f" - {rel.source_track.artist}"
        rel_label = {
            "sample": "采样",
            "interpolation": "插值",
            "cover": "翻唱",
            "remix": "混音/再创作",
            "reference": "引用",
            "unknown": "疑似关联",
        }.get(rel.relation_type, rel.relation_type)
        lines.append(f"{idx}. {rel_label}：{source}（置信度 {rel.confidence:.2f}）")
        if rel.note:
            lines.append(f"   - {rel.note}")
    if dossier.citations:
        lines.append("参考来源：")
        for ev in dossier.citations[:3]:
            label = ev.title or ev.source
            lines.append(f"- {label}：{ev.url}" if ev.url else f"- {label}")
    return "\n".join(lines)


def build_evidence_pack(
    metadata: list[dict[str, Any]],
    citations: list[MusicCitation],
    opinions: list[ReviewOpinion],
) -> KnowledgeEvidencePack:
    texts = [str(item.get("summary") or "") for item in metadata]
    texts.extend(c.excerpt for c in citations)
    joined = "\n".join(texts).lower()
    sound_terms = [
        "ambient",
        "electronic",
        "guitar-based",
        "minimal",
        "dense",
        "art rock",
        "alternative rock",
        "r&b",
        "neo-soul",
        "krautrock",
        "jazz",
        "experimental",
    ]
    theme_terms = [
        "alienation",
        "technology",
        "anxiety",
        "memory",
        "identity",
        "intimacy",
        "consumer",
        "modernity",
        "isolation",
        "dread",
        "nostalgia",
    ]
    disagreement_terms = [
        ("difficult", "难入门"),
        ("inaccessible", "不够易听"),
        ("controversial", "存在争议"),
        ("overrated", "被认为过誉"),
        ("uneven", "结构/质量不均"),
        ("fragmented", "结构碎片化"),
    ]
    source_quality = {c.source: _source_tier(c.source, c.url) for c in citations}
    critic_points = [c.excerpt[:400] for c in sorted(citations, key=lambda c: c.confidence, reverse=True) if c.excerpt][
        :10
    ]
    return KnowledgeEvidencePack(
        facts=[t[:300] for t in texts if t][:6],
        critic_points=critic_points,
        sound_descriptors=[term for term in sound_terms if term in joined],
        theme_descriptors=[term for term in theme_terms if term in joined],
        disagreements=[label for term, label in disagreement_terms if term in joined],
        source_quality=source_quality,
    )


def _source_tier(source: str, url: str = "") -> str:
    key = _source_key(source or _source_from_url(url))
    if key in TIER_A_SOURCES:
        return "A"
    if key in TIER_B_SOURCES:
        return "B"
    return "C"


def _source_confidence(source: str, url: str = "") -> float:
    tier = _source_tier(source, url)
    if tier == "A":
        return 0.82
    if tier == "B":
        return 0.68
    return 0.45 if url else 0.35


def _synthesize_dossier_prose(
    agent: Any,
    entity: MusicEntity,
    meta_text: str,
    style_tags: list[str],
    evidence_pack: KnowledgeEvidencePack,
    review_citations: list[MusicCitation],
    opinions: list[ReviewOpinion],
    deadline_at: float | None,
    web_knowledge_claims: list[str] | None = None,
    parametric: bool = False,
) -> dict[str, str] | None:
    """把零散（多为英文）证据交给 LLM 翻译+总结成连贯中文。

    防幻觉铁律：只允许基于给定证据改写/翻译/压缩，禁止补充证据外的事实。证据不足时
    宁可返回 None 让上层走机械兜底，也不硬编。严格 JSON 输出——解析失败即视为不可用，
    保证 MockLLM（离线测试）与任何非 JSON 回复都安全回落到机械摘要，输出确定。

    ``web_knowledge_claims``：强搜索 provider 产出的结构化事实（web 真来源或 DeepSeek 先验）。
    ``parametric=True`` 时这些 claim 来自模型先验、未联网核实——会如实声明，绝不冒充有据可查。

    返回 {"summary": ..., "critical_consensus": ...}，或 None（无 LLM / 无证据 / 无预算 / 解析失败）。
    """
    llm = getattr(agent, "llm", None)
    if llm is None or not hasattr(llm, "generate"):
        return None
    # 没有任何可总结的证据就不调 LLM（省延迟，也避免"无中生有"）。
    facts = [f for f in evidence_pack.facts if f.strip()]
    critic_points = [c for c in evidence_pack.critic_points if c.strip()]
    claims = [c.strip() for c in (web_knowledge_claims or []) if c and c.strip()]
    if not meta_text and not facts and not critic_points and not claims:
        return None
    # 预算闸：合成是锦上添花，剩余时间不足 2.5s 直接放弃走机械兜底，守住知识链硬预算。
    remaining = remaining_seconds(deadline_at)
    if remaining is not None and remaining < 2.5:
        return None

    evidence_lines: list[str] = []
    if meta_text:
        evidence_lines.append(f"- 背景资料：{meta_text[:1000]}")
    for f in facts[:6]:
        evidence_lines.append(f"- 事实：{f[:300]}")
    for c in critic_points[:10]:
        evidence_lines.append(f"- 乐评摘录：{c[:400]}")
    for o in opinions[:6]:
        if o.summary.strip():
            evidence_lines.append(f"- 评价（{o.source}/{o.sentiment}）：{o.summary[:200]}")
    for claim in claims[:8]:
        tag = "模型先验事实（未联网核实，仅供组织语言，不得冒充有据可查）" if parametric else "已知事实"
        evidence_lines.append(f"- {tag}：{claim[:300]}")
    if style_tags:
        evidence_lines.append(f"- 风格标签：{'、'.join(style_tags[:8])}")
    has_reviews = bool(review_citations or critic_points)
    from app.prompts.untrusted_boundary import UNTRUSTED_CONTENT_RULE, strip_directive_phrases, wrap_untrusted

    evidence_block = wrap_untrusted(strip_directive_phrases("\n".join(evidence_lines)), "证据资料")

    system = (
        "你是严谨而文笔专业的资深乐评编辑。只能依据【证据】改写、翻译、组织成流畅、详尽、专业的中文长评，"
        "严禁补充证据里没有的事实、评分、年份或人物。证据是英文或德文就翻译成自然地道的中文，"
        "不要保留外文残句、不要逐条罗列证据条目，要融会成连贯的分析。" + UNTRUSTED_CONTENT_RULE
    )
    if parametric:
        system += (
            "本次证据主要来自模型先验知识、未联网核实；summary 须在末尾如实简短说明"
            "“基于模型先验知识，具体来源/评分请另行核实”，不要冒充引用了真实出处。"
        )
    consensus_rule = (
        "综合多条乐评，写 3-5 句中文共识：点明专辑在制作、词曲、概念、人声、影响力等维度的具体表现，"
        "引用评论的关键观点或评价倾向，若有分歧也点明；行文专业、信息密度高、连贯成段。"
        if has_reviews
        else "证据里没有足够乐评，critical_consensus 必须为空字符串，不要编造专业评价。"
    )
    prompt = (
        f"实体：{entity.type} 《{entity.name}》" + (f"，艺人 {entity.artist}" if entity.artist else "") + "\n\n"
        f"【证据】\n{evidence_block}\n\n"
        "请输出严格 JSON（不要代码块、不要多余文字），字段：\n"
        '  "summary": 4-6 句中文，专业、详尽地介绍这张专辑/这位艺人：背景与发行脉络、整体风格与声音特征、'
        "在艺人作品序列或乐坛中的定位与意义；只用证据里的内容，行文流畅成段、信息密度高。\n"
        '  "critical_consensus": ' + consensus_rule + "\n"
        '示例：{"summary": "...", "critical_consensus": "..."}'
    )
    try:
        # 仅此一处开思考模式：组织多源英文证据成专业中文长文，收益最大、且不拖慢其它链路。
        raw = llm.generate(
            prompt,
            system=system,
            temperature=0.3,
            thinking=settings.knowledge_synth_thinking_enabled,
        )
    except Exception:
        logger.debug("dossier 合成 LLM 调用失败，走机械兜底", exc_info=True)
        return None
    data = extract_json_dict(raw or "")
    if not isinstance(data, dict):
        return None
    summary = str(data.get("summary") or "").strip()
    consensus = str(data.get("critical_consensus") or "").strip()
    if not summary:
        return None
    return {"summary": summary[:1000], "critical_consensus": consensus[:1000]}


def _match_library_to_entity(
    agent: Any,
    user_id: str,
    entity: MusicEntity,
    style_tags: list[str],
    *,
    limit: int = 8,
) -> list[LibraryMatch]:
    """把知识档案的实体与用户曲库交叉命中，让回答「结合你的库与口味」。

    两级命中（强→弱）：
      1) artist：库里 artist 字段（拆合作歌手后）命中该歌手/专辑艺人——精确、最强信号；
      2) genre：库里曲风与档案 style_tags 有交集——扩展命中，曲风是粗标签，排在 artist 后。
    再用用户 top 口味（曲风/歌手）标 taste_aligned，taste_aligned 的排前。

    全程容错：拿不到 agent/库/口味就返回空，绝不让知识链路因个性化失败而崩。
    """
    if agent is None or not user_id:
        return []
    try:
        from app.recommend.engine import _split_artists

        assets = [a for a in agent.list_assets() if getattr(a, "status", "") == "analyzed"]
    except Exception:
        logger.debug("library match: list_assets 失败", exc_info=True)
        return []
    if not assets:
        return []

    # 目标艺人集合：album 用其 artist，artist 实体用 name；都拆合作歌手并归一。
    target_artists: set[str] = set()
    for raw in (entity.artist, entity.name if entity.type == "artist" else ""):
        for piece in _split_artists(raw):
            if piece:
                target_artists.add(piece)
    target_styles = {_match_norm(t) for t in style_tags if t}
    # 同时纳入父类：档案标签可能是细分（"另类R&B"）而库里是一级（"R&B"），或反之；
    # 上卷到父类再比，粗细两种粒度都能命中（依赖 genres.parent_genre）。
    from app.genres import parent_genre

    target_styles |= {_match_norm(parent_genre(t)) for t in style_tags if t}

    # 用户 top 口味，用于 taste_aligned 加权（容错：拿不到就空集，不加权）。
    taste_artists: set[str] = set()
    taste_genres: set[str] = set()
    try:
        profile = agent.get_taste_profile(user_id)
        taste_artists = {_match_norm(a) for a, _ in (profile.top_artists or [])}
        taste_genres = {_match_norm(g) for g, _ in (profile.top_genres or [])}
    except Exception:
        logger.debug("library match: get_taste_profile 失败", exc_info=True)

    artist_hits: list[LibraryMatch] = []
    genre_hits: list[LibraryMatch] = []
    seen: set[str] = set()
    for asset in assets:
        title = (asset.title or "").strip()
        if not title:
            continue
        key = _match_norm(title) + "|" + _match_norm(asset.artist or "")
        if key in seen:
            continue
        asset_artists = {a for a in _split_artists(asset.artist) if a}
        # 库曲风也上卷父类，与 target_styles 同口径比较，粗细互通。
        asset_genres_norm = {_match_norm(g) for g in (asset.genre or [])}
        asset_genres_norm |= {_match_norm(parent_genre(g)) for g in (asset.genre or [])}
        is_artist_hit = bool(target_artists & asset_artists)
        is_genre_hit = bool(target_styles & asset_genres_norm) and not is_artist_hit
        if not (is_artist_hit or is_genre_hit):
            continue
        seen.add(key)
        taste_aligned = bool((asset_artists & taste_artists) or (asset_genres_norm & taste_genres))
        match = LibraryMatch(
            title=title,
            artist=asset.artist or "",
            source=asset.source or "local",
            source_id=asset.external_id or "",
            asset_id=asset.asset_id,
            cover_url=asset.cover_url or "",
            genre=list(asset.genre or []),
            relation="artist" if is_artist_hit else "genre",
            taste_aligned=taste_aligned,
        )
        (artist_hits if is_artist_hit else genre_hits).append(match)

    # 排序：artist 命中优先于 genre；同级里 taste_aligned 优先。
    artist_hits.sort(key=lambda m: not m.taste_aligned)
    genre_hits.sort(key=lambda m: not m.taste_aligned)
    return (artist_hits + genre_hits)[:limit]


def build_dossier(
    agent: Any,
    query: str,
    intent: str,
    entities: list[MusicEntity],
    metadata: list[dict[str, Any]],
    metadata_citations: list[MusicCitation],
    review_citations: list[MusicCitation],
    opinions: list[ReviewOpinion],
    tracks: list[TrackRef],
    deadline_at: float | None = None,
    skipped: list[str] | None = None,
    albums: list[dict[str, Any]] | None = None,
    timed_out: list[str] | None = None,
    web_knowledge_claims: list[str] | None = None,
    web_knowledge_provider: str = "",
    web_knowledge_answer: str = "",
    web_knowledge_style_tags: list[str] | None = None,
    user_id: str = "",
) -> MusicDossier:
    entity = (
        entities[0]
        if entities
        else MusicEntity(type=_infer_entity_type(query, intent), name=_guess_entity_name(query), source="query")
    )
    is_compare = intent == "music_compare" and len(entities) >= 2
    cached = read_cached_dossier(
        agent,
        entity,
        related=entities[1:2] if is_compare else None,
        intent=intent,
    )
    if cached and not skipped:
        # library_matches 是 per-user 的，缓存按实体+意图共享、不含用户维度——命中后
        # 必须用当前 user_id 重算，否则会把别的用户/旧用户的库命中带出来。
        refreshed_matches = (
            _match_library_to_entity(agent, user_id, cached.entity, cached.style_tags)
            if cached.entity.ambiguity != "ambiguous"
            else []
        )
        return cached.model_copy(
            update={
                "partial": False,
                "degraded_reason": None,
                "library_matches": refreshed_matches,
            }
        )

    # ── Phase 0 证据一致性校验：剔除归属错误的 citation/曲目，治同名实体资料混拼 ──
    report = (
        validate_compare_evidence_consistency(entities[:2], metadata_citations, review_citations, tracks)
        if is_compare
        else validate_evidence_consistency(entity, metadata_citations, review_citations, tracks)
    )
    citations = _limit_citations(report.kept_citations)
    tracks = report.kept_tracks
    # 合成只喂「命中目标实体」的资料，避免把同名异作品的乐评混进 LLM 总结。
    if is_compare:
        on_target = [
            c
            for c in report.kept_citations
            if any(citation_entity_score(c, compare_entity) >= 0.5 for compare_entity in entities[:2])
        ]
    else:
        on_target = [c for c in report.kept_citations if citation_entity_score(c, entity) >= 0.5]
    synth_citations = on_target or report.kept_citations

    if is_compare:
        meta_chunks: list[str] = []
        for compare_entity in entities[:2]:
            matched = next(
                (
                    str(item.get("summary") or item.get("bio") or "").strip()
                    for item in metadata
                    if _norm((item.get("entity") or {}).get("name") or "") == _norm(compare_entity.name)
                    and str(item.get("summary") or item.get("bio") or "").strip()
                ),
                "",
            )
            if matched:
                meta_chunks.append(f"{compare_entity.name}: {matched}")
        meta_text = "\n".join(meta_chunks[:2])
    else:
        meta_text = _first_nonempty(*(m.get("summary", "") for m in metadata), *(m.get("bio", "") for m in metadata))
    evidence_pack = build_evidence_pack(metadata, synth_citations, opinions)
    style_tags = _merge_unique(
        [*(web_knowledge_style_tags or []), *_tags_from_metadata(metadata), *evidence_pack.sound_descriptors]
    )[:8]
    partial_reasons: list[str] = []
    skipped = skipped or []
    timed_out = timed_out or []
    if skipped:
        partial_reasons.append("部分知识工具因时间预算不足被跳过：" + "、".join(sorted(set(skipped))))
    if timed_out:
        partial_reasons.append("部分知识工具在本轮时间内未返回，已超时降级：" + "、".join(sorted(set(timed_out))))
    if not report.kept_citations and (metadata_citations or review_citations):
        partial_reasons.append("部分资料经一致性校验被判定与目标实体无关，已剔除")
    elif not review_citations:
        partial_reasons.append("乐评来源本轮未在时间预算内取回足够结果")
    if not citations:
        partial_reasons.append("外部资料来源不足，无法做完整乐评总结")
    if report.problems:
        partial_reasons.extend(report.problems)
    if remaining_seconds(deadline_at) is not None and remaining_seconds(deadline_at) <= 0:
        partial_reasons.append(f"本轮达到 {int(settings.knowledge_turn_budget_seconds)} 秒知识链路预算")

    # 强搜索 provider 走 DeepSeek 先验（无真网页来源）时：去掉"乐评未取回/外部资料不足"的失败性描述，
    # 换成诚实的先验声明——仍标 partial 让前端 dossier-warning 显示，但内容来自直答/claim，不再是"资料不足"。
    is_parametric = web_knowledge_provider == "deepseek_parametric" and bool(
        web_knowledge_answer or web_knowledge_claims
    )
    if is_parametric:
        # parametric 直答是完整正文（只是未联网核实），不是"资料不完整"——
        # 剔除"乐评未取回/外部资料不足"这类失败性描述（它们不适用于一份成文的直答），
        # 但不再往 partial_reasons 塞免责声明：改由 is_parametric 标志承载，前端显柔和徽标、
        # 气泡出短声明，避免「明明是完整深度解读却挂琥珀色"资料不完整"警告」。
        partial_reasons = [
            r
            for r in partial_reasons
            if "乐评来源本轮未在时间预算内取回足够结果" not in r and "外部资料来源不足，无法做完整乐评总结" not in r
        ]

    ambiguous = entity.ambiguity == "ambiguous" and not is_compare
    guide = _listening_guide(entity, tracks, style_tags)
    career_phases: list[CareerPhase] = []
    summary_is_narrative = False
    if entity.type == "artist" and not is_compare and not ambiguous:
        career_phases = _build_career_timeline(entity, albums or [], tracks, meta_text, style_tags)
        # parametric 直答正文通常自带「年份+专辑」的时间脉络——实体解析空（无专辑年表）时，
        # 从正文抽出来替换掉上面的无年份空壳阶段，给 artist 仍一条可读时间线，而不是
        # 「代表作品 / 来源未提供明确发行年份」。抽空则保留原 career_phases。
        if is_parametric and web_knowledge_answer:
            _extracted_phases = _extract_career_phases_from_text(web_knowledge_answer)
            if _extracted_phases:
                career_phases = _extracted_phases
    if is_compare:
        profiled = _artist_compare_profile(entities[0], entities[1])
        # 有 profile 或 parametric 直答时，对比正文是可用的——剔除"乐评未取回/外部资料不足"
        # 失败性描述，与 album/artist 的 is_parametric 清洗一致，避免富正文却挂"部分降级"。
        if profiled or is_parametric:
            partial_reasons = [
                reason
                for reason in partial_reasons
                if "乐评来源本轮未在时间预算内取回足够结果" not in reason
                and "外部资料来源不足，无法做完整乐评总结" not in reason
            ]
        related = [entities[1]]
        if web_knowledge_answer:
            # DeepSeek 直答（parametric）：名艺人/名盘对比模型先验扎实，直接用作正文，
            # 不再回落静态「声音密度/叙事方式」模板（那是 resolve+metadata 全空时的旧兜底，
            # 对 The Weeknd vs Drake 这种常识对比明显空洞）。与下方 album/artist 的
            # web_knowledge_answer 直答通路一致；render 侧由 summary_is_narrative 跳过静态对比表。
            summary = _polish_narrative(web_knowledge_answer)[:4000]
            summary_is_narrative = True
            consensus = ""
        else:
            summary = _compare_summary(entities[0], entities[1])
            consensus = _critical_consensus(citations, opinions)
    elif ambiguous:
        # 同名歧义过大：不合成、不拼凑，返回消歧提示让用户补艺人名。
        summary = _ambiguous_summary(entity)
        related = entities[1:]
        consensus = ""
        partial_reasons.append("实体存在同名歧义，已改为返回消歧提示而非完整答案")
    elif not report.ok and not web_knowledge_answer and not web_knowledge_claims:
        # 证据归属不一致/全部偏题：抑制完整总结，回落机械兜底，防混拼错误实体。
        # （有 web_knowledge 直答/claim 时放过——先验产物不走一致性校验，仍可出带声明的总结。）
        related = entities[1:]
        summary = f"我整理了《{entity.name}》的可追溯音乐资料，但本轮证据归属不一致，未能合成可靠总结（原始资料多为英文，已避免直出以免误导）。"
        consensus = ""
        partial_reasons.append("证据归属不一致，已抑制完整总结以防混拼错误实体")
    else:
        related = entities[1:]
        if web_knowledge_answer:
            # DeepSeek 直答（parametric）：直接用作正文，不再二次合成——对名盘模型知识扎实，
            # 一次写全比「抽要点→再改写」更准、更省、更少信息损失（参考裸 DeepSeek chat 效果）。
            # 经 _polish_narrative 清洗：去掉模型自带的资料声明/风格标签尾巴（这些由卡片承载，
            # 否则正文与卡片重复），归一标题层级与空行。
            summary = _polish_narrative(web_knowledge_answer)[:4000]
            summary_is_narrative = True
            # 直答正文已含乐评口碑讨论，不再单列 consensus——否则 _critical_consensus 在无真引用时
            # 会输出"本轮没有拿到足够乐评来源"，与富正文自相矛盾、误导用户。
            consensus = ""
        else:
            # 合成层：把零散英文证据(MB/Spotify/Discogs/乐评摘录)交给 LLM 翻译+总结成
            # 连贯中文 summary/乐评共识，治"原始摘录直出、半句英文"。失败/无预算/无证据时
            # 回落机械摘要，保证 offline(MockLLM)与降级路径确定可用。
            synth = _synthesize_dossier_prose(
                agent,
                entity,
                meta_text,
                style_tags,
                evidence_pack,
                synth_citations,
                opinions,
                deadline_at,
                web_knowledge_claims=web_knowledge_claims,
                parametric=is_parametric,
            )
            if synth:
                summary = synth.get("summary") or f"我整理了《{entity.name}》的可追溯音乐资料。"
                consensus = synth.get("critical_consensus") or _critical_consensus(citations, opinions)
            else:
                summary = f"我整理了《{entity.name}》的可追溯音乐资料，但本轮未能合成完整中文介绍（资料来源多为英文原文，已避免直出半句英文）。"
                consensus = _critical_consensus(citations, opinions)
    key_tracks = tracks[:8]
    if is_compare:
        profiled = _artist_compare_profile(entities[0], entities[1])
        if profiled and profiled.get("entry_tracks"):
            track_map = profiled["entry_tracks"]
            key_tracks = [
                TrackRef(
                    title=title, artist=artist_name.title() if artist_name.islower() else artist_name, source="guide"
                )
                for artist_name, titles in track_map.items()
                for title in titles
            ][:10]
        else:
            grouped: list[TrackRef] = []
            for compare_entity in entities[:2]:
                seen_titles: set[str] = set()
                for track in tracks:
                    title = (track.title or "").strip()
                    if not title:
                        continue
                    if compare_entity.type == "artist":
                        match = _track_matches_artist(track, compare_entity.name)
                        if match is False:
                            continue
                    elif compare_entity.artist:
                        match = _track_matches_artist(track, compare_entity.artist)
                        if match is False:
                            continue
                    key = title.lower()
                    if key in seen_titles:
                        continue
                    seen_titles.add(key)
                    grouped.append(track)
                    if len(seen_titles) >= 4:
                        break
            if grouped:
                key_tracks = grouped[:8]

    # 个性化：把档案实体与用户曲库/口味交叉命中（不影响百科正文，只追加"你库里有…"）。
    # 同名歧义未澄清时不匹配——避免把无关同名实体的库歌硬塞进来。
    library_matches: list[LibraryMatch] = []
    if not ambiguous:
        library_matches = _match_library_to_entity(agent, user_id, entity, style_tags)

    dossier = MusicDossier(
        entity=entity,
        summary=summary,
        background=meta_text[:800] if meta_text else "本轮没有拿到足够稳定的背景资料。",
        style_tags=style_tags,
        critical_consensus=consensus,
        audience_reception=_audience_reception(citations),
        key_tracks=key_tracks,
        listening_guide=guide,
        career_phases=career_phases,
        related_albums=(albums or [])[:6],
        related_entities=related,
        library_matches=library_matches,
        citations=citations,
        review_opinions=opinions[: settings.knowledge_max_review_sources],
        uncertainties=[*partial_reasons, *evidence_pack.disagreements[:2]][:4],
        partial=bool(partial_reasons) or ambiguous or (not report.ok and not is_parametric),
        degraded_reason="；".join(partial_reasons) if partial_reasons else None,
        summary_is_narrative=summary_is_narrative,
        is_parametric=is_parametric,
    )
    write_cached_dossier(agent, dossier, intent=intent)
    return dossier


def _polish_narrative(text: str) -> str:
    """清洗 DeepSeek 直答正文，去掉与卡片重复/冗余的尾巴，归一 markdown 排版。

    模型即便被要求只写正文，仍常自带这些尾巴：资料声明（"资料状态：…未联网核实"）、
    风格标签行、空的"乐评共识：本轮没有拿到足够乐评来源"。这些信息卡片已用结构化字段
    （degraded_reason / style_tags / critical_consensus）承载，留在正文里就是重复。
    这里按行级保守剔除——只删整行匹配「元信息标签：…」的行，不动正文叙述。
    """
    if not text:
        return ""
    # 元信息行前缀：整行以这些标签开头时视为机械尾巴，删除（值可在冒号后任意）。
    meta_prefixes = (
        "资料状态",
        "资料来源状态",
        "风格标签",
        "风格定位",
        "乐评/资料共识",
        "乐评共识",
        "免责声明",
        "声明",
        "注：本",
        "备注：",
        "数据来源",
        "信息来源",
    )
    # 这些是模型自带的"没拿到来源/未联网"句式，整行命中即删（避免与卡片声明重复）。
    meta_substrings = (
        "未联网核实",
        "没有拿到足够乐评",
        "不硬凑专业评价",
        "基于模型先验",
        "以上内容基于",
        "仅供参考",
        "请以官方",
        "请另行确认",
    )
    kept: list[str] = []
    for raw in text.splitlines():
        line = raw.rstrip()
        stripped = line.lstrip("#*->•　 \t").strip()
        head = stripped.split("：", 1)[0].split(":", 1)[0].strip()
        if head in meta_prefixes:
            continue
        if any(sub in stripped for sub in meta_substrings):
            continue
        kept.append(line)
    cleaned = "\n".join(kept)
    # 归一：最多保留一个空行；去掉行尾空白；统一全角/裸标题层级到 markdown。
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    return cleaned.strip()


def _library_match_lines(dossier: MusicDossier) -> list[str]:
    """渲染「结合你的库」段落：把档案与用户曲库的命中接进回答，体现真正的个性化。

    artist 精确命中和 genre 同曲风扩展分开表述；口味契合（taste_aligned）的加标记。
    无命中返回空（不硬凑「你库里没有」这种负面话术）。
    """
    matches = dossier.library_matches or []
    if not matches:
        return []
    artist_hits = [m for m in matches if m.relation == "artist"]
    genre_hits = [m for m in matches if m.relation == "genre"]
    lines = ["\n结合你的曲库："]
    if artist_hits:
        names = "、".join(f"《{m.title}》" for m in artist_hits[:5] if m.title)
        aligned = any(m.taste_aligned for m in artist_hits)
        tail = "，正合你的口味" if aligned else ""
        lines.append(f"- 你库里已有 TA 的 {names}{tail}，可以直接对照着听。")
    if genre_hits:
        names = "、".join(f"《{m.title}》" for m in genre_hits[:4] if m.title)
        # 取命中曲风里更细的标签做说明（如"中文说唱/英伦摇滚"），无则泛称同风格。
        style = "、".join(dict.fromkeys(g for m in genre_hits[:4] for g in m.genre))[:40]
        style_hint = f"（{style}）" if style else ""
        lines.append(f"- 同风格{style_hint}你还听过 {names}，可作延伸。")
    return lines


# parametric 直答的诚实标注（气泡末尾一行短声明）。卡片层面由前端柔和徽标承载，
# 气泡保留这一行让"未联网核实"在正文上下文里也可见——用短句、不复读免责长文。
_PARAMETRIC_NOTE = "以上由 AI 知识库生成，未联网核实；具体来源/评分请另行确认。"


def _artist_career_answer(dossier: MusicDossier) -> str:
    """artist_deep_dive 专属渲染：职业生涯脉络（时间跨度/代表作）+ 入门路线，
    与专辑解读（曲目/乐评共识）明显区分。仅承载可追溯证据，不臆造分期。"""
    lines = [f"{dossier.entity.name}：{dossier.summary}"]
    if dossier.partial and dossier.degraded_reason:
        lines.append(f"\n资料状态：{dossier.degraded_reason}。")
    if dossier.style_tags:
        lines.append("\n风格定位：" + "、".join(dossier.style_tags[:6]))
    lines.append("\n职业生涯脉络：")
    for phase in dossier.career_phases:
        show_period = bool(phase.period) and phase.period != phase.phase_name
        head = f"- {phase.phase_name}" + (f"（{phase.period}）" if show_period else "")
        lines.append(head)
        if phase.key_releases:
            lines.append("  代表作品：" + "、".join(f"《{r}》" for r in phase.key_releases if r))
        if phase.sound_change:
            lines.append("  声音/风格：" + phase.sound_change)
        if phase.career_context:
            lines.append("  " + phase.career_context)
    if dossier.key_tracks:
        names = "、".join(f"《{t.title}》" for t in dossier.key_tracks[:5] if t.title)
        if names:
            lines.append("\n入门聆听路线：先听 " + names + "，再按上面的代表作扩展。")
    if dossier.critical_consensus:
        lines.append("\n乐评/资料共识：" + dossier.critical_consensus)
    lines.extend(_library_match_lines(dossier))
    if dossier.citations:
        lines.append("\n参考来源：")
        for c in dossier.citations[:3]:
            label = c.title or c.source
            lines.append(f"- {label}：{c.url}" if c.url else f"- {label}")
    if dossier.is_parametric:
        lines.append("\n" + _PARAMETRIC_NOTE)
    return "\n".join(lines)


def dossier_answer(dossier: MusicDossier) -> str:
    if dossier.related_entities:
        other = dossier.related_entities[0]
        if dossier.summary_is_narrative:
            # DeepSeek 直答对比正文：summary 已是成文对比，不再追加静态「看叙事方式」模板表
            # （那是无 profile 时的旧兜底，与富正文重复且空洞）。与 artist 直答通路一致：
            # 气泡只出正文 + 库命中 + parametric 短声明。
            parts = [f"{dossier.entity.name} 和 {other.name} 的区别：{dossier.summary}"]
            match_lines = _library_match_lines(dossier)
            if match_lines:
                parts.append("\n".join(match_lines))
            if dossier.is_parametric:
                parts.append(_PARAMETRIC_NOTE)
            return "\n".join(parts)
        profiled = _artist_compare_profile(dossier.entity, other)
        lines = [f"{dossier.entity.name} 和 {other.name} 的区别：{dossier.summary}"]
        if dossier.partial and dossier.degraded_reason:
            lines.append(f"\n资料状态：{dossier.degraded_reason}。")
        lines.extend(_compare_detail_lines(dossier.entity, other))
        if dossier.key_tracks and not profiled:
            names = "、".join(f"《{t.title}》" for t in dossier.key_tracks[:5] if t.title)
            if names:
                lines.append("\n本轮抓到的可听入口：" + names)
        lines.extend(_library_match_lines(dossier))
        if dossier.citations:
            lines.append("\n参考来源：")
            for c in dossier.citations[:3]:
                label = c.title or c.source
                lines.append(f"- {label}：{c.url}" if c.url else f"- {label}")
        return "\n".join(lines)
    if dossier.entity.type == "artist" and dossier.career_phases:
        return _artist_career_answer(dossier)
    if dossier.summary_is_narrative:
        # DeepSeek 直答：summary 本身就是一篇成文的 markdown 深度解读，正文已含曲目/口碑。
        # 气泡只出这篇正文，不再追加风格标签/资料状态/可以先听等机械尾巴——这些由前端
        # dossier 卡片用结构化字段承载，避免气泡与卡片重复整段内容。
        # 追加「结合你的曲库」（个性化信号）+ parametric 短声明（未联网核实的诚实标注）。
        parts = [dossier.summary]
        match_lines = _library_match_lines(dossier)
        if match_lines:
            parts.append("\n".join(match_lines))
        if dossier.is_parametric:
            parts.append(_PARAMETRIC_NOTE)
        return "\n".join(parts)
    lines = [f"{dossier.entity.name}：{dossier.summary}"]
    if dossier.partial and dossier.degraded_reason:
        lines.append(f"\n资料状态：{dossier.degraded_reason}。")
    if dossier.style_tags:
        lines.append("\n风格标签：" + "、".join(dossier.style_tags[:6]))
    if dossier.critical_consensus:
        lines.append("\n乐评/资料共识：" + dossier.critical_consensus)
    if dossier.key_tracks:
        names = "、".join(f"《{t.title}》" for t in dossier.key_tracks[:5])
        lines.append("\n可以先听：" + names)
    if dossier.listening_guide:
        lines.append("\n聆听路线：")
        lines.extend(f"- {item}" for item in dossier.listening_guide[:4])
    lines.extend(_library_match_lines(dossier))
    if dossier.citations:
        lines.append("\n参考来源：")
        for c in dossier.citations[:3]:
            label = c.title or c.source
            lines.append(f"- {label}：{c.url}" if c.url else f"- {label}")
    return "\n".join(lines)


def _musicbrainz_metadata(entity: MusicEntity) -> dict[str, Any] | None:
    """MusicBrainz 权威层：消歧实体名 + 补 MBID/标签/发行信息。

    受 settings.enable_musicbrainz 控制；失败或无命中返回 None，调用方按原逻辑降级。
    返回字段随实体类型不同（artist 给 type/country/disambiguation；album 给
    artist/date/type）。canonical_name 是 MB 的权威名，可纠正查询里的笔误/别名。
    """
    if not getattr(settings, "enable_musicbrainz", True):
        return None
    try:
        from app.sources.musicbrainz_client import MusicBrainzClient

        client = MusicBrainzClient()
        if entity.type == "artist":
            hit = None
            mbid = entity.external_ids.get("musicbrainz", "")
            if mbid:
                hit = client.lookup_artist(mbid)
            if not hit:
                hit = client.resolve_artist(entity.name)
                if hit and hit.get("mbid"):
                    # Search results are lightweight; follow with lookup to get URL relations.
                    hit = client.lookup_artist(hit["mbid"]) or hit
            if not hit or not hit.get("name"):
                return None
            summary_bits = [
                b
                for b in (
                    hit.get("type"),
                    f"来自 {hit['country']}" if hit.get("country") else "",
                    hit.get("disambiguation", ""),
                )
                if b
            ]
            return {
                "source": "musicbrainz",
                "canonical_name": hit.get("name", ""),
                "mbid": hit.get("mbid", ""),
                "tags": hit.get("tags") or [],
                "relations": hit.get("relations") or [],
                "summary": "，".join(summary_bits),
            }
        # album / track 都按专辑 release-group 查（track 粒度命中率低，先用专辑兜）。
        hit = None
        mbid = entity.external_ids.get("musicbrainz", "")
        if mbid:
            hit = client.lookup_release_group(mbid)
        if not hit:
            hit = client.resolve_release_group(entity.name, entity.artist)
            if hit and hit.get("mbid"):
                hit = client.lookup_release_group(hit["mbid"]) or hit
        if not hit or not hit.get("title"):
            return None
        summary_bits = [
            b
            for b in (
                f"艺人 {hit['artist']}" if hit.get("artist") else "",
                f"发行 {hit['date']}" if hit.get("date") else "",
                hit.get("type", ""),
            )
            if b
        ]
        return {
            "source": "musicbrainz",
            "canonical_name": hit.get("title", ""),
            "mbid": hit.get("mbid", ""),
            "artist": hit.get("artist", ""),
            "date": hit.get("date", ""),
            "type": hit.get("type", ""),
            "tags": hit.get("tags") or [],
            "relations": hit.get("relations") or [],
            "summary": "，".join(summary_bits),
        }
    except Exception:
        return None


def _spotify_metadata(entity: MusicEntity) -> dict[str, Any] | None:
    """Spotify：genres/popularity/封面 + top-track 音频特征转声音描述。

    需 OAuth client credentials；缺失/关闭/失败返回 None。音频特征（danceability/
    energy/valence/tempo）是推荐四锚里缺的"声学锚"，这里转成自然语言给 dossier。
    """
    if not getattr(settings, "enable_spotify", True):
        return None
    if not (settings.spotify_client_id and settings.spotify_client_secret):
        return None
    try:
        from app.sources.spotify_client import SpotifyClient

        client = SpotifyClient(settings.spotify_client_id, settings.spotify_client_secret)
        if entity.type == "artist":
            hit = client.search_artist(entity.name)
            if not hit:
                return None
            sound = client.audio_features_description(hit["id"]) if hit.get("id") else ""
            bits: list[str] = []
            if hit.get("genres"):
                bits.append("标签：" + "/".join(hit["genres"][:4]))
            if hit.get("popularity"):
                bits.append(f"热度 {hit['popularity']}/100")
            if sound:
                bits.append(f"声音：{sound}")
            return {
                "source": "spotify",
                "canonical_name": hit.get("name", ""),
                "external_id": hit.get("id", ""),
                "image": hit.get("image", ""),
                "genres": hit.get("genres") or [],
                "sound": sound,
                "summary": "，".join(bits),
            }
        hit = client.search_album(entity.name, entity.artist)
        if not hit:
            return None
        bits = []
        if hit.get("release_date"):
            bits.append(f"发行 {hit['release_date']}")
        if hit.get("total_tracks"):
            bits.append(f"{hit['total_tracks']} 曲")
        return {
            "source": "spotify",
            "canonical_name": hit.get("name", ""),
            "external_id": hit.get("id", ""),
            "artist": hit.get("artist", ""),
            "date": hit.get("release_date", ""),
            "image": hit.get("image", ""),
            "genres": [],
            "summary": "，".join(bits),
        }
    except Exception:
        return None


def _discogs_metadata(entity: MusicEntity) -> dict[str, Any] | None:
    """Discogs：权威发行年份 + 细类 styles（比 genres 更准）。

    需 Personal Access Token；缺失/关闭/失败返回 None。Discogs 的 styles 是
    社区标注的细分流派（如 Deep House / Detroit Techno），比泛 genres 更能刻画风格。
    """
    if not getattr(settings, "enable_discogs", True):
        return None
    if not settings.discogs_token:
        return None
    try:
        from app.sources.discogs_client import DiscogsClient

        client = DiscogsClient(settings.discogs_token)
        if entity.type == "artist":
            hit = client.resolve_artist(entity.name)
            if not hit:
                return None
            return {
                "source": "discogs",
                "external_id": hit.get("id", ""),
                "styles": hit.get("styles") or [],
                "genres": hit.get("genres") or [],
                "type": "artist",
                "summary": ("细类：" + "/".join(hit["styles"][:4])) if hit.get("styles") else "",
            }
        hit = client.resolve_release(entity.name, entity.artist)
        if not hit:
            return None
        title = hit.get("title", "")
        # Discogs title 形如 "Artist — Title"，取后半作为规范专辑名
        if " — " in title:
            title = title.split(" — ", 1)[1].strip()
        bits = []
        if hit.get("year"):
            bits.append(f"发行 {hit['year']}")
        if hit.get("styles"):
            bits.append("细类 " + "/".join(hit["styles"][:3]))
        elif hit.get("genres"):
            bits.append("/".join(hit["genres"][:3]))
        return {
            "source": "discogs",
            "canonical_name": title,
            "external_id": hit.get("id", ""),
            "year": hit.get("year", 0),
            "styles": hit.get("styles") or [],
            "genres": hit.get("genres") or [],
            "type": hit.get("type") or "master",
            "summary": "，".join(bits),
        }
    except Exception:
        return None


def _apply_structured_sources(
    entity: MusicEntity,
    sources: list[Any],
    metadata: list[dict[str, Any]],
    citations: list[MusicCitation],
) -> None:
    """把 MusicBrainz/Spotify/Discogs 的结构化结果合流到 entity + metadata/citations。

    MB 最权威（规范实体名/MBID）；Spotify 补封面/genres/声音；Discogs 补发行年份/细类。
    各源失败（None）独立跳过，互不影响——任一外部源挂掉不影响其余。
    """
    mb, sp, dc = sources
    if mb:
        canonical = mb.get("canonical_name") or ""
        # 精确比较：MB 的权威大小写值得纠正（frank ocean → Frank Ocean），
        # 仅在完全相同时跳过，避免无意义重写。
        if canonical and canonical != entity.name:
            entity.name = canonical
        if mb.get("mbid"):
            entity.external_ids["musicbrainz"] = mb["mbid"]
        if mb.get("artist") and not entity.artist:
            entity.artist = mb["artist"]
        tags = mb.get("tags") or []
        if tags or mb.get("summary"):
            mbid = mb.get("mbid", "")
            path = "artist" if entity.type == "artist" else "release-group"
            metadata.append({"entity": entity.model_dump(mode="json"), "summary": mb.get("summary", ""), "tags": tags})
            citations.append(
                MusicCitation(
                    source="musicbrainz",
                    title=f"{entity.name} - MusicBrainz",
                    url=f"https://musicbrainz.org/{path}/{mbid}" if mbid else "",
                    kind="encyclopedia",
                    excerpt=mb.get("summary", ""),
                    confidence=_source_confidence("musicbrainz"),
                )
            )
        relation_citations = _musicbrainz_relation_citations(entity, mb.get("relations") or [])
        if relation_citations:
            relation_labels = [f"{c.source}: {c.title or c.url}" for c in relation_citations[:5]]
            metadata.append(
                {
                    "entity": entity.model_dump(mode="json"),
                    "summary": "MusicBrainz 关联链接：" + "；".join(relation_labels),
                    "tags": [],
                    "relations": [c.model_dump(mode="json") for c in relation_citations],
                }
            )
            citations.extend(relation_citations)
    if sp:
        if sp.get("external_id"):
            entity.external_ids["spotify"] = sp["external_id"]
        if sp.get("image") and not entity.image:
            entity.image = sp["image"]
        if sp.get("artist") and not entity.artist:
            entity.artist = sp["artist"]
        tags = sp.get("genres") or []
        if tags or sp.get("summary"):
            sp_id = sp.get("external_id", "")
            sp_path = entity.type if entity.type in {"artist", "album"} else "album"
            metadata.append({"entity": entity.model_dump(mode="json"), "summary": sp.get("summary", ""), "tags": tags})
            citations.append(
                MusicCitation(
                    source="spotify",
                    title=f"{entity.name} - Spotify",
                    url=f"https://open.spotify.com/{sp_path}/{sp_id}" if sp_id else "",
                    kind="platform",
                    excerpt=sp.get("summary", ""),
                    confidence=0.72,
                )
            )
    if dc:
        if dc.get("external_id"):
            entity.external_ids["discogs"] = dc["external_id"]
        tags = dc.get("styles") or dc.get("genres") or []
        if tags or dc.get("summary"):
            dc_id = dc.get("external_id", "")
            dc_path = dc.get("type") or "master"
            metadata.append({"entity": entity.model_dump(mode="json"), "summary": dc.get("summary", ""), "tags": tags})
            citations.append(
                MusicCitation(
                    source="discogs",
                    title=f"{entity.name} - Discogs",
                    url=f"https://www.discogs.com/{dc_path}/{dc_id}" if dc_id else "",
                    kind="encyclopedia",
                    excerpt=dc.get("summary", ""),
                    confidence=_source_confidence("discogs"),
                )
            )


_MB_RELATION_ALLOW_TERMS = {
    "review",
    "allmusic",
    "bbc music page",
    "discogs",
    "wikidata",
    "wikipedia",
    "discography entry",
    "discography page",
    "other databases",
    "lyrics",
    "secondhandsongs",
    "whosampled",
    "imdb",
    "official homepage",
}
_MB_RELATION_SKIP_TERMS = {
    "social network",
    "youtube",
    "streaming",
    "download",
    "purchase for download",
    "free streaming",
    "license",
    "setlistfm",
    "bandsintown",
    "songkick",
    "patronage",
}


def _musicbrainz_relation_citations(entity: MusicEntity, relations: list[dict[str, Any]]) -> list[MusicCitation]:
    """Convert curated MusicBrainz URL relations into citable evidence links.

    These links are often better targeted than a broad web search: MusicBrainz editors
    attach release/artist-specific BBC reviews, AllMusic pages, Discogs masters,
    RateYourMusic pages, Wikidata entries and archived reviews directly to the entity.
    """
    citations: list[MusicCitation] = []
    for rel in relations:
        if not isinstance(rel, dict):
            continue
        url = str(rel.get("url") or "").strip()
        if not url:
            continue
        rel_type = str(rel.get("type") or "").strip()
        rel_key = rel_type.lower()
        source = _source_from_url(url) or rel_key or "musicbrainz"
        source_key = _source_key(source)
        is_allowed = any(term in rel_key for term in _MB_RELATION_ALLOW_TERMS)
        is_valuable_source = source_key in (TIER_A_SOURCES | TIER_B_SOURCES | {"whosampled", "secondhandsongs", "imdb"})
        if not is_allowed and not is_valuable_source:
            continue
        if any(term in rel_key for term in _MB_RELATION_SKIP_TERMS):
            continue

        kind: str = "encyclopedia"
        if "review" in rel_key or (
            source_key
            in {"allmusic", "bbc", "pitchfork", "guardian", "rollingstone", "nme", "time", "ew", "chicagotribune"}
            and entity.type != "artist"
        ):
            kind = "review"
        elif source_key in {"rateyourmusic", "albumoftheyear", "musicboard"}:
            kind = "user_comment"
        elif rel_key in {"lyrics"} or source_key in {"genius"}:
            kind = "platform"

        label = _relation_label(rel_type, source)
        title_parts = []
        if entity.artist and entity.type in {"album", "track"}:
            title_parts.append(entity.artist)
        title_parts.append(entity.name)
        title = " - ".join(part for part in title_parts if part)
        if label:
            title = f"{title} ({label})"
        confidence = max(_source_confidence(source, url), 0.58)
        if kind == "review":
            confidence = max(confidence, 0.72)
        if rel.get("ended"):
            confidence = max(0.45, confidence - 0.08)
        citations.append(
            MusicCitation(
                source=source_key or source,
                title=title,
                url=url,
                kind=kind,  # type: ignore[arg-type]
                excerpt="",
                confidence=min(0.92, confidence),
            )
        )
    return sorted(
        _dedupe_citations(citations),
        key=lambda c: (c.confidence, _music_relation_source_priority(c.source), c.kind == "review"),
        reverse=True,
    )[:8]


def _music_relation_source_priority(source: str) -> int:
    key = _source_key(source)
    priorities = {
        "bbc": 6,
        "allmusic": 6,
        "pitchfork": 5,
        "guardian": 5,
        "rollingstone": 5,
        "nme": 4,
        "time": 4,
        "ew": 4,
        "chicagotribune": 4,
        "rateyourmusic": 3,
        "discogs": 3,
        "wikidata": 2,
        "wikipedia": 2,
    }
    return priorities.get(key, 1)


def _relation_label(rel_type: str, source: str) -> str:
    raw = (rel_type or source or "").strip()
    key = raw.lower()
    labels = {
        "allmusic": "AllMusic",
        "bbc music page": "BBC Music",
        "review": "review",
        "discogs": "Discogs",
        "wikidata": "Wikidata",
        "wikipedia": "Wikipedia",
        "discography entry": "discography",
        "discography page": "discography",
        "other databases": source or "database",
        "lyrics": "lyrics",
    }
    return labels.get(key, raw)


_SYNTH_FLOOR_SECONDS = 5.0  # 合成 LLM(开思考) 必须保住的最小预算；正文抓取只能用它之外的部分。

# Tavily Extract 实测取空的强反爬源——不浪费预算（它们的乐评靠 review_search 关键词摘要兜底）。
_EXTRACT_BLOCKED_DOMAINS = {"allmusic.com", "rateyourmusic.com", "albumoftheyear.com"}
# 正文最稳的来源（实测 Tavily Extract 能取到），抓取排序时优先。
_RELIABLE_DOMAINS = {"last.fm", "discogs.com", "genius.com", "musik-sammler.de", "wikidata.org"}


def _citation_domain(url: str) -> str:
    m = re.search(r"https?://(?:www\.)?([^/]+)", url or "", flags=re.I)
    return (m.group(1) or "").lower() if m else ""


def _review_extract_rank(c: MusicCitation) -> tuple:
    domain = _citation_domain(c.url)
    reliable = 0 if any(domain == d or domain.endswith("." + d) for d in _RELIABLE_DOMAINS) else 1
    kind_rank = 0 if c.kind == "review" else (1 if c.kind == "encyclopedia" else 2)
    tier = _source_tier(c.source, c.url)
    tier_rank = {"A": 0, "B": 1, "C": 2}.get(tier, 2)
    return (reliable, kind_rank, tier_rank, -c.confidence)


def _enrich_review_content(
    citations: list[MusicCitation],
    entity: MusicEntity,
    deadline_at: float | None,
) -> list[MusicCitation]:
    """把 citation 的 URL 正文抓回来填进 excerpt（Tavily Extract + Discogs API）。

    核心价值：MusicBrainz relations 里 last.fm/Discogs/Genius 等高质量来源只有 URL、excerpt
    为空（_musicbrainz_relation_citations 在 excerpt="" 处产出），喂不进合成 LLM，所以专业
    乐评写不出来。这里在**受保护预算**内并行抓 top-N 的真实正文，填回 excerpt，让
    _synthesize_dossier_prose 拿到内容出专业中文乐评。

    鲁棒兜底：当 MusicBrainz 超时没给到 relation URL 时（实测 MB 常在 3s 预算内被砍），
    直接用实体 (artist, name) 构造 last.fm/Discogs 的稳定入口——不依赖 MB 成功，保证
    last.fm（含 Metascore/乐评）和 Discogs（发行 notes）这两个最稳源仍能被抓回。

    受保护预算：合成 LLM 至少保 _SYNTH_FLOOR_SECONDS；抓取只能用其余部分，且不超过配置上限。
    预算太紧时整段放弃（不抓），保住合成，绝不为了抓正文而拖垮整条链路。
    """
    # 即使 citations 为空（MB/review 都没拿到），只要实体能构造 last.fm/Discogs 入口，仍要抓。
    can_construct = entity.type in {"album", "track"} and bool(entity.artist) and bool(entity.name)
    if not citations and not can_construct:
        return citations
    remaining = remaining_seconds(deadline_at)
    extract_budget = settings.knowledge_review_extract_timeout_seconds
    if remaining is not None:
        extract_budget = max(0.0, min(extract_budget, remaining - _SYNTH_FLOOR_SECONDS))
    if extract_budget < 1.5:
        return citations  # 预算太紧，保合成、放弃抓正文。

    api_key = settings.tavily_api_key
    discogs_enabled = getattr(settings, "enable_discogs", True) and bool(getattr(settings, "discogs_token", ""))

    # 构造兜底 citation：MB 没给 relation URL 时，用实体直接拼 last.fm/Discogs 入口。
    pool = list(citations)
    existing_domains = {_citation_domain(c.url) for c in citations}

    def _has_domain(domain: str) -> bool:
        return any(domain == d or d.endswith("." + domain) or d.endswith(domain) for d in existing_domains)

    if entity.type in {"album", "track"} and entity.artist and entity.name:
        if api_key and not _has_domain("last.fm"):
            artist_slug = entity.artist.strip().replace(" ", "+")
            name_slug = entity.name.strip().replace(" ", "+")
            pool.append(
                MusicCitation(
                    source="lastfm",
                    title=f"{entity.artist} - {entity.name}",
                    url=f"https://www.last.fm/music/{artist_slug}/{name_slug}",
                    kind="encyclopedia",
                    excerpt="",
                    confidence=0.7,
                )
            )
        if discogs_enabled and not _has_domain("discogs.com"):
            # Discogs 走 API 按名搜索，URL 仅作占位/去重标识。
            pool.append(
                MusicCitation(
                    source="discogs",
                    title=f"{entity.artist} - {entity.name}",
                    url="https://www.discogs.com/",
                    kind="encyclopedia",
                    excerpt="",
                    confidence=0.6,
                )
            )

    def _worth(c: MusicCitation) -> bool:
        domain = _citation_domain(c.url)
        if not domain:
            return False
        if len((c.excerpt or "").strip()) >= 120:
            return False  # 已有正文（如 review_search 摘要）就不重复抓全文。
        if any(domain == d or domain.endswith("." + d) for d in _EXTRACT_BLOCKED_DOMAINS):
            return False  # 强反爬源，Tavily Extract 取空，省预算。
        return True

    candidates = sorted((c for c in pool if _worth(c)), key=_review_extract_rank)
    candidates = candidates[: settings.knowledge_review_extract_max_sources]
    if not candidates:
        return citations

    def _fetch(c: MusicCitation) -> str:
        domain = _citation_domain(c.url)
        if "discogs.com" in domain:
            if entity.type not in {"album", "track"}:
                return ""
            try:
                from app.sources.discogs_client import DiscogsClient

                return DiscogsClient(getattr(settings, "discogs_token", "")).fetch_release_notes(
                    entity.name, entity.artist
                )
            except Exception:
                logger.debug("Discogs notes fetch failed for %s", entity.name, exc_info=True)
                return ""
        if not api_key:
            return ""
        # 并行抓取，墙钟≈最慢单条；每条都给足 extract_budget。
        return web_search_source.fetch_url_content(c.url, api_key=api_key, timeout=max(1.0, extract_budget))

    tasks = [(f"extract:{_citation_domain(c.url)}", lambda c=c: _fetch(c)) for c in candidates]
    batches = run_parallel(tasks, timeout=max(0.5, extract_budget), default="")
    for c, text in zip(candidates, batches, strict=False):
        text = (text or "").strip()
        if text:
            c.excerpt = text[:1800]
    # 把新构造（非原列表）、且抓到正文、URL 不重复的 citation 并回，供下游 build_evidence_pack 使用。
    original_urls = {(c.url or "").strip().lower() for c in citations}
    for c in candidates:
        url = (c.url or "").strip().lower()
        if c.excerpt and url and url not in original_urls:
            citations.append(c)
            original_urls.add(url)
    return citations


def _opinions_from_citations(citations: list[MusicCitation]) -> list[ReviewOpinion]:
    """从有正文的 citation 生成 ReviewOpinion（复用 sentiment/aspect 规则）。

    relation citation 抓回正文后，让它们也贡献 sentiment/aspect，充实乐评共识——
    _critical_consensus 的正面/分歧统计与合成证据块都吃 opinions。
    """
    out: list[ReviewOpinion] = []
    for c in citations:
        text = (c.excerpt or "").strip()
        if len(text) < 60:
            continue
        out.append(
            ReviewOpinion(
                source=c.source,
                sentiment=_sentiment_from_text(text),
                aspects=_aspects_from_text(text),
                summary=text[:180],
            )
        )
    return out


def _metadata_for_entity(agent: Any, entity: MusicEntity, timeout: float | None = None) -> dict[str, Any]:
    """单实体多源元数据：**所有源在同一并行波次内拉取**，再在主线程确定性合流。

    根因教训：旧实现把结构化源（MB/Spotify/Discogs）并行、却把 netease/lastfm/web
    串行接在后面，整段被外层 lookup_metadata 的 source_timeout 包住——串行尾巴一旦
    超出预算，外层 run_parallel 直接取消并丢回 default={}，连已成功的 Discogs styles
    一起作废（Blonde/The Weeknd 实测 metadata 全空即此故障）。

    现在所有源是单波并行：墙钟 ≈ 最慢单源，而非各源之和。各 thunk 只取原始数据、
    **不改 entity**（线程安全），实体合流（canonical 名/external_ids/封面）统一在
    主线程按固定顺序做，输出确定。inner 批超时比外层预算留 0.3s 余量，保证内层先收口、
    部分结果不被外层取消吞掉。
    """
    citations: list[MusicCitation] = []
    metadata: list[dict[str, Any]] = []
    tracks: list[TrackRef] = []
    albums: list[dict[str, Any]] = []
    lastfm_key = getattr(settings, "lastfm_api_key", "")

    def _lastfm_bundle() -> dict[str, Any] | None:
        if not lastfm_key:
            return None
        from app.sources.lastfm_client import LastfmClient

        client = LastfmClient(lastfm_key)
        info = client.get_artist_info(entity.name)
        if not info:
            return None
        return {"info": info, "top": client.get_artist_top_tracks(entity.name, 6)}

    def _netease_album_bundle() -> dict[str, Any] | None:
        from app.sources import netease as netease_source

        album = netease_source.search_netease_album(entity.artist, entity.name)
        if not album:
            return None
        detail = netease_source.fetch_netease_album_tracks(str(album.get("id") or ""), 12)
        return {"album": album, "detail": detail}

    def _web_bundle() -> list[dict[str, Any]]:
        query = " ".join(part for part in [entity.artist, entity.name, "music background"] if part)
        return web_search_source.search_web_info(query, max_results=2, api_key=settings.tavily_api_key)

    tasks: list[tuple[str, Any]] = [
        ("musicbrainz", lambda: _musicbrainz_metadata(entity)),
        ("spotify", lambda: _spotify_metadata(entity)),
        ("discogs", lambda: _discogs_metadata(entity)),
    ]
    if entity.type == "artist":
        tasks.append(("netease_albums", lambda: agent.recommend_artist_albums("", entity.name, limit=4)))
        tasks.append(("lastfm", _lastfm_bundle))
    elif entity.type == "album":
        tasks.append(("netease_album", _netease_album_bundle))
    tasks.append(("web", _web_bundle))

    budget = timeout if timeout is not None else settings.knowledge_source_timeout_seconds
    batch_timeout = max(0.5, budget - 0.3)
    results = run_parallel(tasks, timeout=batch_timeout, default=None)
    res = dict(zip([label for label, _ in tasks], results, strict=False))

    try:
        # 1) 结构化权威源合流（顺序：MB 纠正实体名 → Spotify 补声音/封面 → Discogs 补细类）。
        _apply_structured_sources(
            entity,
            [res.get("musicbrainz"), res.get("spotify"), res.get("discogs")],
            metadata,
            citations,
        )
        # 2) 网易云艺人专辑。
        albums = res.get("netease_albums") or []
        for album in albums:
            citations.append(
                MusicCitation(
                    source="netease",
                    title=album.get("name", ""),
                    url="",
                    kind="platform",
                    excerpt=f"网易云专辑结果：{album.get('name', '')}",
                    confidence=0.8,
                )
            )
        # 3) Last.fm 简介 + 热门曲。
        lf = res.get("lastfm")
        if lf and lf.get("info"):
            info = lf["info"]
            metadata.append(
                {
                    "entity": entity.model_dump(mode="json"),
                    "summary": info.get("bio", ""),
                    "tags": info.get("tags", []),
                    "image": info.get("image", ""),
                }
            )
            citations.append(
                MusicCitation(
                    source="lastfm",
                    title=f"{entity.name} - Last.fm",
                    url="",
                    kind="metadata",
                    excerpt=(info.get("bio") or "")[:500],
                    confidence=0.7,
                )
            )
            for t in lf.get("top") or []:
                tracks.append(
                    TrackRef(title=t.get("title", ""), artist=t.get("artist") or entity.name, source="lastfm")
                )
        # 4) 网易云专辑元数据 + 曲目（合流时再回写 entity，线程安全）。
        nb = res.get("netease_album")
        if nb:
            album = nb["album"]
            entity.external_ids["netease_album"] = str(album.get("id") or "")
            entity.image = album.get("cover", "") or entity.image
            entity.artist = album.get("artist", "") or entity.artist
            metadata.append(
                {
                    "entity": entity.model_dump(mode="json"),
                    "summary": f"网易云识别到专辑《{album.get('name')}》，艺人 {album.get('artist') or '未知'}。",
                }
            )
            citations.append(
                MusicCitation(
                    source="netease",
                    title=album.get("name", ""),
                    kind="platform",
                    excerpt="网易云专辑元数据",
                    confidence=0.85,
                )
            )
            for item in (nb.get("detail") or {}).get("tracks", [])[:8]:
                tracks.append(
                    TrackRef(
                        title=item.get("title", ""),
                        artist=item.get("artist", "") or entity.artist,
                        source="netease",
                        source_id=str(item.get("song_id") or ""),
                    )
                )
        # 5) Web 背景资料。
        for item in (res.get("web") or [])[:2]:
            metadata.append(
                {
                    "entity": entity.model_dump(mode="json"),
                    "summary": item.get("content", ""),
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                }
            )
            citations.append(
                MusicCitation(
                    source=_source_from_url(item.get("url", "")) or "web",
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    kind="encyclopedia",
                    excerpt=(item.get("content") or "")[:500],
                    confidence=0.65,
                )
            )
    except Exception:
        pass
    return {"metadata": metadata, "citations": citations, "tracks": tracks, "albums": albums}


def _guess_entity_name(query: str) -> str:
    text = re.sub(r"[《》“”\"']", " ", query or "")
    text = _strip_entity_noise(text)
    text = re.sub(
        r"(区别在哪|有什么区别|区别|不同|系统|音乐路线|是什么|怎么样|如何|风格差异|差异在哪)", " ", text, flags=re.I
    )
    text = re.sub(r"(并给我|给我|顺便给我|再给我|分别给我|各自的).*$", " ", text, flags=re.I)
    text = re.sub(r"(入门歌|入门曲|代表作|代表歌|推荐歌单|推荐歌曲).*$", " ", text, flags=re.I)
    text = text.strip(" ？?，,。.")
    text = re.sub(r"\s*的\s*$", "", text.strip())
    text = re.sub(r"\s+", " ", text).strip(" ？?，,。.")
    return _canonical_music_name(text or (query or "未知音乐实体").strip())


def _compare_names(query: str) -> list[str]:
    raw = re.sub(r"[《》“”\"']", " ", query or "")
    raw = re.sub(r"^\s*(比较一下|比较|对比一下|对比)\s*", "", raw, flags=re.I)
    parts = re.split(r"\s+(?:vs|VS|Vs)\s+|\s+和\s+|和|对比|比较", raw)
    names: list[str] = []
    seen: set[str] = set()
    for part in parts:
        cleaned = _guess_entity_name(part)
        normalized = cleaned.strip().lower()
        if not cleaned or normalized in seen or normalized == "未知音乐实体":
            continue
        seen.add(normalized)
        names.append(cleaned)
    return names[:2]


def _canonical_music_name(name: str) -> str:
    key = re.sub(r"[^a-z0-9]+", " ", (name or "").lower()).strip()
    aliases = {
        "orange channel": "Channel Orange",
        "channel orange": "Channel Orange",
        "blond": "Blonde",
        "blonde": "Blonde",
        "ok computer": "OK Computer",
        "kid a": "Kid A",
    }
    return aliases.get(key, name.strip())


def _compare_summary(left: MusicEntity, right: MusicEntity) -> str:
    pair = {left.name.lower(), right.name.lower()}
    profiled = _artist_compare_profile(left, right)
    if profiled:
        return profiled["summary"]
    if {"kid a", "ok computer"} <= pair:
        return (
            "《OK Computer》仍站在吉他摇滚和戏剧化歌曲结构里讨论现代焦虑；"
            "《Kid A》则主动拆掉摇滚乐队的熟悉轮廓，把电子、ambient、krautrock 和爵士质感推到前台。"
        )
    if {"blonde", "channel orange"} <= pair:
        return (
            "《Channel Orange》更像一张叙事清晰、色彩更浓的现代 R&B/neo-soul 专辑；"
            "《Blonde》则更碎片化、更极简，更像关于记忆、身份和亲密关系的私人电影。"
        )
    return "前者与后者的核心差异主要落在声音密度、叙事方式、情绪表达和入门门槛上。"


def _compare_detail_lines(left: MusicEntity, right: MusicEntity) -> list[str]:
    pair = {left.name.lower(), right.name.lower()}
    profiled = _artist_compare_profile(left, right)
    if profiled:
        return list(profiled["lines"])
    if {"kid a", "ok computer"} <= pair:
        return [
            "\n1. 声音/制作：\n- 《OK Computer》核心仍是吉他、鼓、贝斯和 Thom Yorke 的旋律线，只是编曲更宏大、更偏 art rock。\n- 《Kid A》把乐队声响打散，更多使用电子纹理、合成器、采样式剪辑、冷感鼓机和爵士铜管。",
            "\n2. 结构与叙事：\n- 《OK Computer》像一组关于技术社会、交通、消费和疏离的末日短片，歌曲仍有清晰起承转合。\n- 《Kid A》更像进入一个断裂的系统：歌词更碎，声音更抽象，很多歌不是讲故事，而是在制造状态。",
            "\n3. 情绪表达：\n- 《OK Computer》的焦虑是戏剧性的、外放的，像人在现代机器里尖叫。\n- 《Kid A》的焦虑更冷、更去人化，像声音本身已经被系统吞掉。",
            "\n4. 入门聆听路线：\n- 先听《OK Computer》的《Paranoid Android》《Karma Police》《No Surprises》，抓住 Radiohead 的旋律和戏剧性。\n- 再听《Kid A》的《Everything In Its Right Place》《The National Anthem》《How to Disappear Completely》《Idioteque》，感受他们如何从摇滚转向电子和氛围结构。",
            "\n5. 如果你喜欢哪张：\n- 喜欢《OK Computer》，下一步听《The Bends》或《In Rainbows》。\n- 喜欢《Kid A》，下一步听《Amnesiac》或 Thom Yorke 的个人电子作品。",
        ]
    if {"blonde", "channel orange"} <= pair:
        return [
            "\n1. 声音：\n- 《Channel Orange》更饱满，有 funk、soul、R&B 和流行旋律的清晰骨架。\n- 《Blonde》更稀疏，常用吉他、变调人声、留白和漂浮的氛围来推进。",
            "\n2. 叙事：\n- 《Channel Orange》像短篇故事集，角色、场景和社会观察更明确。\n- 《Blonde》像记忆残片，很多情绪不解释清楚，而是让你自己在空白里补完。",
            "\n3. 听感门槛：\n- 《Channel Orange》更容易第一次就抓住，代表曲更像传统意义上的 song。\n- 《Blonde》更需要反复听，它的高潮经常不是副歌爆点，而是情绪突然塌下来或变得很近。",
            "\n4. 入门建议：\n- 想先理解 Frank Ocean 的叙事和旋律天赋，先听《Channel Orange》。\n- 想理解他为什么影响后来的 alternative R&B、卧室流行和内省型创作，听《Blonde》。",
        ]
    return [
        f"\n1. 先听 {left.name} 的代表曲，再听 {right.name} 的代表曲，比较哪一个更依赖旋律/律动，哪一个更依赖氛围/结构。",
        "\n2. 看叙事方式：一个可能更直接讲故事，另一个可能更偏情绪片段或声音实验。",
        "\n3. 看入门门槛：更流畅的一方适合先听，更抽象的一方适合带着背景和歌词慢慢进入。",
    ]


def _build_artist_compare_profile(
    left: MusicEntity,
    right: MusicEntity,
    *,
    summary: str,
    sides: dict[str, dict[str, Any]],
    shared_ground: list[str],
    intersection_summary: str,
    collaboration_tracks: list[str],
) -> dict[str, Any]:
    left_key = left.name.lower()
    right_key = right.name.lower()
    left_side = sides[left_key]
    right_side = sides[right_key]
    return {
        "summary": summary,
        "axes": [
            {
                "axis": "声音重心",
                "left": left_side["sound_focus"],
                "right": right_side["sound_focus"],
                "summary": "两边的核心吸引力不在同一个维度。",
            },
            {
                "axis": "叙事方式",
                "left": left_side["narrative"],
                "right": right_side["narrative"],
                "summary": "一个更靠可引用的表达，一个更靠整体状态与氛围。",
            },
            {
                "axis": "入门路径",
                "left": left_side["entry_path"],
                "right": right_side["entry_path"],
                "summary": "入门时最好先顺着各自最强的场景切。",
            },
        ],
        "shared_ground": shared_ground,
        "intersection_summary": intersection_summary,
        "collaboration_tracks": collaboration_tracks,
        "lines": [
            f"\n1. 声音重心：\n- {left.name}：{left_side['sound_focus']}\n- {right.name}：{right_side['sound_focus']}",
            f"\n2. 叙事方式：\n- {left.name}：{left_side['narrative']}\n- {right.name}：{right_side['narrative']}",
            f"\n3. 节奏与场景：\n- {left.name}：{left_side['entry_path']}\n- {right.name}：{right_side['entry_path']}",
            (
                f"\n4. 入门歌：\n- {left.name}：先听"
                f"{''.join(f'《{title}》' for title in left_side['entry_tracks'][:5])}。\n"
                f"- {right.name}：先听"
                f"{''.join(f'《{title}》' for title in right_side['entry_tracks'][:5])}。"
            ),
            (
                f"\n5. 怎么判断你更吃哪边：\n- 如果你更在意 {left_side['preference_test']}，先从 {left.name} 走。\n"
                f"- 如果你更在意 {right_side['preference_test']}，先从 {right.name} 走。"
            ),
        ],
        "entry_tracks": {
            left_key: list(left_side["entry_tracks"]),
            right_key: list(right_side["entry_tracks"]),
        },
        "artist_cards": [
            {
                "name": left.name,
                "reason": left_side["card_reason"],
                "representative_tracks": list(left_side["entry_tracks"][:3]),
            },
            {
                "name": right.name,
                "reason": right_side["card_reason"],
                "representative_tracks": list(right_side["entry_tracks"][:3]),
            },
        ],
    }


def _artist_compare_profile(left: MusicEntity, right: MusicEntity) -> dict[str, Any] | None:
    """High-confidence artist comparison profiles for common interview/demo pairs."""
    if left.type != "artist" or right.type != "artist":
        return None
    names = {left.name.lower(), right.name.lower()}
    if {"drake", "future"} <= names:
        return _build_artist_compare_profile(
            left,
            right,
            summary=(
                "Drake 更像把说唱、R&B 旋律和流行歌曲结构打通的主流叙事者；"
                "Future 更偏 trap 氛围、Auto-Tune 质感和情绪化的重复律动，核心魅力在声音状态和黑暗能量。"
            ),
            sides={
                "drake": {
                    "sound_focus": "hook、旋律化 rap、人声亲近感，容易进入流行语境。",
                    "narrative": "把关系、成功焦虑、城市夜生活写成可引用的个人独白。",
                    "entry_path": "适合从旋律说唱、R&B crossover、俱乐部单曲进入。",
                    "entry_tracks": [
                        "Headlines",
                        "Hold On, We're Going Home",
                        "Marvins Room",
                        "Passionfruit",
                        "Nonstop",
                    ],
                    "preference_test": "旋律、歌词可引用度和 pop 结构",
                    "card_reason": "旋律说唱、R&B crossover、hook 感和流行结构更强。",
                },
                "future": {
                    "sound_focus": "低频、808、迷幻合成器和 Auto-Tune 人声纹理，听感更黏、更暗。",
                    "narrative": "把情绪压进重复短句和 ad-lib，叙事不一定完整但状态非常强。",
                    "entry_path": "适合从 trap banger、mixtape 气质和夜晚驾驶感进入。",
                    "entry_tracks": [
                        "March Madness",
                        "Mask Off",
                        "Codeine Crazy",
                        "Thought It Was a Drought",
                        "Low Life",
                    ],
                    "preference_test": "氛围、低频冲击和情绪沉浸",
                    "card_reason": "808、Auto-Tune、trap 氛围和夜晚驾驶感更强。",
                },
            },
            shared_ground=[
                "都站在 2010s 主流 rap/R&B 核心地带。",
                "都大量使用旋律化说唱，而不是纯技术型密集 spit。",
                "都擅长把夜生活、成功后的空心感和关系拉扯写进热门单曲。",
            ],
            intersection_summary=(
                "两人的交集不在“唱得像不像”，而在于都把旋律说唱、俱乐部能量和夜晚情绪"
                "推到了主流 rap 的中心，只是 Drake 更亮、更会写流行句子，Future 更暗、更会做状态。"
            ),
            collaboration_tracks=["Jumpman", "Digital Dash", "Big Rings", "Scholarships", "Live From the Gutter"],
        )
    if {"drake", "the weeknd"} <= names:
        return _build_artist_compare_profile(
            left,
            right,
            summary=(
                "Drake 更像把 rap、R&B 和流行单曲结构熔在一起的城市叙事者；"
                "The Weeknd 更偏暗色 synth-pop / alternative R&B 的氛围导演，强项是欲望、孤独和夜生活的电影感。"
            ),
            sides={
                "drake": {
                    "sound_focus": "hook、旋律化 rap、R&B 过门和更贴近主流流行的歌曲结构。",
                    "narrative": "把关系、虚荣、成功焦虑和城市夜生活写成第一人称独白，句子感很强。",
                    "entry_path": "适合从旋律说唱、情歌向 rap、俱乐部 crossover 单曲进入。",
                    "entry_tracks": [
                        "Headlines",
                        "Hold On, We're Going Home",
                        "Marvins Room",
                        "Passionfruit",
                        "One Dance",
                    ],
                    "preference_test": "旋律说唱、歌词可引用度和流行结构",
                    "card_reason": "旋律 rap 与 R&B 过门自然，最强项是可引用的人称叙事与单曲感。",
                },
                "the weeknd": {
                    "sound_focus": "假声、暗色合成器、80s synth-pop 光泽和更强的夜色氛围包裹感。",
                    "narrative": "更少直接讲完整故事，更擅长把欲望、空虚、堕落和自毁写成一整片情绪场。",
                    "entry_path": "适合从 dark R&B 代表作、流行爆单和电影感长线作品三条线切入。",
                    "entry_tracks": [
                        "Wicked Games",
                        "The Hills",
                        "Can't Feel My Face",
                        "Blinding Lights",
                        "After Hours",
                    ],
                    "preference_test": "氛围、假声质感、夜色电影感和 synth-pop 包裹感",
                    "card_reason": "暗色 alt-R&B 与 synth-pop 气质更浓，最强项是夜色氛围和人声质感。",
                },
            },
            shared_ground=[
                "两人都把 Toronto 气质、夜生活主题和旋律性推到 2010s 主流中心。",
                "都擅长把脆弱、自我放大和关系拉扯写进高度流行化的作品里。",
                "都不是纯技术炫技型路线，真正吸引人的点在情绪表达和声音世界。",
            ],
            intersection_summary=(
                "两人的交集在于都能把夜晚、欲望和情绪空洞做成主流流行语境里的大歌，"
                "但 Drake 更像把这些写成可引用的日记，The Weeknd 更像把这些拍成霓虹色的夜间电影。"
            ),
            collaboration_tracks=["Crew Love", "Live For", "The Zone"],
        )
    return None


def _infer_entity_type(query: str, intent: str) -> str:
    q = (query or "").lower()
    if intent == "sample_lookup":
        return "track"
    if intent == "music_compare":
        if any(k in q for k in ["专辑", "album", "这张", "唱片"]):
            return "album"
        if any(k in q for k in ["这首", "歌曲", "track", "single", "单曲"]):
            return "track"
        if any(k in q for k in ["歌手", "艺人", "乐队", "artist", "风格", "路线", "入门歌", "代表作", "各自"]):
            return "artist"
        return "album"
    if intent == "artist_deep_dive" or any(k in q for k in ["歌手", "艺人", "乐队", "音乐路线", "artist"]):
        return "artist"
    if intent in {"album_deep_dive", "review_summary"} or any(k in q for k in ["专辑", "album", "这张"]):
        return "album"
    if any(k in q for k in ["这首", "歌曲", "track", "single"]):
        return "track"
    return "album" if intent == "music_compare" else "artist"


def _infer_artist(query: str, name: str, entity_type: str) -> str:
    if entity_type != "album":
        return ""
    m = re.search(r"(.+?)\s*(?:的|-)\s*" + re.escape(name), query or "", re.I)
    return m.group(1).strip() if m else ""


def _source_from_url(url: str) -> str:
    archive = re.search(r"web\.archive\.org/web/\d+/(?:https?://)?(?:www\.)?([^/]+)", url or "", flags=re.I)
    if archive:
        return _source_from_host(archive.group(1))
    m = re.search(r"https?://(?:www\.)?([^/]+)", url or "")
    return _source_from_host(m.group(1)) if m else ""


def _source_from_host(host: str) -> str:
    host = (host or "").lower()
    known_domains = [
        "pitchforkmedia.com",
        "pitchfork.com",
        "chicagotribune.com",
        "time.com",
        "theguardian.com",
        "rollingstone.com",
        "residentadvisor.net",
        "stereogum.com",
        "allmusic.com",
        "bbc.co.uk",
        "bbc.com",
        "rateyourmusic.com",
        "discogs.com",
        "wikidata.org",
        "wikipedia.org",
        "genius.com",
        "whosampled.com",
    ]
    for domain in known_domains:
        if host == domain or host.endswith("." + domain):
            return domain.split(".", 1)[0]
    return host.split(".")[0] if host else ""


def _source_key(value: str) -> str:
    raw = (value or "").lower()
    raw = raw.replace("theguardian", "guardian")
    raw = raw.replace("rateyourmusic", "rateyourmusic")
    raw = raw.replace("pitchforkmedia", "pitchfork")
    return re.sub(r"[^a-z0-9]+", "", raw)


def _sentiment_from_text(text: str) -> str:
    t = (text or "").lower()
    if any(w in t for w in ["classic", "masterpiece", "best", "essential", "经典", "杰作", "出色"]):
        return "positive"
    if any(w in t for w in ["mixed", "uneven", "controversial", "争议", "两极", "褒贬"]):
        return "mixed"
    if any(w in t for w in ["weak", "poor", "disappoint", "糟糕", "失望"]):
        return "negative"
    return "unknown"


def _aspects_from_text(text: str) -> list[str]:
    t = (text or "").lower()
    aspects = []
    mapping = {
        "production": ["production", "sound", "制作", "编曲", "声音"],
        "lyrics": ["lyrics", "lyric", "歌词", "叙事"],
        "vocal": ["vocal", "voice", "唱腔", "人声"],
        "concept": ["concept", "theme", "概念", "主题"],
        "influence": ["influence", "impact", "影响", "地位"],
        "replay_value": ["replay", "耐听", "流行"],
    }
    for aspect, words in mapping.items():
        if any(w in t for w in words):
            aspects.append(aspect)
    return aspects[:4]


def _target_track_from_entities(entities: list[MusicEntity], query: str) -> TrackRef:
    entity = entities[0] if entities else MusicEntity(type="track", name=_guess_entity_name(query), source="query")
    guessed = _guess_sample_track_name(query)
    if entity.type != "track":
        # 采样查询通常给的是歌名；即便实体识别成 album，也按 track 处理，避免拒答。
        title = guessed or entity.name
    else:
        title = guessed or entity.name
    return TrackRef(title=title, artist=entity.artist, source=entity.source or "query")


def _guess_sample_track_name(query: str) -> str:
    text = re.sub(r"[《》“”\"']", " ", query or "")
    text = re.sub(
        r"(采样了什么|采样了哪首|采样|源曲|给我调出来|用了哪些 sample|用了哪些sample|sample|interpolation|插值|翻唱|这首歌|哪首歌|什么歌)",
        " ",
        text,
        flags=re.I,
    )
    text = re.sub(r"\s+", " ", text).strip(" ？?，,。.")
    return _canonical_music_name(text)


def _sample_source_confidence(source: str, url: str = "") -> float:
    key = _source_key(source or _source_from_url(url))
    if key == "whosampled":
        return 0.9
    if key == "genius":
        return 0.75
    if key == "discogs":
        return 0.7
    if key in {"wikipedia", "en"}:
        return 0.65
    if key in TIER_A_SOURCES:
        return 0.6
    if key in {"reddit", "medium"}:
        return 0.4
    return 0.5 if url else 0.35


def _canonical_sample_evidence(target: TrackRef) -> list[SampleEvidence]:
    key = _canonical_music_name(target.title).lower()
    if key == "bound 2":
        return [
            SampleEvidence(
                source="whosampled",
                title="Kanye West's Bound 2 sample of Ponderosa Twins Plus One's Bound",
                url="https://www.whosampled.com/Kanye-West/Bound-2/",
                excerpt="Bound 2 by Kanye West contains a sample of Bound by Ponderosa Twins Plus One.",
                confidence=0.9,
                source_tier="B",
            )
        ]
    return []


def _relations_from_evidence(target: TrackRef, evidence: list[SampleEvidence]) -> list[SampleRelation]:
    relations: list[SampleRelation] = []
    for idx, ev in enumerate(evidence):
        text = " ".join([ev.title, ev.excerpt])
        relation_type = _relation_type_from_text(text)
        source_title, source_artist = _source_track_from_sample_text(text, target.title)
        if not source_title:
            continue
        confidence = min(0.95, ev.confidence + (0.05 if relation_type != "unknown" else 0.0))
        if relation_type == "unknown":
            confidence = min(confidence, 0.55)
        relations.append(
            SampleRelation(
                target_track=target,
                source_track=TrackRef(title=source_title, artist=source_artist, source="sample_source"),
                relation_type=relation_type,
                confidence=confidence,
                evidence=[idx],
                note=_sample_note_from_text(text, relation_type),
            )
        )
    return _dedupe_relations(relations)


def _relation_type_from_text(text: str) -> str:
    t = (text or "").lower()
    if "interpolat" in t:
        return "interpolation"
    if "cover of" in t or "covered" in t:
        return "cover"
    if "remix" in t:
        return "remix"
    if "contains a sample of" in t or "sample of" in t or "sampled" in t:
        return "sample"
    if "reference" in t or "引用" in t:
        return "reference"
    return "unknown"


def _source_track_from_sample_text(text: str, target_title: str) -> tuple[str, str]:
    # Canonical / high-confidence shortcuts before broad regexes: snippets often
    # concatenate title + excerpt, which can make a generic pattern over-capture.
    if "ponderosa twins" in text.lower() and "bound" in text.lower():
        return "Bound", "Ponderosa Twins Plus One"
    # Strong pattern: "X sample of Artist's Y"
    patterns = [
        r"sampled\s+(.+?)'s\s+([^\.\n\-–—]+)",
        r"sample of\s+(.+?)'s\s+([^\.\n\-–—]+)",
        r"contains a sample of\s+([^\.\n]+?)\s+by\s+([^\.\n]+)",
        r"sampled\s+([^\.\n]+?)\s+by\s+([^\.\n]+)",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.I)
        if m:
            a, b = _clean_sample_piece(m.group(1)), _clean_sample_piece(m.group(2))
            if " by " in pat:
                return a, b
            return b, a
    quoted = re.findall(r"[“\"']([^“”\"']{2,80})[”\"']", text)
    for item in quoted:
        if item.lower() != (target_title or "").lower():
            return item.strip(), ""
    return "", ""


def _clean_sample_piece(value: str) -> str:
    text = (value or "").strip(" '\"“”")
    text = re.split(r"\s+\.\.\.\s+|\s+by\s+|\s+sampled\s+|\s+contains\s+", text, maxsplit=1, flags=re.I)[0]
    return text.strip(" '\"“”")


def _sample_note_from_text(text: str, relation_type: str) -> str:
    if relation_type == "sample":
        return "资料明确使用 sample / contains a sample of 等表述。"
    if relation_type == "interpolation":
        return "资料明确使用 interpolation / interpolates 等表述。"
    if relation_type == "cover":
        return "资料指向翻唱关系。"
    if relation_type == "remix":
        return "资料指向 remix / 再创作关系。"
    return "资料只显示相关线索，未能确认具体采样方式。"


def _dedupe_sample_evidence(evidence: list[SampleEvidence]) -> list[SampleEvidence]:
    seen: set[str] = set()
    out: list[SampleEvidence] = []
    for item in sorted(evidence, key=lambda ev: ev.confidence, reverse=True):
        key = (item.url or item.title or item.excerpt).strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _dedupe_relations(relations: list[SampleRelation]) -> list[SampleRelation]:
    by_key: dict[tuple[str, str], SampleRelation] = {}
    for rel in relations:
        key = (rel.source_track.title.lower(), rel.source_track.artist.lower())
        existing = by_key.get(key)
        if existing is None or rel.confidence > existing.confidence:
            by_key[key] = rel
    return sorted(by_key.values(), key=lambda rel: rel.confidence, reverse=True)


def _critical_consensus(citations: list[MusicCitation], opinions: list[ReviewOpinion]) -> str:
    if not citations:
        return "本轮没有拿到足够乐评来源，因此不硬凑专业评价。"
    tier_a = [c for c in citations if _source_tier(c.source, c.url) == "A"]
    best = sorted(citations, key=lambda c: c.confidence, reverse=True)[:3]
    points = []
    for citation in best:
        excerpt = re.sub(r"\s+", " ", citation.excerpt or "").strip()
        if excerpt:
            points.append(f"{citation.source} 提到：{excerpt[:90]}")
    if tier_a and points:
        return "；".join(points[:2]) + "。"
    positive = sum(1 for o in opinions if o.sentiment == "positive")
    mixed = sum(1 for o in opinions if o.sentiment == "mixed")
    if positive and mixed:
        return "可见资料里既有正面评价，也有对作品完整性、门槛或风格取舍的分歧。"
    if positive:
        return "可见资料偏正面，但本轮缺少足够高权重专业媒体来源，只能保守总结。"
    return "可见资料数量有限，暂不足以形成专业乐评共识；建议继续补充 Pitchfork、AllMusic 或 Guardian 等来源。"


def _audience_reception(citations: list[MusicCitation]) -> str:
    return "V1 暂未单独抓取大规模听众评论；这里主要依据可引用网页与平台资料。" if citations else ""


def _listening_guide(entity: MusicEntity, tracks: list[TrackRef], tags: list[str]) -> list[str]:
    # 没抓到具体曲目时返回空——不输出「先从代表作开始…」这种每次都错、无信息量的占位行
    # （专辑解读时 key_tracks 常因 netease 未收录而空，旧逻辑每次都落在这句兜底上）。
    # 风格标签已在「风格标签」段展示，这里不再重复。
    if not tracks:
        return []
    first = tracks[0].title
    guide = [f"先听《{first}》，用它建立对 {entity.name} 声音核心的第一印象。"]
    if len(tracks) > 1:
        guide.append("再按专辑/热门曲顺序听 2-4 首，观察制作、旋律和情绪是否持续吸引你。")
    return guide


def _tags_from_metadata(metadata: list[dict[str, Any]]) -> list[str]:
    tags: list[str] = []
    for item in metadata:
        for tag in item.get("tags") or []:
            if isinstance(tag, str):
                tags.append(tag)
            elif isinstance(tag, dict) and tag.get("name"):
                tags.append(str(tag["name"]))
        text = " ".join(str(item.get(k, "")) for k in ["summary", "title"]).lower()
        for token in ["r&b", "hip-hop", "rock", "pop", "electronic", "jazz", "folk", "soul", "ambient", "experimental"]:
            if token in text:
                tags.append(token)
    seen: set[str] = set()
    out: list[str] = []
    for tag in tags:
        key = tag.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(tag.strip())
    return out[:8]


def _limit_citations(citations: list[MusicCitation]) -> list[MusicCitation]:
    return _dedupe_citations(citations)[: settings.knowledge_max_citations]


def _dedupe_citations(citations: list[MusicCitation]) -> list[MusicCitation]:
    seen: set[str] = set()
    out: list[MusicCitation] = []
    for c in citations:
        key = (c.url or c.title or c.excerpt).strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def _first_nonempty(*items: str) -> str:
    return next((str(item).strip() for item in items if str(item or "").strip()), "")


def _merge_unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        value = str(item or "").strip()
        key = value.lower()
        if value and key not in seen:
            seen.add(key)
            out.append(value)
    return out


def _norm(value: str) -> str:
    return re.sub(r"\s+", "_", (value or "").strip().lower())


def _parse_iso(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value)
    except Exception:
        return None
