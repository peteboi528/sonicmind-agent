from __future__ import annotations

import logging
import re
import time
from datetime import UTC, datetime, timedelta
from typing import Any

from pydantic import BaseModel, Field

from app.config import settings
from app.concurrency import run_parallel
from app.llm.structured import extract_json_dict
from app.models import (
    KnowledgeEvidencePack,
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
    "resolve_music_entity", "music_metadata_lookup", "review_search", "build_music_dossier",
    "sample_relation_search", "locate_sample_sources", "build_sample_dossier",
}

TIER_A_SOURCES = {"pitchfork", "allmusic", "theguardian", "guardian", "rollingstone", "nme", "bbc", "residentadvisor", "stereogum", "musicbrainz"}
TIER_B_SOURCES = {"wikipedia", "lastfm", "last", "albumoftheyear", "rateyourmusic", "musicboard", "discogs", "genius", "whosampled"}


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


def cache_key(entity: MusicEntity) -> str:
    artist = _norm(entity.artist)
    name = _norm(entity.name)
    source = _norm(entity.source or "unknown")
    return re.sub(r"[^a-z0-9_\-]+", "_", f"{entity.type}:{name}:{artist}:{source}")[:160]


def read_cached_dossier(agent: Any, entity: MusicEntity) -> MusicDossier | None:
    try:
        item = agent.store.read_model("knowledge_cache", cache_key(entity), KnowledgeCacheItem)
    except Exception:
        return None
    if item is None:
        return None
    created = _parse_iso(item.created_at)
    if created and datetime.now(UTC) - created <= timedelta(hours=24):
        return item.dossier
    return None


def write_cached_dossier(agent: Any, dossier: MusicDossier) -> None:
    if dossier.partial:
        return
    try:
        key = cache_key(dossier.entity)
        agent.store.write_model("knowledge_cache", key, KnowledgeCacheItem(key=key, dossier=dossier))
    except Exception:
        return


def resolve_music_entities(query: str, intent: str, plan: dict[str, Any] | None = None) -> list[MusicEntity]:
    query = (query or "").strip()
    plan = plan or {}
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
        names = planned_entities[:2 if intent == "music_compare" else 1]
    if not names:
        names = [_guess_entity_name(query)]
    return [
        MusicEntity(type=entity_type, name=name, artist=_infer_artist(query, name, entity_type), source="query")
        for name in names if name
    ][:2 if intent == "music_compare" else 1]


def _structured_entity_from_query(query: str, intent: str) -> MusicEntity | None:
    """Parse compact field-style UI input such as:

    album
    Blonde on Blonde
    West Norwood Cassette Library

    The knowledge agent must preserve the provided artist for disambiguation;
    otherwise same-title albums can be canonicalized to a famous but wrong work.
    """
    lines = [
        re.sub(r"^[\s:：\-•]+|[\s:：]+$", "", line).strip()
        for line in (query or "").splitlines()
        if line.strip()
    ]
    if len(lines) < 2:
        return None

    first = lines[0].lower().strip()
    kind_aliases = {
        "album": "album", "专辑": "album", "唱片": "album", "release": "album",
        "track": "track", "song": "track", "歌曲": "track", "单曲": "track",
        "artist": "artist", "艺人": "artist", "歌手": "artist", "乐队": "artist",
    }
    entity_type = kind_aliases.get(first)
    if entity_type is None and len(lines) == 2 and intent in {"album_deep_dive", "review_summary", "sample_lookup"}:
        inferred = "track" if intent == "sample_lookup" else "album"
        name = re.sub(r"^(?:title|name|标题|名称)\s*[:：]\s*", "", lines[0], flags=re.I).strip()
        artist = re.sub(r"^(?:artist|艺人|歌手|乐队)\s*[:：]\s*", "", lines[1], flags=re.I).strip()
        if name and artist and name.lower() != artist.lower():
            return MusicEntity(
                type=inferred,
                name=_canonical_music_name(name),
                artist=artist,
                source="query",
            )
    if entity_type is None:
        return None

    name = _canonical_music_name(lines[1])
    artist = ""
    if entity_type in {"album", "track"} and len(lines) >= 3:
        artist = lines[2].strip()
    # If a UI includes field labels on following lines, strip the most common ones.
    name = re.sub(r"^(?:title|name|标题|名称)\s*[:：]\s*", "", name, flags=re.I).strip()
    artist = re.sub(r"^(?:artist|艺人|歌手|乐队)\s*[:：]\s*", "", artist, flags=re.I).strip()
    if not name:
        return None
    return MusicEntity(type=entity_type, name=name, artist=artist, source="query")


def _explicit_artist_entity_from_query(query: str, entity_type: str, intent: str) -> MusicEntity | None:
    if intent == "music_compare" or entity_type not in {"album", "track"}:
        return None
    text = re.sub(r"[《》“”\"']", " ", query or "")
    text = re.sub(
        r"(讲讲|解读|为什么经典|为什么|乐评怎么说|评价如何|评价|介绍|请|帮我|搜索|查一下|这张专辑|这首歌)",
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


def canonicalize_entities(entities: list[MusicEntity], deadline_at: float | None = None) -> list[MusicEntity]:
    """消歧阶段：用 MusicBrainz 把裸名/裸标题钉成权威 (name, artist, type)。

    这是知识链路的「消歧」职责所在——在 resolve 阶段一次性钉准实体，下游 metadata/review
    全部继承，避免各源(MB/Spotify/Discogs)各自对裸标题模糊匹配出三个不同的同名作品
    (Blonde 实测被解析成 Frank Ocean / Bob Dylan《Blonde on Blonde》/ West Norwood 三个)。

    album 类型且缺 artist 时收益最大：MB 回填权威艺人后，Spotify/Discogs 带着 artist 检索
    即可收敛。失败/超时/关闭 MB 时原样返回，绝不报错（保持 offline 测试与降级契约）。
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
                hit = client.resolve_artist(entity.name)
                if hit and hit.get("name"):
                    entity.name = hit["name"]
                    if hit.get("mbid"):
                        entity.external_ids.setdefault("musicbrainz", hit["mbid"])
            elif entity.type in {"album", "track"}:
                hit = client.resolve_release_group(entity.name, entity.artist)
                if hit and hit.get("title"):
                    entity.name = hit["title"]
                    if hit.get("artist") and not entity.artist:
                        entity.artist = hit["artist"]
                    if hit.get("mbid"):
                        entity.external_ids.setdefault("musicbrainz", hit["mbid"])
    except Exception:
        logger.debug("canonicalize_entities 失败，按原实体降级", exc_info=True)
    return entities


def lookup_metadata(agent: Any, entities: list[MusicEntity], deadline_at: float | None = None) -> dict[str, Any]:
    remaining = remaining_seconds(deadline_at)
    if remaining is not None and remaining < 1.0:
        return {"metadata": [], "citations": [], "tracks": [], "albums": [], "skipped_due_to_deadline": ["music_metadata_lookup"]}
    # 元数据现在是单波并行（MB/Spotify/Discogs/netease/web 同时发），墙钟≈最慢单源，
    # 不再是各源之和——所以这一波该拿到比旧串行模型(3s)更宽的预算，否则慢源(Spotify
    # OAuth/Discogs/web)永远跑不完，只剩最快的 netease 活下来(实测"来源 netease only")。
    # 取 source_timeout 与「剩余预算留给乐评的余量」中较大者，但不超过剩余预算。
    floor = settings.knowledge_source_timeout_seconds
    if remaining is not None:
        # 给后续 review_search 留 review_timeout 余量，其余尽量给元数据波。
        budget_for_meta = max(floor, remaining - settings.knowledge_review_timeout_seconds - 0.5)
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


def search_reviews(entities: list[MusicEntity], deadline_at: float | None = None) -> dict[str, Any]:
    remaining = remaining_seconds(deadline_at)
    if remaining is not None and remaining < 1.0:
        return {"citations": [], "opinions": [], "skipped_due_to_deadline": ["review_search"]}
    timeout = min(settings.knowledge_review_timeout_seconds, remaining or settings.knowledge_review_timeout_seconds)
    queries: list[str] = []
    for entity in entities:
        base = " ".join(part for part in [entity.artist, entity.name] if part).strip() or entity.name
        queries.extend([
            f"{base} Pitchfork review",
            f"{base} AllMusic review",
            f"{base} Guardian review",
            f"{base} critical reception",
            f"{base} 乐评 专辑 评价",
        ])
    deduped = []
    seen: set[str] = set()
    for q in queries:
        key = q.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(q)
    queries = deduped[:settings.knowledge_max_search_queries]
    # Leave a small margin for parsing/ToolRuntime bookkeeping.  Otherwise the
    # outer wait_for may cancel the whole handler at exactly the same wall-clock
    # boundary and report a misleading "skipped due to deadline" even though the
    # search was actually attempted.
    batch_timeout = max(0.5, timeout - 0.3)
    request_timeout = max(0.5, batch_timeout)
    tasks = [
        (f"review:{q}", lambda q=q: web_search_source.search_web_info(
            q,
            max_results=max(2, settings.knowledge_max_review_sources),
            api_key=settings.tavily_api_key,
            timeout=request_timeout,
        ))
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
            citations.append(MusicCitation(
                source=source,
                title=title,
                url=url,
                kind="review",
                excerpt=excerpt[:500],
                confidence=_source_confidence(source, url),
            ))
    citations = sorted(
        _dedupe_citations(citations),
        key=lambda item: item.confidence,
        reverse=True,
    )[:settings.knowledge_max_review_sources]
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


def search_sample_relations(entities: list[MusicEntity], query: str, deadline_at: float | None = None) -> dict[str, Any]:
    target = _target_track_from_entities(entities, query)
    remaining = remaining_seconds(deadline_at)
    if remaining is not None and remaining < 1.0:
        return {"target": target.model_dump(mode="json"), "evidence": [], "skipped_due_to_deadline": ["sample_relation_search"]}
    timeout = min(settings.knowledge_review_timeout_seconds, remaining or settings.knowledge_review_timeout_seconds)
    base = " ".join(part for part in [target.artist, target.title] if part).strip() or target.title
    queries = [
        f"{base} WhoSampled",
        f"{base} sampled what song",
        f"{base} Genius sample interpolation",
        f"{base} Discogs sample credits",
    ][:settings.knowledge_max_search_queries]
    tasks = [
        (f"sample:{q}", lambda q=q: web_search_source.search_web_info(
            q, max_results=3, api_key=settings.tavily_api_key,
        ))
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
            evidence.append(SampleEvidence(
                source=source,
                title=title,
                url=url,
                excerpt=excerpt[:500],
                confidence=_sample_source_confidence(source, url),
                source_tier=_source_tier(source, url),
            ))
    evidence = _dedupe_sample_evidence([*_canonical_sample_evidence(target), *evidence])[:6]
    return {"target": target.model_dump(mode="json"), "evidence": [e.model_dump(mode="json") for e in evidence]}


def locate_sample_sources(agent: Any, target: TrackRef, evidence: list[SampleEvidence], deadline_at: float | None = None) -> dict[str, Any]:
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
            cards.append({
                "title": relation.source_track.title,
                "artist": relation.source_track.artist,
                "source": relation.source_track.source or "sample_source",
                "source_id": relation.source_track.source_id,
            })
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
        "ambient", "electronic", "guitar-based", "minimal", "dense", "art rock",
        "alternative rock", "r&b", "neo-soul", "krautrock", "jazz", "experimental",
    ]
    theme_terms = [
        "alienation", "technology", "anxiety", "memory", "identity", "intimacy",
        "consumer", "modernity", "isolation", "dread", "nostalgia",
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
    critic_points = [
        c.excerpt[:180] for c in sorted(citations, key=lambda c: c.confidence, reverse=True)
        if c.excerpt
    ][:5]
    return KnowledgeEvidencePack(
        facts=[t[:180] for t in texts if t][:4],
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
) -> dict[str, str] | None:
    """把零散（多为英文）证据交给 LLM 翻译+总结成连贯中文。

    防幻觉铁律：只允许基于给定证据改写/翻译/压缩，禁止补充证据外的事实。证据不足时
    宁可返回 None 让上层走机械兜底，也不硬编。严格 JSON 输出——解析失败即视为不可用，
    保证 MockLLM（离线测试）与任何非 JSON 回复都安全回落到机械摘要，输出确定。

    返回 {"summary": ..., "critical_consensus": ...}，或 None（无 LLM / 无证据 / 无预算 / 解析失败）。
    """
    llm = getattr(agent, "llm", None)
    if llm is None or not hasattr(llm, "generate"):
        return None
    # 没有任何可总结的证据就不调 LLM（省延迟，也避免"无中生有"）。
    facts = [f for f in evidence_pack.facts if f.strip()]
    critic_points = [c for c in evidence_pack.critic_points if c.strip()]
    if not meta_text and not facts and not critic_points:
        return None
    # 预算闸：合成是锦上添花，剩余时间不足 2.5s 直接放弃走机械兜底，守住知识链硬预算。
    remaining = remaining_seconds(deadline_at)
    if remaining is not None and remaining < 2.5:
        return None

    evidence_lines: list[str] = []
    if meta_text:
        evidence_lines.append(f"- 背景资料：{meta_text[:600]}")
    for f in facts[:4]:
        evidence_lines.append(f"- 事实：{f[:200]}")
    for c in critic_points[:5]:
        evidence_lines.append(f"- 乐评摘录：{c[:200]}")
    for o in opinions[:4]:
        if o.summary.strip():
            evidence_lines.append(f"- 评价（{o.source}/{o.sentiment}）：{o.summary[:160]}")
    if style_tags:
        evidence_lines.append(f"- 风格标签：{'、'.join(style_tags[:8])}")
    has_reviews = bool(review_citations or critic_points)
    evidence_block = "\n".join(evidence_lines)

    system = (
        "你是严谨的音乐资料编辑。只能依据【证据】改写、翻译、压缩成连贯中文，"
        "严禁补充证据里没有的事实、评分、年份或人物。证据是英文就翻译成自然中文。"
    )
    consensus_rule = (
        "把多条乐评摘录归纳成 1-2 句中文共识，点明评价的侧重与分歧；不要逐条罗列、不要保留英文残句。"
        if has_reviews
        else "证据里没有足够乐评，critical_consensus 必须为空字符串，不要编造专业评价。"
    )
    prompt = (
        f"实体：{entity.type} 《{entity.name}》" + (f"，艺人 {entity.artist}" if entity.artist else "") + "\n\n"
        f"【证据】\n{evidence_block}\n\n"
        "请输出严格 JSON（不要代码块、不要多余文字），字段：\n"
        '  "summary": 2-3 句中文介绍这位艺人/这张专辑的核心信息（风格、定位、背景），只用证据里的内容。\n'
        '  "critical_consensus": ' + consensus_rule + "\n"
        "示例：{\"summary\": \"...\", \"critical_consensus\": \"...\"}"
    )
    try:
        raw = llm.generate(prompt, system=system, temperature=0.3)
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
    return {"summary": summary[:400], "critical_consensus": consensus[:400]}


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
) -> MusicDossier:
    entity = entities[0] if entities else MusicEntity(type=_infer_entity_type(query, intent), name=_guess_entity_name(query), source="query")
    cached = read_cached_dossier(agent, entity)
    if cached and not skipped:
        return cached.model_copy(update={"partial": False, "degraded_reason": None})
    citations = _limit_citations([*metadata_citations, *review_citations])
    meta_text = _first_nonempty(*(m.get("summary", "") for m in metadata), *(m.get("bio", "") for m in metadata))
    evidence_pack = build_evidence_pack(metadata, review_citations, opinions)
    style_tags = _merge_unique([*_tags_from_metadata(metadata), *evidence_pack.sound_descriptors])[:8]
    partial_reasons: list[str] = []
    skipped = skipped or []
    timed_out = timed_out or []
    if skipped:
        partial_reasons.append("部分知识工具因时间预算不足被跳过：" + "、".join(sorted(set(skipped))))
    if timed_out:
        partial_reasons.append("部分知识工具在本轮时间内未返回，已超时降级：" + "、".join(sorted(set(timed_out))))
    if not review_citations:
        partial_reasons.append("乐评来源本轮未在时间预算内取回足够结果")
    if not citations:
        partial_reasons.append("外部资料来源不足，无法做完整乐评总结")
    if remaining_seconds(deadline_at) is not None and remaining_seconds(deadline_at) <= 0:
        partial_reasons.append(f"本轮达到 {int(settings.knowledge_turn_budget_seconds)} 秒知识链路预算")

    guide = _listening_guide(entity, tracks, style_tags)
    if intent == "music_compare" and len(entities) >= 2:
        summary = _compare_summary(entities[0], entities[1])
        related = [entities[1]]
        consensus = _critical_consensus(review_citations, opinions)
    else:
        related = entities[1:]
        # 合成层：把零散英文证据(MB/Spotify/Discogs/乐评摘录)交给 LLM 翻译+总结成
        # 连贯中文 summary/乐评共识，治"原始摘录直出、半句英文"。失败/无预算/无证据时
        # 回落机械摘要，保证 offline(MockLLM)与降级路径确定可用。
        synth = _synthesize_dossier_prose(
            agent, entity, meta_text, style_tags, evidence_pack, review_citations, opinions, deadline_at,
        )
        if synth:
            summary = synth.get("summary") or (meta_text[:220] if meta_text else f"我整理了 {entity.name} 的可追溯音乐资料。")
            consensus = synth.get("critical_consensus") or _critical_consensus(review_citations, opinions)
        else:
            summary = meta_text[:220] if meta_text else f"我整理了 {entity.name} 的可追溯音乐资料。"
            consensus = _critical_consensus(review_citations, opinions)
    dossier = MusicDossier(
        entity=entity,
        summary=summary,
        background=meta_text[:800] if meta_text else "本轮没有拿到足够稳定的背景资料。",
        style_tags=style_tags,
        critical_consensus=consensus,
        audience_reception=_audience_reception(review_citations),
        key_tracks=tracks[:8],
        listening_guide=guide,
        related_albums=(albums or [])[:6],
        related_entities=related,
        citations=citations,
        review_opinions=opinions[:settings.knowledge_max_review_sources],
        uncertainties=[*partial_reasons, *evidence_pack.disagreements[:2]][:4],
        partial=bool(partial_reasons),
        degraded_reason="；".join(partial_reasons) if partial_reasons else None,
    )
    write_cached_dossier(agent, dossier)
    return dossier


def dossier_answer(dossier: MusicDossier) -> str:
    if dossier.related_entities:
        other = dossier.related_entities[0]
        lines = [f"{dossier.entity.name} 和 {other.name} 的区别：{dossier.summary}"]
        if dossier.partial and dossier.degraded_reason:
            lines.append(f"\n资料状态：{dossier.degraded_reason}。")
        lines.extend(_compare_detail_lines(dossier.entity, other))
        if dossier.key_tracks:
            names = "、".join(f"《{t.title}》" for t in dossier.key_tracks[:5] if t.title)
            if names:
                lines.append("\n本轮抓到的可听入口：" + names)
        if dossier.citations:
            lines.append("\n参考来源：")
            for c in dossier.citations[:3]:
                label = c.title or c.source
                lines.append(f"- {label}：{c.url}" if c.url else f"- {label}")
        return "\n".join(lines)
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
            hit = client.resolve_artist(entity.name)
            if not hit or not hit.get("name"):
                return None
            summary_bits = [b for b in (hit.get("type"), f"来自 {hit['country']}" if hit.get("country") else "", hit.get("disambiguation", "")) if b]
            return {
                "source": "musicbrainz",
                "canonical_name": hit.get("name", ""),
                "mbid": hit.get("mbid", ""),
                "tags": hit.get("tags") or [],
                "summary": "，".join(summary_bits),
            }
        # album / track 都按专辑 release-group 查（track 粒度命中率低，先用专辑兜）。
        hit = client.resolve_release_group(entity.name, entity.artist)
        if not hit or not hit.get("title"):
            return None
        summary_bits = [b for b in (
            f"艺人 {hit['artist']}" if hit.get("artist") else "",
            f"发行 {hit['date']}" if hit.get("date") else "",
            hit.get("type", ""),
        ) if b]
        return {
            "source": "musicbrainz",
            "canonical_name": hit.get("title", ""),
            "mbid": hit.get("mbid", ""),
            "artist": hit.get("artist", ""),
            "date": hit.get("date", ""),
            "type": hit.get("type", ""),
            "tags": hit.get("tags") or [],
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
            citations.append(MusicCitation(
                source="musicbrainz", title=f"{entity.name} - MusicBrainz",
                url=f"https://musicbrainz.org/{path}/{mbid}" if mbid else "",
                kind="encyclopedia", excerpt=mb.get("summary", ""), confidence=_source_confidence("musicbrainz"),
            ))
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
            citations.append(MusicCitation(
                source="spotify", title=f"{entity.name} - Spotify",
                url=f"https://open.spotify.com/{sp_path}/{sp_id}" if sp_id else "",
                kind="platform", excerpt=sp.get("summary", ""), confidence=0.72,
            ))
    if dc:
        if dc.get("external_id"):
            entity.external_ids["discogs"] = dc["external_id"]
        tags = dc.get("styles") or dc.get("genres") or []
        if tags or dc.get("summary"):
            dc_id = dc.get("external_id", "")
            dc_path = dc.get("type") or "master"
            metadata.append({"entity": entity.model_dump(mode="json"), "summary": dc.get("summary", ""), "tags": tags})
            citations.append(MusicCitation(
                source="discogs", title=f"{entity.name} - Discogs",
                url=f"https://www.discogs.com/{dc_path}/{dc_id}" if dc_id else "",
                kind="encyclopedia", excerpt=dc.get("summary", ""), confidence=_source_confidence("discogs"),
            ))


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
    res = dict(zip([label for label, _ in tasks], results))

    try:
        # 1) 结构化权威源合流（顺序：MB 纠正实体名 → Spotify 补声音/封面 → Discogs 补细类）。
        _apply_structured_sources(
            entity, [res.get("musicbrainz"), res.get("spotify"), res.get("discogs")], metadata, citations,
        )
        # 2) 网易云艺人专辑。
        albums = res.get("netease_albums") or []
        for album in albums:
            citations.append(MusicCitation(
                source="netease", title=album.get("name", ""), url="", kind="platform",
                excerpt=f"网易云专辑结果：{album.get('name', '')}", confidence=0.8,
            ))
        # 3) Last.fm 简介 + 热门曲。
        lf = res.get("lastfm")
        if lf and lf.get("info"):
            info = lf["info"]
            metadata.append({
                "entity": entity.model_dump(mode="json"),
                "summary": info.get("bio", ""),
                "tags": info.get("tags", []),
                "image": info.get("image", ""),
            })
            citations.append(MusicCitation(
                source="lastfm", title=f"{entity.name} - Last.fm", url="", kind="metadata",
                excerpt=(info.get("bio") or "")[:500], confidence=0.7,
            ))
            for t in lf.get("top") or []:
                tracks.append(TrackRef(title=t.get("title", ""), artist=t.get("artist") or entity.name, source="lastfm"))
        # 4) 网易云专辑元数据 + 曲目（合流时再回写 entity，线程安全）。
        nb = res.get("netease_album")
        if nb:
            album = nb["album"]
            entity.external_ids["netease_album"] = str(album.get("id") or "")
            entity.image = album.get("cover", "") or entity.image
            entity.artist = album.get("artist", "") or entity.artist
            metadata.append({"entity": entity.model_dump(mode="json"), "summary": f"网易云识别到专辑《{album.get('name')}》，艺人 {album.get('artist') or '未知'}。"})
            citations.append(MusicCitation(source="netease", title=album.get("name", ""), kind="platform", excerpt="网易云专辑元数据", confidence=0.85))
            for item in (nb.get("detail") or {}).get("tracks", [])[:8]:
                tracks.append(TrackRef(
                    title=item.get("title", ""), artist=item.get("artist", "") or entity.artist,
                    source="netease", source_id=str(item.get("song_id") or ""),
                ))
        # 5) Web 背景资料。
        for item in (res.get("web") or [])[:2]:
            metadata.append({"entity": entity.model_dump(mode="json"), "summary": item.get("content", ""), "title": item.get("title", ""), "url": item.get("url", "")})
            citations.append(MusicCitation(
                source=_source_from_url(item.get("url", "")) or "web",
                title=item.get("title", ""), url=item.get("url", ""), kind="encyclopedia",
                excerpt=(item.get("content") or "")[:500], confidence=0.65,
            ))
    except Exception:
        pass
    return {"metadata": metadata, "citations": citations, "tracks": tracks, "albums": albums}


def _guess_entity_name(query: str) -> str:
    text = re.sub(r"[《》“”\"']", " ", query or "")
    text = re.sub(r"(讲讲|解读|为什么经典|为什么|乐评怎么说|评价如何|评价|区别在哪|有什么区别|区别|不同|介绍|系统|音乐路线|这张专辑|这首歌|这个艺人|这个歌手|专辑|歌手|艺人|是什么|怎么样|如何|请|帮我)", " ", text, flags=re.I)
    text = re.sub(r"的\s*$", "", text.strip())
    text = re.sub(r"\s+", " ", text).strip(" ？?，,。.")
    return _canonical_music_name(text or (query or "未知音乐实体").strip())


def _compare_names(query: str) -> list[str]:
    raw = re.sub(r"[《》“”\"']", " ", query or "")
    parts = re.split(r"\s+(?:vs|VS|Vs)\s+|\s+和\s+|和|对比|比较", raw)
    names = [_guess_entity_name(part) for part in parts]
    return [n for n in names if n][:2]


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


def _infer_entity_type(query: str, intent: str) -> str:
    q = (query or "").lower()
    if intent == "sample_lookup":
        return "track"
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
    return (m.group(1).strip() if m else "")


def _source_from_url(url: str) -> str:
    m = re.search(r"https?://(?:www\.)?([^/]+)", url or "")
    return m.group(1).split(".")[0] if m else ""


def _source_key(value: str) -> str:
    raw = (value or "").lower()
    raw = raw.replace("theguardian", "guardian")
    raw = raw.replace("rateyourmusic", "rateyourmusic")
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
    text = re.sub(r"(采样了什么|采样了哪首|采样|源曲|给我调出来|用了哪些 sample|用了哪些sample|sample|interpolation|插值|翻唱|这首歌|哪首歌|什么歌)", " ", text, flags=re.I)
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
        return [SampleEvidence(
            source="whosampled",
            title="Kanye West's Bound 2 sample of Ponderosa Twins Plus One's Bound",
            url="https://www.whosampled.com/Kanye-West/Bound-2/",
            excerpt="Bound 2 by Kanye West contains a sample of Bound by Ponderosa Twins Plus One.",
            confidence=0.9,
            source_tier="B",
        )]
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
        relations.append(SampleRelation(
            target_track=target,
            source_track=TrackRef(title=source_title, artist=source_artist, source="sample_source"),
            relation_type=relation_type,
            confidence=confidence,
            evidence=[idx],
            note=_sample_note_from_text(text, relation_type),
        ))
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
    if tracks:
        first = tracks[0].title
        guide = [f"先听《{first}》，用它建立对 {entity.name} 声音核心的第一印象。"]
        if len(tracks) > 1:
            guide.append("再按专辑/热门曲顺序听 2-4 首，观察制作、旋律和情绪是否持续吸引你。")
    else:
        guide = [f"先从 {entity.name} 的代表作品或最高频被讨论作品开始，不要一次性补完整目录。"]
    if tags:
        guide.append("听的时候重点留意：" + "、".join(tags[:3]))
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
    return _dedupe_citations(citations)[:settings.knowledge_max_citations]


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
