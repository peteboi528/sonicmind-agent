from __future__ import annotations

import re
import time
from datetime import UTC, datetime, timedelta
from typing import Any

from pydantic import BaseModel, Field

from app.config import settings
from app.concurrency import run_parallel
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

KNOWLEDGE_INTENTS = {"album_deep_dive", "artist_deep_dive", "review_summary", "music_compare", "sample_lookup"}
KNOWLEDGE_TOOLS = {
    "resolve_music_entity", "music_metadata_lookup", "review_search", "build_music_dossier",
    "sample_relation_search", "locate_sample_sources", "build_sample_dossier",
}

TIER_A_SOURCES = {"pitchfork", "allmusic", "theguardian", "guardian", "rollingstone", "nme", "bbc", "residentadvisor", "stereogum"}
TIER_B_SOURCES = {"wikipedia", "musicbrainz", "lastfm", "last", "albumoftheyear", "rateyourmusic", "musicboard", "discogs", "genius", "whosampled"}


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
    retrieval = plan.get("retrieval_plan") or {}
    planned_entities = [str(e).strip() for e in (retrieval.get("entities") or []) if str(e).strip()]
    names = _compare_names(query) if intent == "music_compare" else []
    if not names and planned_entities:
        names = planned_entities[:2 if intent == "music_compare" else 1]
    if not names:
        names = [_guess_entity_name(query)]
    entity_type = _infer_entity_type(query, intent)
    return [
        MusicEntity(type=entity_type, name=name, artist=_infer_artist(query, name, entity_type), source="query")
        for name in names if name
    ][:2 if intent == "music_compare" else 1]


def lookup_metadata(agent: Any, entities: list[MusicEntity], deadline_at: float | None = None) -> dict[str, Any]:
    remaining = remaining_seconds(deadline_at)
    if remaining is not None and remaining < 1.0:
        return {"metadata": [], "citations": [], "tracks": [], "albums": [], "skipped_due_to_deadline": ["music_metadata_lookup"]}
    timeout = min(settings.knowledge_source_timeout_seconds, remaining or settings.knowledge_source_timeout_seconds)
    tasks: list[tuple[str, Any]] = []
    for entity in entities:
        tasks.append((f"metadata:{entity.name}", lambda e=entity: _metadata_for_entity(agent, e)))
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
    tasks = [
        (f"review:{q}", lambda q=q: web_search_source.search_web_info(
            q, max_results=max(2, settings.knowledge_max_review_sources), api_key=settings.tavily_api_key,
        ))
        for q in queries
    ]
    batches = run_parallel(tasks, timeout=max(0.2, timeout), default=[])
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
    citations = _dedupe_citations(citations)[:settings.knowledge_max_review_sources]
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
    if skipped:
        partial_reasons.append("部分知识工具因时间预算不足被跳过：" + "、".join(sorted(set(skipped))))
    if not review_citations:
        partial_reasons.append("乐评来源本轮未在时间预算内取回足够结果")
    if not citations:
        partial_reasons.append("外部资料来源不足，无法做完整乐评总结")
    if remaining_seconds(deadline_at) is not None and remaining_seconds(deadline_at) <= 0:
        partial_reasons.append("本轮达到 12 秒知识链路预算")

    guide = _listening_guide(entity, tracks, style_tags)
    if intent == "music_compare" and len(entities) >= 2:
        summary = _compare_summary(entities[0], entities[1])
        related = [entities[1]]
    else:
        summary = meta_text[:220] if meta_text else f"我整理了 {entity.name} 的可追溯音乐资料。"
        related = entities[1:]
    dossier = MusicDossier(
        entity=entity,
        summary=summary,
        background=meta_text[:800] if meta_text else "本轮没有拿到足够稳定的背景资料。",
        style_tags=style_tags,
        critical_consensus=_critical_consensus(review_citations, opinions),
        audience_reception=_audience_reception(review_citations),
        key_tracks=tracks[:8],
        listening_guide=guide,
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


def _metadata_for_entity(agent: Any, entity: MusicEntity) -> dict[str, Any]:
    citations: list[MusicCitation] = []
    metadata: list[dict[str, Any]] = []
    tracks: list[TrackRef] = []
    albums: list[dict[str, Any]] = []
    try:
        if entity.type == "artist":
            albums = agent.recommend_artist_albums(entity.name, limit=4)
            for album in albums:
                citations.append(MusicCitation(
                    source="netease", title=album.get("name", ""), url="", kind="platform",
                    excerpt=f"网易云专辑结果：{album.get('name', '')}", confidence=0.8,
                ))
            if getattr(settings, "lastfm_api_key", ""):
                from app.sources.lastfm_client import LastfmClient

                info = LastfmClient(settings.lastfm_api_key).get_artist_info(entity.name)
                if info:
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
                    for t in LastfmClient(settings.lastfm_api_key).get_artist_top_tracks(entity.name, 6):
                        tracks.append(TrackRef(title=t.get("title", ""), artist=t.get("artist") or entity.name, source="lastfm"))
        elif entity.type == "album":
            from app.sources import netease as netease_source

            album = netease_source.search_netease_album(entity.artist, entity.name)
            if album:
                entity.external_ids["netease_album"] = str(album.get("id") or "")
                entity.image = album.get("cover", "") or entity.image
                entity.artist = album.get("artist", "") or entity.artist
                metadata.append({"entity": entity.model_dump(mode="json"), "summary": f"网易云识别到专辑《{album.get('name')}》，艺人 {album.get('artist') or '未知'}。"})
                citations.append(MusicCitation(source="netease", title=album.get("name", ""), kind="platform", excerpt="网易云专辑元数据", confidence=0.85))
                detail = netease_source.fetch_netease_album_tracks(str(album.get("id") or ""), 12)
                for item in (detail or {}).get("tracks", [])[:8]:
                    tracks.append(TrackRef(
                        title=item.get("title", ""), artist=item.get("artist", "") or entity.artist,
                        source="netease", source_id=str(item.get("song_id") or ""),
                    ))
        query = " ".join(part for part in [entity.artist, entity.name, "music background"] if part)
        web = web_search_source.search_web_info(query, max_results=2, api_key=settings.tavily_api_key)
        for item in web[:2]:
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
    if entity.type != "track":
        # 采样查询通常给的是歌名；即便实体识别成 album，也按 track 处理，避免拒答。
        title = _guess_sample_track_name(query) or entity.name
    else:
        title = entity.name
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
