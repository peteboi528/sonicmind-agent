from __future__ import annotations

from typing import Any

from app.answer import song_card
from app.intents import expand_content_negation, normalize_content_negation
from app.tools.actions import AUX_TOOL_NAMES, execute_aux_tool
from app.tools.contracts import ToolContext, ToolResult, ToolStatus
from app.tools.registry import bind_async_tool_handler, bind_tool_handler


def _aux_handler(name: str):
    def execute(arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        result, summary = execute_aux_tool(
            context.agent,
            name,
            context.user_id,
            arguments,
            query=context.query,
            prior_tracks=_collect_tracks(context.prior_results),
        )
        status = _status_for(result)
        tracks = result.get("tracks") or []
        provenance = [
            {"source": getattr(track, "source", "unknown"), "source_id": getattr(track, "external_id", "")}
            for track in tracks
        ]
        return ToolResult(
            tool=name,
            status=status,
            data=result,
            summary=summary,
            cards=[song_card(track) for track in tracks],
            provenance=provenance,
        )

    return execute


def _collect_tracks(results: list[dict[str, Any]]) -> list[Any]:
    from app.answer import collect_tracks

    return collect_tracks(results)


def _status_for(result: dict[str, Any]) -> ToolStatus:
    result_type = result.get("type")
    if result_type == "auth_required":
        return ToolStatus.AUTH_REQUIRED
    if result_type == "confirmation_required":
        return ToolStatus.CONFIRMATION_REQUIRED
    if result_type == "unsupported_write":
        return ToolStatus.UNSUPPORTED
    if not result.get("tracks") and result_type in {"find_on_platform", "lyrics", "concert_events"}:
        return ToolStatus.EMPTY
    return ToolStatus.OK


def _filter_excluded(tracks: list[Any], excluded: list[dict[str, str]]) -> list[Any]:
    """过滤掉上一轮已展示给用户的歌曲，避免延续指令时推荐重复曲目。

    匹配策略：(title, source_id) 组合键，source_id 为空时退化为 title 匹配。
    """
    if not excluded:
        return tracks
    seen_keys: set[tuple[str, str]] = set()
    seen_titles: set[str] = set()
    for ex in excluded:
        title = ex.get("title", "").lower().strip()
        sid = ex.get("source_id", "").strip()
        if title:
            seen_titles.add(title)
            if sid:
                seen_keys.add((title, sid))
    filtered = []
    for t in tracks:
        t_title = (getattr(t, "title", "") or "").lower().strip()
        t_sid = getattr(t, "external_id", "") or getattr(t, "asset_id", "") or ""
        if t_title and t_sid and (t_title, t_sid) in seen_keys:
            continue
        if t_title and t_title in seen_titles:
            continue
        filtered.append(t)
    return filtered


def _apply_language_filter(tracks: list[Any], language_filter: str, target: int) -> list[Any]:
    """按语言偏好对候选做安全后过滤（detect_language 仅判 zh/en）。

    仅在 zh/en 两种可判定语言上过滤；过滤后候选太少（<目标一半）时回退不过滤，
    避免删空。LLM 已把语言需求转进 search_query 做正向检索，这里只是兜底纠偏。
    """
    if language_filter not in {"zh", "en"} or not tracks:
        return tracks
    from app.recommend.rerank import detect_language

    kept = [t for t in tracks if detect_language(t) == language_filter]
    if len(kept) < max(1, target // 2):
        return tracks
    return kept


def _filter_content_exclusions(tracks: list[Any], exclusions: list[str]) -> list[Any]:
    """Hard-filter rewritten negative constraints from every result-producing path."""
    if not tracks or not exclusions:
        return tracks
    import re

    from app.recommend.rerank import detect_language

    canonical = {normalize_content_negation(item) for item in exclusions if item.strip()}
    aliases = {
        alias.lower()
        for item in canonical
        for alias in expand_content_negation(item)
        if alias.strip()
    }

    def blocked(track: Any) -> bool:
        text = " ".join([
            str(getattr(track, "title", "") or ""),
            str(getattr(track, "artist", "") or ""),
            str(getattr(track, "album", "") or ""),
            *[str(item) for item in getattr(track, "genre", []) or []],
            *[str(item) for item in getattr(track, "mood", []) or []],
        ]).lower()
        if any(alias in text for alias in aliases):
            return True
        if "中文" in canonical and detect_language(track) == "zh":
            return True
        if "英文" in canonical and detect_language(track) == "en":
            return True
        if "日语" in canonical and re.search(r"[\u3040-\u30ff]", text):
            return True
        if "韩语" in canonical and re.search(r"[\uac00-\ud7af]", text):
            return True
        if "越南" in canonical and re.search(r"[ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ]", text):
            return True
        return False

    return [track for track in tracks if not blocked(track)]


def install_default_handlers() -> None:
    handlers = {
        "recommend": _recommend,
        "search": _search,
        "playlist": _playlist,
        "taste": _taste,
        "taste_experiment": _taste_experiment,
        "resolve_music_entity": _resolve_music_entity,
        "music_metadata_lookup": _music_metadata_lookup,
        "review_search": _review_search,
        "build_music_dossier": _build_music_dossier,
        "sample_relation_search": _sample_relation_search,
        "locate_sample_sources": _locate_sample_sources,
        "build_sample_dossier": _build_sample_dossier,
        "web_music_search": _web_music_search,
        "artist_albums": _artist_albums,
        "similar_artists": _similar_artists,
        "import_netease_playlist": _import_netease_playlist,
        "journey": _journey,
        "video_search": _video_search,
        "web_info_search": _web_info_search,
        "fetch_metadata": _fetch_metadata,
        "memory_update": _memory_update,
        "similar_cross": _similar_cross,
        "similar_intra": _similar_intra,
        "retrieve": _retrieve,
        "analyze": _analyze,
        "report": _report,
    }
    for name, handler in handlers.items():
        bind_tool_handler(name, handler)
    bind_async_tool_handler("web_music_search", _web_music_search_async)
    bind_async_tool_handler("artist_albums", _artist_albums_async)
    bind_async_tool_handler("video_search", _video_search_async)
    bind_async_tool_handler("web_info_search", _web_info_search_async)
    for name in AUX_TOOL_NAMES:
        bind_tool_handler(name, _aux_handler(name))


def _result(
    name: str,
    data: dict[str, Any],
    summary: str,
    tracks: list[Any] | None = None,
    *,
    expects_tracks: bool = False,
) -> ToolResult:
    tracks = tracks or []
    return ToolResult(
        tool=name,
        status=ToolStatus.EMPTY if expects_tracks and not tracks else ToolStatus.OK if data or tracks else ToolStatus.EMPTY,
        data=data,
        summary=summary,
        cards=[song_card(track) for track in tracks],
        provenance=[{"source": getattr(track, "source", "unknown"), "source_id": getattr(track, "external_id", "") or getattr(track, "asset_id", "")} for track in tracks],
    )


def _recommend(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    if ctx.asset_id:
        answer = ctx.agent.recommend_with_memory(ctx.asset_id, ctx.user_id, args["query"], args.get("top_k", 5))
        return _result("recommend", {"type": "recommend", "answer": answer}, f"生成 {len(answer.recommended_segments)} 个片段推荐。")
    plan = ctx.plan or {}
    retrieval = plan.get("retrieval_plan") or {}
    recommendation = ctx.agent.recommend_for_query(
        ctx.user_id, args["query"], top_k=args.get("top_k", 5),
        seed_tracks=_collect_tracks(ctx.prior_results),
        excluded_tracks=plan.get("_excluded_tracks") or None,
        search_variants=retrieval.get("search_variants") or None,
    )
    tracks = [item.asset for item in recommendation.tracks]
    kept = _filter_content_exclusions(tracks, retrieval.get("excluded_terms") or [])
    if len(kept) != len(tracks):
        allowed = {
            (getattr(track, "external_id", "") or getattr(track, "asset_id", ""), track.title.lower())
            for track in kept
        }
        recommendation.tracks = [
            item for item in recommendation.tracks
            if (
                getattr(item.asset, "external_id", "") or getattr(item.asset, "asset_id", ""),
                item.asset.title.lower(),
            ) in allowed
        ]
        tracks = kept
    return _result("recommend", {"type": "daily_recommend", "recommendation": recommendation}, f"生成 {len(tracks)} 首推荐。", tracks, expects_tracks=True)


def _search(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    plan = ctx.plan or {}
    retrieval = plan.get("retrieval_plan") or {}
    excluded = plan.get("_excluded_tracks") or []
    rec_offset = len(excluded) if excluded else 0
    response = ctx.agent.search(
        ctx.user_id, args["query"],
        include_external=args.get("include_external", True), top_k=12, offset=rec_offset,
    )
    if excluded:
        response.external = _filter_excluded(response.external, excluded)
    response.external = _filter_content_exclusions(response.external, retrieval.get("excluded_terms") or [])
    response.local = _filter_content_exclusions(response.local, retrieval.get("excluded_terms") or [])
    tracks = [*response.external, *response.local]
    return _result("search", {"type": "search", "response": response}, f"本地 {len(response.local)} 首，外部 {len(response.external)} 首。", tracks, expects_tracks=True)


def _playlist(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    plan = ctx.plan or {}
    retrieval = plan.get("retrieval_plan") or {}
    excluded = plan.get("_excluded_tracks") or []
    playlist = ctx.agent.generate_playlist(ctx.user_id, args["instruction"], seed_tracks=_collect_tracks(ctx.prior_results), target_count=args.get("target_count"))
    if excluded and playlist.tracks:
        playlist.tracks = _filter_excluded(playlist.tracks, excluded)
    playlist.tracks = _filter_content_exclusions(
        list(playlist.tracks), retrieval.get("excluded_terms") or [],
    )
    return _result("playlist", {"type": "playlist", "playlist": playlist}, f"生成 {len(playlist.tracks)} 首歌单。", list(playlist.tracks), expects_tracks=True)


def _taste(_args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    summary = ctx.agent.summarize_taste(ctx.user_id)
    return _result("taste", {"type": "taste", "summary": summary}, "已总结用户品味。")


def _taste_experiment(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    experiment = ctx.agent.generate_taste_experiment(ctx.user_id, args["prompt"], total=args.get("total", 12))
    from app.graph.nodes import _taste_experiment_card
    cards = [_taste_experiment_card(item) for segment in experiment.segments for item in segment.tracks]
    result = _result("taste_experiment", {"type": "taste_experiment", "experiment": experiment}, f"生成 {len(cards)} 首三档品味实验候选。")
    result.cards = cards
    result.status = ToolStatus.OK if cards else ToolStatus.EMPTY
    return result


def _knowledge_entities_from_prior(ctx: ToolContext) -> list[Any]:
    from app.models import MusicEntity

    for result in reversed(ctx.prior_results or []):
        if result.get("type") == "music_entity_resolution":
            return [MusicEntity.model_validate(item) for item in result.get("entities", [])]
    return []


def _resolve_music_entity(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    from app.knowledge import resolve_music_entities

    plan = ctx.plan or {}
    intent = str(args.get("intent") or plan.get("intent") or "")
    entities = resolve_music_entities(args.get("query") or ctx.query, intent, plan)
    data = {
        "type": "music_entity_resolution",
        "entities": [entity.model_dump(mode="json") for entity in entities],
        "partial": not bool(entities),
    }
    summary = f"解析到 {len(entities)} 个音乐实体。" if entities else "未能稳定解析音乐实体。"
    return ToolResult(tool="resolve_music_entity", status=ToolStatus.OK if entities else ToolStatus.EMPTY, data=data, summary=summary)


def _music_metadata_lookup(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    from app.knowledge import lookup_metadata, resolve_music_entities

    plan = ctx.plan or {}
    entities = _knowledge_entities_from_prior(ctx) or resolve_music_entities(args.get("query") or ctx.query, str(plan.get("intent") or ""), plan)
    payload = lookup_metadata(ctx.agent, entities, ctx.deadline_at)
    data = {
        "type": "music_metadata",
        "entities": [entity.model_dump(mode="json") for entity in entities],
        **payload,
    }
    skipped = payload.get("skipped_due_to_deadline") or []
    status = ToolStatus.EMPTY if skipped or not (payload.get("metadata") or payload.get("tracks") or payload.get("citations")) else ToolStatus.OK
    summary = "资料查询因时间预算不足被跳过。" if skipped else f"获取 {len(payload.get('citations') or [])} 条资料来源。"
    return ToolResult(tool="music_metadata_lookup", status=status, data=data, summary=summary)


def _review_search(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    from app.knowledge import resolve_music_entities, search_reviews

    plan = ctx.plan or {}
    entities = _knowledge_entities_from_prior(ctx) or resolve_music_entities(args.get("query") or ctx.query, str(plan.get("intent") or ""), plan)
    payload = search_reviews(entities, ctx.deadline_at)
    data = {
        "type": "review_search",
        "entities": [entity.model_dump(mode="json") for entity in entities],
        **payload,
    }
    skipped = payload.get("skipped_due_to_deadline") or []
    status = ToolStatus.EMPTY if skipped or not payload.get("citations") else ToolStatus.OK
    summary = "乐评搜索因时间预算不足被跳过。" if skipped else f"获取 {len(payload.get('citations') or [])} 条乐评来源。"
    return ToolResult(tool="review_search", status=status, data=data, summary=summary)


def _build_music_dossier(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    from app.knowledge import build_dossier, dossier_answer, resolve_music_entities
    from app.models import MusicCitation, MusicEntity, ReviewOpinion, TrackRef

    plan = ctx.plan or {}
    query = args.get("query") or ctx.query
    intent = str(plan.get("intent") or "")
    entities = _knowledge_entities_from_prior(ctx) or resolve_music_entities(query, intent, plan)
    metadata: list[dict[str, Any]] = []
    metadata_citations: list[MusicCitation] = []
    review_citations: list[MusicCitation] = []
    opinions: list[ReviewOpinion] = []
    tracks: list[TrackRef] = []
    skipped: list[str] = []
    for result in ctx.prior_results or []:
        if result.get("type") == "music_metadata":
            metadata.extend(result.get("metadata") or [])
            metadata_citations.extend(MusicCitation.model_validate(item) for item in result.get("citations", []) or [])
            tracks.extend(TrackRef.model_validate(item) for item in result.get("tracks", []) or [])
            skipped.extend(result.get("skipped_due_to_deadline") or [])
        elif result.get("type") == "review_search":
            review_citations.extend(MusicCitation.model_validate(item) for item in result.get("citations", []) or [])
            opinions.extend(ReviewOpinion.model_validate(item) for item in result.get("opinions", []) or [])
            skipped.extend(result.get("skipped_due_to_deadline") or [])
        elif result.get("type") == "music_entity_resolution" and not entities:
            entities = [MusicEntity.model_validate(item) for item in result.get("entities", []) or []]
    dossier = build_dossier(
        ctx.agent, query, intent, entities, metadata, metadata_citations,
        review_citations, opinions, tracks, ctx.deadline_at, skipped,
    )
    data = {
        "type": "music_dossier",
        "dossier": dossier.model_dump(mode="json"),
        "answer": dossier_answer(dossier),
    }
    return ToolResult(
        tool="build_music_dossier",
        status=ToolStatus.OK,
        data=data,
        summary="生成音乐档案。" + ("（部分资料降级）" if dossier.partial else ""),
        provenance=[{"source": c.source, "url": c.url, "kind": c.kind} for c in dossier.citations],
    )


def _sample_target_from_prior(ctx: ToolContext):
    from app.models import TrackRef

    for result in reversed(ctx.prior_results or []):
        if result.get("type") == "sample_relation_search" and result.get("target"):
            return TrackRef.model_validate(result.get("target"))
    return None


def _sample_evidence_from_prior(ctx: ToolContext):
    from app.models import SampleEvidence

    evidence = []
    for result in ctx.prior_results or []:
        if result.get("type") == "sample_relation_search":
            evidence.extend(SampleEvidence.model_validate(item) for item in result.get("evidence", []) or [])
    return evidence


def _sample_relations_from_prior(ctx: ToolContext):
    from app.models import SampleRelation

    relations = []
    for result in ctx.prior_results or []:
        if result.get("type") == "locate_sample_sources":
            relations.extend(SampleRelation.model_validate(item) for item in result.get("relations", []) or [])
    return relations


def _sample_cards_from_prior(ctx: ToolContext) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    for result in ctx.prior_results or []:
        if result.get("type") == "locate_sample_sources":
            cards.extend(result.get("source_cards", []) or [])
    return cards


def _sample_relation_search(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    from app.knowledge import resolve_music_entities, search_sample_relations

    plan = ctx.plan or {}
    query = args.get("query") or ctx.query
    entities = _knowledge_entities_from_prior(ctx) or resolve_music_entities(query, "sample_lookup", plan)
    payload = search_sample_relations(entities, query, ctx.deadline_at)
    data = {"type": "sample_relation_search", **payload}
    skipped = payload.get("skipped_due_to_deadline") or []
    evidence = payload.get("evidence") or []
    status = ToolStatus.EMPTY if skipped or not evidence else ToolStatus.OK
    summary = "采样关系搜索因时间预算不足被跳过。" if skipped else f"获取 {len(evidence)} 条采样证据。"
    return ToolResult(tool="sample_relation_search", status=status, data=data, summary=summary)


def _locate_sample_sources(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    from app.knowledge import locate_sample_sources, resolve_music_entities, search_sample_relations
    from app.models import SampleEvidence, TrackRef

    plan = ctx.plan or {}
    query = args.get("query") or ctx.query
    target = _sample_target_from_prior(ctx)
    evidence = _sample_evidence_from_prior(ctx)
    if target is None:
        entities = _knowledge_entities_from_prior(ctx) or resolve_music_entities(query, "sample_lookup", plan)
        payload = search_sample_relations(entities, query, ctx.deadline_at)
        target = TrackRef.model_validate(payload.get("target") or {"title": query})
        evidence = [SampleEvidence.model_validate(item) for item in payload.get("evidence", []) or []]
    payload = locate_sample_sources(ctx.agent, target, evidence, ctx.deadline_at)
    data = {"type": "locate_sample_sources", "target": target.model_dump(mode="json"), **payload}
    skipped = payload.get("skipped_due_to_deadline") or []
    status = ToolStatus.EMPTY if skipped or not payload.get("relations") else ToolStatus.OK
    summary = "源曲定位因时间预算不足被跳过。" if skipped else f"定位 {len(payload.get('source_cards') or [])} 个源曲候选。"
    return ToolResult(tool="locate_sample_sources", status=status, data=data, summary=summary, cards=payload.get("source_cards") or [])


def _build_sample_dossier(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    from app.knowledge import build_sample_dossier, sample_dossier_answer
    from app.models import TrackRef

    target = _sample_target_from_prior(ctx) or TrackRef(title=args.get("query") or ctx.query, source="query")
    evidence = _sample_evidence_from_prior(ctx)
    relations = _sample_relations_from_prior(ctx)
    cards = _sample_cards_from_prior(ctx)
    skipped: list[str] = []
    for result in ctx.prior_results or []:
        skipped.extend(result.get("skipped_due_to_deadline") or [])
    dossier = build_sample_dossier(target, evidence, relations, cards, skipped)
    data = {
        "type": "sample_dossier",
        "sample_dossier": dossier.model_dump(mode="json"),
        "sample_relations": [rel.model_dump(mode="json") for rel in dossier.relations],
        "source_cards": cards,
        "answer": sample_dossier_answer(dossier),
    }
    return ToolResult(
        tool="build_sample_dossier",
        status=ToolStatus.OK if dossier.relations else ToolStatus.EMPTY,
        data=data,
        summary="生成采样溯源结果。" + ("（部分资料降级）" if dossier.partial else ""),
        cards=cards,
        provenance=[{"source": ev.source, "url": ev.url, "confidence": ev.confidence} for ev in dossier.citations],
    )


def _web_music_search(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    plan = ctx.plan or {}
    retrieval = plan.get("retrieval_plan") or {}
    excluded = plan.get("_excluded_tracks") or []
    # args["query"] 已由 _planned_arguments 注入实体（_query_with_entities），直接用作搜索词；
    # relevance_query 用 LLM 改写的正向 search_query 做相关性过滤（空则回退原始 query）。
    search_core = (retrieval.get("search_query") or "").strip() or ctx.query
    variants = retrieval.get("search_variants")
    language_filter = (retrieval.get("language_filter") or "").strip().lower()
    target = plan.get("target_count") or args.get("top_k", 5)
    rec_offset = len(excluded) if excluded else 0
    tracks = ctx.agent.search_web_music(
        args["query"], top_k=max(target, args.get("top_k", 5)),
        relevance_query=search_core, offset=rec_offset, variants=variants,
    )
    if excluded:
        tracks = _filter_excluded(tracks, excluded)
    tracks = _filter_content_exclusions(tracks, retrieval.get("excluded_terms") or [])
    tracks = _apply_language_filter(tracks, language_filter, target)
    for track in tracks:
        ctx.agent.library.upsert_external(track)
    return _result("web_music_search", {"type": "web_music_search", "tracks": tracks}, f"获取 {len(tracks)} 个线上候选。", tracks, expects_tracks=True)


async def _web_music_search_async(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    plan = ctx.plan or {}
    retrieval = plan.get("retrieval_plan") or {}
    excluded = plan.get("_excluded_tracks") or []
    search_core = (retrieval.get("search_query") or "").strip() or ctx.query
    variants = retrieval.get("search_variants")
    language_filter = (retrieval.get("language_filter") or "").strip().lower()
    target = plan.get("target_count") or args.get("top_k", 5)
    rec_offset = len(excluded) if excluded else 0
    tracks = await ctx.agent.search_web_music_async(
        args["query"], top_k=max(target, args.get("top_k", 5)),
        relevance_query=search_core, offset=rec_offset, variants=variants,
    )
    if excluded:
        tracks = _filter_excluded(tracks, excluded)
    tracks = _filter_content_exclusions(tracks, retrieval.get("excluded_terms") or [])
    tracks = _apply_language_filter(tracks, language_filter, target)
    return _result(
        "web_music_search", {"type": "web_music_search", "tracks": tracks},
        f"获取 {len(tracks)} 个线上候选。", tracks, expects_tracks=True,
    )


def _artist_albums(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    albums = ctx.agent.recommend_artist_albums(ctx.user_id, args["query"], limit=6)
    result = _result("artist_albums", {"type": "artist_albums", "albums": albums}, f"获取 {len(albums)} 张专辑。")
    result.status = ToolStatus.OK if albums else ToolStatus.EMPTY
    return result


async def _artist_albums_async(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    albums = await ctx.agent.recommend_artist_albums_async(ctx.user_id, args["query"], limit=6)
    result = _result(
        "artist_albums", {"type": "artist_albums", "albums": albums},
        f"获取 {len(albums)} 张专辑。",
    )
    result.status = ToolStatus.OK if albums else ToolStatus.EMPTY
    return result


def _similar_artists(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    from collections import Counter

    seed = str(args.get("artist") or "").strip()
    top_k = max(1, min(int(args.get("top_k", 6) or 6), 12))
    if not seed:
        return ToolResult(tool="similar_artists", status=ToolStatus.EMPTY, summary="缺少要参照的歌手。")

    profiles: dict[str, dict[str, Any]] = {}
    for track in ctx.agent.list_resource_tracks(2500):
        artist = (getattr(track, "artist", "") or "").strip()
        if not artist:
            continue
        profile = profiles.setdefault(artist, {"genres": Counter(), "moods": Counter(), "tracks": []})
        profile["genres"].update(getattr(track, "genre", []) or [])
        profile["moods"].update(getattr(track, "mood", []) or [])
        if len(profile["tracks"]) < 3:
            profile["tracks"].append(getattr(track, "title", ""))

    seed_name = next(
        (name for name in profiles if ctx.agent.artist_name_matches(seed, name)),
        seed,
    )
    seed_profile = profiles.get(seed_name)
    seed_genres = {name for name, _ in seed_profile["genres"].most_common(3)} if seed_profile else set()
    seed_moods = {name for name, _ in seed_profile["moods"].most_common(4)} if seed_profile else set()
    if not seed_genres:
        from app.graph.tag_rules import extract_genre_from_artist
        seed_genres.update(extract_genre_from_artist(seed))

    ranked: list[tuple[float, str, dict[str, Any]]] = []
    for artist, profile in profiles.items():
        if ctx.agent.artist_name_matches(seed, artist):
            continue
        # 只比较候选艺人的主导标签，避免一条误标 R&B 让 Beatles 这类宽标签艺人冲到首位。
        genres = {name for name, _ in profile["genres"].most_common(2)}
        moods = {name for name, _ in profile["moods"].most_common(3)}
        genre_overlap = seed_genres & genres
        mood_overlap = seed_moods & moods
        genre_score = sum(seed_profile["genres"].get(name, 1) for name in genre_overlap) if seed_profile else len(genre_overlap)
        mood_score = sum(seed_profile["moods"].get(name, 1) for name in mood_overlap) if seed_profile else len(mood_overlap)
        score = genre_score * 3.0 + mood_score * 1.2
        if score <= 0:
            continue
        ranked.append((score, artist, {
            "name": artist,
            "genres": [name for name, _ in profile["genres"].most_common(3)],
            "moods": [name for name, _ in profile["moods"].most_common(3)],
            "representative_tracks": [title for title in profile["tracks"] if title],
            "reason": "、".join([*sorted(genre_overlap), *sorted(mood_overlap)]) or "曲库标签相近",
            "source": "local_library",
            "seed_artist": seed_name,
        }))
    ranked.sort(key=lambda item: (-item[0], item[1].lower()))
    artists = [item[2] for item in ranked[:top_k]]
    return ToolResult(
        tool="similar_artists",
        status=ToolStatus.OK if artists else ToolStatus.EMPTY,
        data={"type": "similar_artists", "seed_artist": seed_name, "artists": artists},
        summary=f"基于《{seed_name}》的曲库标签找到 {len(artists)} 位相似歌手。",
        provenance=[{"source": "local_library", "artist": item["name"]} for item in artists],
    )


def _import_netease_playlist(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    imported = ctx.agent.import_netease_playlist(args["playlist_ref"], user_id=ctx.user_id, limit=args.get("limit", 100))
    tracks = imported.get("tracks", [])
    return _result("import_netease_playlist", {"type": "import_netease_playlist", "result": imported}, f"导入《{imported.get('name', '')}》：新增 {imported.get('imported', 0)} 首。", tracks[:12], expects_tracks=True)


def _journey(_args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    from app.models import ExternalTrack
    target_count = (ctx.plan or {}).get("target_count") if isinstance(ctx.plan, dict) else None
    if target_count:
        journey = ctx.agent.generate_music_journey(ctx.user_id, ctx.query, target_count=target_count)
    else:
        journey = ctx.agent.generate_music_journey(ctx.user_id, ctx.query)
    tracks = [ExternalTrack.model_validate(track) for phase in journey.get("phases", []) for track in phase.get("tracks", [])]
    result = _result("journey", {"type": "journey", "journey": journey}, f"生成 {len(journey.get('phases', []))} 个阶段、{len(tracks)} 首曲目。", tracks, expects_tracks=True)
    phase_reasons = [(phase["name"], phase["goal"]) for phase in journey.get("phases", []) for _ in phase.get("tracks", [])]
    for card, (phase, goal) in zip(result.cards, phase_reasons, strict=False):
        card["reason"] = f"{phase}：{goal}"
        card["journey_phase"] = phase
    return result


def _video_search(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    tracks = ctx.agent.search_videos(args["query"], top_k=5)
    for track in tracks:
        ctx.agent.library.upsert_external(track)
    return _result("video_search", {"type": "video_search", "tracks": tracks}, f"获取 {len(tracks)} 个视频结果。", tracks, expects_tracks=True)


def _web_info_search(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    items = ctx.agent.search_artist_info(args["query"])
    return _result("web_info_search", {"type": "web_info_search", "search_results": items}, f"获取 {len(items)} 条可追溯资料。")


async def _video_search_async(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    tracks = await ctx.agent.search_videos_async(args["query"], top_k=5)
    return _result(
        "video_search", {"type": "video_search", "tracks": tracks},
        f"获取 {len(tracks)} 个视频结果。", tracks, expects_tracks=True,
    )


async def _web_info_search_async(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    items = await ctx.agent.search_artist_info_async(args["query"])
    return _result(
        "web_info_search", {"type": "web_info_search", "search_results": items},
        f"获取 {len(items)} 条可追溯资料。",
    )


def _fetch_metadata(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    data = ctx.agent.fetch_track_metadata(asset_id=args.get("asset_id") or ctx.asset_id, url=args.get("url"), use_network=args.get("use_network", True))
    return _result("fetch_metadata", {"type": "fetch_metadata", "metadata": data}, "元数据抓取完成。" if data.get("found") else "未抓到可用元数据。")


def _memory_update(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    from app.models import MemoryUpdateRequest
    _, changed = ctx.agent.update_memory(MemoryUpdateRequest(user_id=ctx.user_id, event=args["event"], asset_id=ctx.asset_id))
    return _result("memory_update", {"type": "memory_update", "changed": changed}, f"记忆{'已更新' if changed else '无变化'}。")


def _similar_cross(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    if not ctx.asset_id:
        return ToolResult(tool="similar_cross", status=ToolStatus.UNSUPPORTED, summary="缺少媒体上下文。")
    items = ctx.agent.find_similar_assets(ctx.asset_id, args.get("top_k", 5))
    return _result("similar_cross", {"type": "similar_cross", "results": items}, f"找到 {len(items)} 个相似媒体。")


def _similar_intra(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    if not ctx.asset_id:
        return ToolResult(tool="similar_intra", status=ToolStatus.UNSUPPORTED, summary="缺少媒体上下文。")
    segments = ctx.agent.media.get_segments(ctx.asset_id)
    items = ctx.agent.find_similar_segments(ctx.asset_id, segments[0].segment_id, args.get("top_k", 5)) if segments else []
    return _result("similar_intra", {"type": "similar_intra", "results": items}, f"找到 {len(items)} 个相似片段。")


def _retrieve(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    if not ctx.asset_id:
        return ToolResult(tool="retrieve", status=ToolStatus.UNSUPPORTED, summary="缺少媒体上下文。")
    evidences = ctx.agent.retrieve_evidence(ctx.asset_id, args["query"], args.get("top_k", 5))
    return _result("retrieve", {"type": "retrieve", "evidences": evidences}, f"检索到 {len(evidences)} 个证据片段。")


def _analyze(_args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    if not ctx.asset_id:
        return ToolResult(tool="analyze", status=ToolStatus.UNSUPPORTED, summary="缺少媒体上下文。")
    asset, segments = ctx.agent.analyze_media(ctx.asset_id)
    return _result("analyze", {"type": "analyze", "asset": asset, "segments": segments}, f"已分析 {asset.title}：生成 {len(segments)} 个片段。")


def _report(_args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    if not ctx.asset_id:
        return ToolResult(tool="report", status=ToolStatus.UNSUPPORTED, summary="缺少媒体上下文。")
    report = ctx.agent.generate_report(ctx.asset_id)
    return _result("report", {"type": "report", "report": report}, report.get("summary", "报告已生成。"))
