from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from collections.abc import Callable, Iterable
from typing import Any

from app.config import settings
from app.models import Asset, ExternalTrack, RankingBreakdown, TasteProfile, UserMemory, utc_now_iso

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RecommendationContext:
    memory: UserMemory
    memory_query: str
    search_goal: str
    anchors: Any
    scene_queries: list[str]
    has_entity: bool
    taste_summary: str
    library_artists: list[str]
    local_ratio: float
    seed_supply: int


class RecommendationService:
    """Incremental extraction target for recommendation-side agent logic."""

    def __init__(
        self,
        *,
        store: Any,
        memory: Any,
        library: Any,
        list_assets: Callable[[], list[Asset]],
        track_key: Callable[[Asset | ExternalTrack], str],
        is_quality_track: Callable[[Any], bool],
        query_noise: Iterable[str],
    ) -> None:
        self.store = store
        self.memory = memory
        self.library = library
        self._list_assets = list_assets
        self._track_key = track_key
        self._is_quality_track = is_quality_track
        self._query_noise = {item.lower() for item in query_noise}

    def build_context(
        self,
        *,
        user_id: str,
        goal: str,
        top_k: int,
        local_ratio: float,
        search_query_override: str | None,
        seed_tracks: list[Asset | ExternalTrack] | None,
        extract_search_query: Callable[[str], str],
        extract_recommendation_anchors: Callable[[str], Any],
        scene_playlist_queries: Callable[[str], list[str]],
        query_has_entity: Callable[[str], bool],
        summarize_taste: Callable[..., str],
        is_verified_track: Callable[[Asset | ExternalTrack], bool],
        is_quality_track: Callable[[Asset | ExternalTrack], bool],
    ) -> RecommendationContext:
        memory = self.memory.get_memory(user_id)
        memory_query = self.memory.weighted_query(memory, include_artists=False)
        search_goal = (search_query_override or extract_search_query(goal)).strip()
        anchors = extract_recommendation_anchors(goal)
        scene_queries = scene_playlist_queries(goal) or scene_playlist_queries(search_goal)
        has_entity = query_has_entity(search_goal)
        taste_summary = summarize_taste(user_id, include_artists=has_entity, memory=memory) if memory.taste_profile else ""
        library_artists = list({asset.artist for asset in self._list_assets() if asset.artist})[:10] if has_entity else []
        seed_supply = sum(
            1 for track in (seed_tracks or [])
            if is_verified_track(track) and is_quality_track(track)
        )
        return RecommendationContext(
            memory=memory,
            memory_query=memory_query,
            search_goal=search_goal,
            anchors=anchors,
            scene_queries=scene_queries,
            has_entity=has_entity,
            taste_summary=taste_summary,
            library_artists=library_artists,
            local_ratio=local_ratio,
            seed_supply=seed_supply,
        )

    def local_recommendation_candidates(
        self,
        user_id: str,
        query: str,
        memory: UserMemory,
        limit: int = 120,
    ) -> list[Asset]:
        """Select relevant local songs without mutating the user's library."""
        taste = memory.taste_profile or TasteProfile()
        preferred = {
            *(name.lower() for name, _ in taste.top_genres[:5]),
            *(name.lower() for name, _ in taste.top_moods[:5]),
            *(name.lower() for name, _ in taste.top_artists[:8]),
        }
        query_terms = {
            token.lower() for token in re.findall(r"[A-Za-z0-9&'-]+|[一-鿿㐀-䶿]{2,}", query or "")
            if token.lower() not in self._query_noise
        }
        scored: list[tuple[int, Asset]] = []
        for track in self._list_assets():
            if track.status != "analyzed" or not self._is_quality_track(track):
                continue
            if self.library.is_disliked(user_id, track):
                continue
            searchable = " ".join([
                track.title, track.artist or "", *track.genre, *track.mood,
            ]).lower()
            query_hits = sum(1 for term in query_terms if term in searchable)
            taste_hits = sum(1 for term in preferred if term and term in searchable)
            score = query_hits * 3 + taste_hits
            if score > 0:
                scored.append((score, track))
        scored.sort(key=lambda item: (-item[0], item[1].title.lower(), (item[1].artist or "").lower()))
        return [track for _, track in scored[:limit]]

    def extend_exact_route_candidates(
        self,
        *,
        candidates: list[Asset | ExternalTrack],
        search_goal: str,
        goal: str,
        anchors: Any,
        search_variants: list[str] | None,
        top_k: int,
        excluded_tracks: list[dict[str, str]] | None,
        search_web_music: Callable[..., list[ExternalTrack]],
        dedupe_tracks: Callable[[list[Asset | ExternalTrack]], list[Asset | ExternalTrack]],
        recommendation_search_seeds: Callable[[str, str, Any, list[str] | None], list[str]],
    ) -> tuple[list[Asset | ExternalTrack], list[str]]:
        seeds = recommendation_search_seeds(search_goal, goal, anchors, search_variants)
        trace_lines = [
            f"route=anchor_exact, search_goal={search_goal}, seeds={len(seeds)}, "
            f"artists={len(anchors.artists)}, styles={len(anchors.styles)}"
        ]
        rec_offset = len(excluded_tracks) if excluded_tracks else 0
        for idx, seed in enumerate(seeds):
            batch = search_web_music(
                seed,
                top_k=max(top_k, 6),
                relevance_query=seed,
                offset=rec_offset if idx == 0 else 0,
                variants=None,
            )
            candidates.extend(batch)
            if len(dedupe_tracks(candidates)) >= max(top_k * 3, top_k):
                break
        return candidates, trace_lines

    def extend_discovery_route_candidates(
        self,
        *,
        candidates: list[Asset | ExternalTrack],
        goal: str,
        search_goal: str,
        scene_queries: list[str],
        prefer_playlist: bool,
        top_k: int,
        memory: UserMemory,
        taste_summary: str,
        library_artists: list[str],
        dedupe_tracks: Callable[[list[Asset | ExternalTrack]], list[Asset | ExternalTrack]],
        search_and_extract: Callable[..., list[ExternalTrack]],
        discover_from_llm: Callable[..., list[ExternalTrack]],
        discover_from_lastfm: Callable[..., list[ExternalTrack]],
    ) -> tuple[list[Asset | ExternalTrack], list[str]]:
        trace_lines: list[str] = []
        taste_genres = ""
        if memory.taste_profile and memory.taste_profile.top_genres:
            taste_genres = " ".join(g for g, _ in memory.taste_profile.top_genres[:2])
        playlist_query = scene_queries[0] if scene_queries else (f"{taste_genres} {search_goal}".strip() or goal)
        candidate_budget = max(top_k * 3, 12)
        playlist_tracks = search_and_extract(
            playlist_query,
            max_playlists=3,
            tracks_per_playlist=candidate_budget,
        )
        trace_lines.append(f"route=playlist, query={playlist_query!r}, extracted={len(playlist_tracks)}")
        candidates.extend(playlist_tracks)

        if not prefer_playlist:
            llm_tracks = discover_from_llm(
                query=goal,
                taste_summary=taste_summary,
                exclusion_rules=memory.exclusion_rules,
                library_artists=library_artists,
                target_count=max(top_k * 2, 8),
            )
            trace_lines.append(f"route=llm_candidates, generated={len(llm_tracks)}")
            candidates.extend(llm_tracks)

            taste_artists = [a for a, _ in (memory.taste_profile.top_artists if memory.taste_profile else [])]
            taste_genre_names = [g for g, _ in (memory.taste_profile.top_genres if memory.taste_profile else [])]
            lastfm_tracks = discover_from_lastfm(
                top_artists=taste_artists or library_artists,
                top_genres=taste_genre_names,
                target_count=max(top_k * 2, 8),
            )
            if lastfm_tracks:
                trace_lines.append(f"route=lastfm, verified={len(lastfm_tracks)}")
                candidates.extend(lastfm_tracks)

        if prefer_playlist and len(dedupe_tracks(candidates)) < candidate_budget:
            follow_up_scene_queries = scene_queries[1:]
        else:
            follow_up_scene_queries = []
        for scene_query in follow_up_scene_queries:
            if len(dedupe_tracks(candidates)) >= max(top_k * 3, top_k):
                break
            batch = search_and_extract(
                scene_query,
                max_playlists=2,
                tracks_per_playlist=max(top_k * 2, 10),
            )
            trace_lines.append(f"route=scene_playlist, query={scene_query!r}, extracted={len(batch)}")
            candidates.extend(batch)
        return candidates, trace_lines

    def record_recommendation_history(self, user_id: str, tracks: list[Asset | ExternalTrack]) -> None:
        keys = [self._track_key(track) for track in tracks]
        if not keys or self.store is None:
            return
        with self.store.lock("memory", user_id):
            memory = self.memory.get_memory(user_id)
            memory.recommendation_history = [*memory.recommendation_history, *keys][-200:]
            memory.updated_at = utc_now_iso()
            self.store.write_model("memory", user_id, memory)

    def filter_verified_candidates(
        self,
        *,
        candidates: list[Asset | ExternalTrack],
        user_id: str,
        goal: str,
        excluded_tracks: list[dict[str, str]] | None,
        dedupe_tracks: Callable[[list[Asset | ExternalTrack]], list[Asset | ExternalTrack]],
        is_verified_track: Callable[[Asset | ExternalTrack], bool],
        is_quality_track: Callable[[Asset | ExternalTrack], bool],
        is_context_compatible: Callable[[str, Asset | ExternalTrack], bool],
        allow_variants: bool,
        filter_excluded_tracks: Callable[[list[Asset | ExternalTrack], list[dict[str, str]]], list[Asset | ExternalTrack]],
    ) -> list[Asset | ExternalTrack]:
        verified = [
            track for track in dedupe_tracks(candidates)
            if is_verified_track(track)
            and not self.library.is_disliked(user_id, track)
            and is_quality_track(track, allow_variants=allow_variants)
            and is_context_compatible(goal, track)
        ]
        if excluded_tracks:
            verified = filter_excluded_tracks(verified, excluded_tracks)
        return verified

    def extend_with_online_fallback(
        self,
        *,
        verified: list[Asset | ExternalTrack],
        user_id: str,
        goal: str,
        search_goal: str,
        top_k: int,
        excluded_tracks: list[dict[str, str]] | None,
        search_variants: list[str] | None,
        can_fallback: bool,
        search_web_music: Callable[..., list[ExternalTrack]],
        is_verified_online_track: Callable[[Asset | ExternalTrack], bool],
        is_quality_track: Callable[[Asset | ExternalTrack], bool],
        is_context_compatible: Callable[[str, Asset | ExternalTrack], bool],
        allow_variants: bool,
    ) -> list[Asset | ExternalTrack]:
        if len(verified) >= top_k or not search_goal or not can_fallback:
            return verified
        fb_offset = len(excluded_tracks or []) + len(verified)
        fallback_batch = search_web_music(
            search_goal,
            top_k=max(top_k * 2, top_k),
            offset=fb_offset,
            variants=search_variants,
        )
        for track in fallback_batch:
            if (
                is_verified_online_track(track)
                and not self.library.is_disliked(user_id, track)
                and is_quality_track(track, allow_variants=allow_variants)
                and is_context_compatible(goal, track)
            ):
                if not any(self._track_key(track) == self._track_key(existing) for existing in verified):
                    verified.append(track)
        return verified

    def extend_with_resource_pool(
        self,
        *,
        verified: list[Asset | ExternalTrack],
        user_id: str,
        goal: str,
        search_goal: str,
        top_k: int,
        anchors_explicit: bool,
        prefer_playlist: bool,
        dense_library_fallback: Callable[[str, list[Asset | ExternalTrack], int], list[ExternalTrack]],
        is_quality_track: Callable[[Asset | ExternalTrack], bool],
        is_context_compatible: Callable[[str, Asset | ExternalTrack], bool],
        anchor_matcher: Callable[[Asset | ExternalTrack], bool],
        allow_variants: bool,
    ) -> tuple[list[Asset | ExternalTrack], int]:
        if len(verified) >= top_k:
            return verified, 0
        pool_hits = dense_library_fallback(search_goal or goal, verified, max(top_k * 2, top_k))
        pool_hits = [
            track for track in pool_hits
            if not self.library.is_disliked(user_id, track)
            and is_quality_track(track, allow_variants=allow_variants)
            and is_context_compatible(goal, track)
        ]
        if anchors_explicit and not prefer_playlist:
            pool_hits = [track for track in pool_hits if anchor_matcher(track)]
        if pool_hits:
            verified.extend(pool_hits)
        return verified, len(pool_hits)

    def prioritize_fresh_candidates(
        self,
        verified: list[Asset | ExternalTrack],
        recent_history: list[str],
        *,
        top_k: int,
    ) -> list[Asset | ExternalTrack]:
        recent = set(recent_history[-120:])
        fresh = [track for track in verified if self._track_key(track) not in recent]
        repeated = [track for track in verified if self._track_key(track) in recent]
        return fresh if len(fresh) >= top_k else [*fresh, *repeated]

    def rerank_tracks(
        self,
        user_id: str,
        query: str,
        tracks: list[Any],
        top_k: int,
        *,
        profile_signal_provider: Callable[[str], tuple[set[str], set[str]]],
    ) -> list[tuple[Any, RankingBreakdown]]:
        from app.graph.tag_rules import extract_scenario
        from app.memory import compute_behavior_scores
        from app.recommend.rerank import language_distribution, rerank_candidates

        if not settings.enable_rerank or not tracks:
            fallback = [
                (track, RankingBreakdown(
                    title=getattr(track, "title", ""),
                    source=getattr(track, "source", "local"),
                    score=round(1.0 - index * 0.04, 4),
                    reason="顺序兜底（rerank 关闭）",
                ))
                for index, track in enumerate(tracks[:top_k])
            ]
            return fallback

        memory = self.memory.get_memory(user_id)
        taste = memory.taste_profile
        durations = {asset.asset_id: asset.duration_seconds for asset in self._list_assets()}
        behavior = compute_behavior_scores(memory.listening_history, durations)
        scenarios = {scenario.lower() for scenario in extract_scenario(query)}
        self.enrich_candidate_tags(tracks)
        lang_pref = language_distribution(self._list_assets())
        exclusion_rules = memory.exclusion_rules or None
        profile_boost, profile_penalty = profile_signal_provider(user_id)
        cf_scores, cf_ok = self.collaborative_scores(user_id, tracks, memory)
        ts_scores = self.library.sample_ts_scores(tracks) if settings.enable_explore else None
        return rerank_candidates(
            query,
            tracks,
            taste,
            behavior_scores=behavior,
            scenarios=scenarios,
            top_k=top_k,
            lang_pref=lang_pref,
            exclusion_rules=exclusion_rules,
            collaborative_scores=cf_scores,
            collaborative_ok=cf_ok,
            ts_scores=ts_scores,
            profile_boost_artists=profile_boost or None,
            profile_penalty_artists=profile_penalty or None,
        )

    def collaborative_scores(
        self,
        user_id: str,
        tracks: list[Any],
        memory: Any,
    ) -> tuple[list[float] | None, bool]:
        if settings.tri_anchor_w_collaborative <= 0 or not tracks or self.store is None:
            return None, False
        try:
            from app.recommend.collaborative import (
                build_cooccurrence,
                collaborative_scores,
                recent_listened_ids,
            )
            from app.recommend.rerank import _track_id

            recent = recent_listened_ids(memory.listening_history)
            if not recent:
                return None, False
            histories: list[list[str]] = []
            for uid in self.store.list_keys("memory"):
                user_memory = self.memory.get_memory(uid)
                ids = [getattr(event, "asset_id", "") for event in user_memory.listening_history]
                if ids:
                    histories.append(ids)
            cooccurrence = build_cooccurrence(histories)
            scores, ok = collaborative_scores(
                [_track_id(track) for track in tracks], recent, cooccurrence,
            )
            return (scores, ok) if ok else (None, False)
        except Exception:
            logger.debug("CF 协同锚计算失败，降级三锚", exc_info=True)
            return None, False

    @staticmethod
    def enrich_candidate_tags(tracks: list[Any]) -> None:
        from app.graph.tag_rules import extract_genre, extract_genre_from_artist, extract_mood

        for track in tracks:
            text = f"{getattr(track, 'title', '')} {getattr(track, 'artist', '') or ''}"
            if not getattr(track, "genre", None):
                inferred = extract_genre(text)
                if inferred and hasattr(track, "genre"):
                    try:
                        track.genre = inferred
                    except Exception:
                        pass
            if not getattr(track, "genre", None):
                artist = getattr(track, "artist", "") or ""
                if artist:
                    inferred = extract_genre_from_artist(artist)
                    if inferred and hasattr(track, "genre"):
                        try:
                            track.genre = inferred
                        except Exception:
                            pass
            if not getattr(track, "mood", None):
                inferred = extract_mood(text)
                if inferred and hasattr(track, "mood"):
                    try:
                        track.mood = inferred
                    except Exception:
                        pass

    @staticmethod
    def query_has_entity(search_goal: str, query_noise: Iterable[str]) -> bool:
        """Detect whether the query carries an artist/song entity instead of only vibe words."""
        if not search_goal:
            return False

        generic_en = {
            "chill", "lofi", "lo-fi", "vibe", "vibes", "mix", "remix", "relax", "relaxing",
            "mood", "moody", "groove", "groovy", "upbeat", "slow", "fast", "happy", "sad",
            "deep", "party", "cozy", "dreamy", "mellow", "smooth", "calm", "calming", "peaceful",
            "soothing", "soft", "warm", "bright", "dark", "melancholy", "melancholic",
            "nostalgic", "uplifting", "energetic", "emotional", "romantic", "sexy", "sensual",
            "dramatic", "epic", "ethereal", "atmospheric", "minimal", "lush",
            "r&b", "rnb", "soul", "pop", "rock", "rap", "hip", "hop", "hiphop", "jazz",
            "electronic", "edm", "ambient", "acoustic", "indie", "funk", "house", "techno",
            "trap", "disco", "reggae", "blues", "country", "classical", "metal", "punk",
            "folk", "dance", "dreampop", "shoegaze", "synthwave", "instrumental", "vocal",
            "morning", "night", "nighttime", "evening", "afternoon", "midnight", "summer",
            "winter", "autumn", "spring", "rainy", "sunny", "study", "focus", "sleep", "sleepy",
            "workout", "gym", "running", "driving", "coffee", "work", "working", "commute",
            "playlist", "playlists", "songs", "song", "music", "track", "tracks", "recommend",
            "recommendation", "recommendations", "best", "top", "new", "old", "classic",
            "popular", "trending", "favorite", "favourites", "similar", "like", "beats", "tunes",
        }
        english = re.findall(r"[A-Za-z][A-Za-z0-9'&\-]*", search_goal)
        english = [t for t in english if len(t) > 1 and t.lower() not in generic_en]
        if english:
            return True

        cjk_tokens = re.findall(r"[一-鿿㐀-䶿]{2,}", search_goal)
        general_words = {
            "慵懒", "律动", "放松", "治愈", "欢快", "伤感", "浪漫", "激昂", "宁静", "梦幻",
            "轻松", "开心", "忧郁", "温馨", "热血", "安静", "舒缓", "劲爆", "性感", "温柔",
            "甜蜜", "兴奋", "空灵", "愉悦", "感动", "舒服", "烦躁", "低沉", "吵闹",
            "跑步", "运动", "工作", "学习", "睡眠", "开车", "通勤", "派对", "咖啡",
            "健身", "旅行", "约会", "散步", "泡澡", "专注",
            "深夜", "早晨", "下午", "夜晚", "凌晨", "熬夜", "今夜", "今晚", "周末",
            "早上", "晚上", "白天", "午后", "傍晚",
            "说唱", "摇滚", "电子", "古典", "爵士", "民谣", "国风", "金属", "朋克",
            "嘻哈", "蓝调", "乡村", "雷鬼", "灵魂", "放克", "迪斯科", "浩室",
            "独立", "后摇", "新浪潮", "实验", "氛围", "新金属",
            "混搭", "推荐", "适合", "流行", "好听", "经典", "热门", "小众", "风格",
            "陪伴", "陪你", "感觉", "能量", "曲风", "节奏", "全部", "一些", "几首", "都有", "全都有",
            "唱歌", "跳舞", "听歌", "背景",
            "从", "到", "帮", "让", "给", "想", "要", "能", "来", "去",
            "一个人", "两个人", "朋友", "恋人", "情侣",
            "好听的音乐", "推荐一些歌", "推荐几首歌", "帮我推荐一些歌",
            "给我推荐", "推荐一些", "推荐几首", "帮我推荐",
        }
        blocked = {item.lower() for item in query_noise}
        non_general = [t for t in cjk_tokens if t not in general_words and t.lower() not in blocked]
        return bool(non_general)
