from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

from app.config import settings
from app.models import Asset, ExternalTrack, SearchResponse
from app.sources import bilibili as bilibili_source
from app.sources import web_search as web_search_source
from app.sources import youtube as youtube_source

logger = logging.getLogger(__name__)


class DiscoverService:
    def __init__(
        self,
        *,
        memory: Any,
        list_assets: Any,
        library: Any,
        retrieve_library_evidence: Any,
        search_web_music: Any,
        track_key: Any,
        dedupe_tracks: Any,
        classify_candidate_kind: Any,
        sync_search_videos: Any,
        sync_search_artist_info: Any,
        extract_search_query: Any,
        format_search_summary: Any,
        valid_verified_online_track: Any,
        is_fallback_track: Any,
        artist_query_matches: Any,
        normalize_match_text: Any,
        artist_credit_parts: Any,
        artist_alias_keys: Any,
        looks_like_bare_artist_query: Any,
        string_similarity: Any,
    ) -> None:
        self.memory = memory
        self._list_assets = list_assets
        self.library = library
        self._retrieve_library_evidence = retrieve_library_evidence
        self._search_web_music = search_web_music
        self._track_key = track_key
        self._dedupe_tracks = dedupe_tracks
        self._classify_candidate_kind = classify_candidate_kind
        self._sync_search_videos = sync_search_videos
        self._sync_search_artist_info = sync_search_artist_info
        self._extract_search_query = extract_search_query
        self._format_search_summary = format_search_summary
        self._is_verified_online_track = valid_verified_online_track
        self._is_fallback_track = is_fallback_track
        self._artist_query_matches = artist_query_matches
        self._normalize_match_text = normalize_match_text
        self._artist_credit_parts = artist_credit_parts
        self._artist_alias_keys = artist_alias_keys
        self._looks_like_bare_artist_query = looks_like_bare_artist_query
        self._string_similarity = string_similarity

    def search(
        self,
        user_id: str,
        query: str,
        include_external: bool = True,
        top_k: int = 20,
        offset: int = 0,
    ) -> SearchResponse:
        memory = self.memory.get_memory(user_id)
        memory_query = self.memory.weighted_query(memory, include_artists=False)
        expanded_query = f"{query} {memory_query}".strip()
        search_goal = self._extract_search_query(query)
        classification = self.classify_discover_query(query)
        query_kind = classification.get("kind")
        artist_query = query_kind == "artist"
        resolved_artist_query = classification.get("normalized_query") or search_goal or query
        local_terms = search_goal.lower().split() if search_goal else query.lower().split()
        local_results: list[Asset] = []
        library_assets = self._list_assets()
        for asset in library_assets:
            if artist_query:
                if self._artist_query_matches(resolved_artist_query, asset.artist or "", allow_fuzzy=True):
                    local_results.append(asset)
                continue
            searchable = f"{asset.title} {asset.artist or ''} {' '.join(asset.genre)} {' '.join(asset.mood)}".lower()
            if any(term in searchable for term in local_terms):
                local_results.append(asset)

        evidences = [] if artist_query else self._retrieve_library_evidence(expanded_query, top_k=min(top_k, 6))
        local_by_id = {asset.asset_id: asset for asset in local_results}
        for evidence in evidences:
            asset_id = str(evidence.metadata.get("asset_id", ""))
            asset = next((item for item in library_assets if item.asset_id == asset_id), None)
            if asset is not None:
                local_by_id.setdefault(asset.asset_id, asset)

        external_results: list[ExternalTrack] = []
        if include_external:
            external_results = self._search_web_music(
                resolved_artist_query if artist_query else expanded_query,
                top_k=top_k,
                relevance_query=search_goal,
                offset=offset,
            )

        if include_external and not artist_query and len(external_results) < 3:
            try:
                from app.search.netease_playlist import search_and_extract

                playlist_hits = search_and_extract(
                    f"{search_goal or query}音乐",
                    max_playlists=3,
                    tracks_per_playlist=top_k,
                )
                existing_keys = {self._track_key(track) for track in external_results}
                for track in playlist_hits:
                    if self._is_verified_online_track(track) and self._track_key(track) not in existing_keys:
                        external_results.append(track)
                        existing_keys.add(self._track_key(track))
            except Exception:
                logger.debug("search playlist fallback failed for %s", query, exc_info=True)

        summary = self._format_search_summary(
            query=query,
            local=list(local_by_id.values())[:top_k],
            external=external_results[:top_k],
            memory_query="" if artist_query else memory_query,
        )
        return SearchResponse(
            local=list(local_by_id.values())[:top_k],
            external=external_results[:top_k],
            summary=summary,
            evidences=evidences,
            agent_trace=[
                f"query={query}",
                f"memory_query={memory_query or 'none'}",
                f"local_hits={len(local_by_id)}",
                f"external_hits={len(external_results)}",
                f"online_verified={sum(1 for track in external_results if self._is_verified_online_track(track))}",
                f"fallback_hits={sum(1 for track in external_results if self._is_fallback_track(track))}",
            ],
        )

    def search_videos(self, query: str, top_k: int = 5) -> list[ExternalTrack]:
        from app.concurrency import run_parallel

        def fetch_bili() -> list[ExternalTrack]:
            out: list[ExternalTrack] = []
            for item in bilibili_source.search_bilibili_many(query, limit=min(top_k, 5)):
                out.append(
                    ExternalTrack(
                        external_id=item["bvid"],
                        title=item["title"],
                        artist=item.get("author", ""),
                        source="bilibili",
                        candidate_kind=self._classify_candidate_kind(item["title"], "bilibili"),
                        playback_url=f"https://player.bilibili.com/player.html?bvid={item['bvid']}&autoplay=0&high_quality=1&danmaku=0",
                    )
                )
            return out

        def fetch_youtube() -> list[ExternalTrack]:
            out: list[ExternalTrack] = []
            for item in youtube_source.search_youtube_many(query, limit=min(top_k, 3)):
                vid = item["video_id"]
                title = (
                    item.get("title")
                    or youtube_source.fetch_youtube_title(f"https://www.youtube.com/watch?v={vid}")
                    or ""
                )
                out.append(
                    ExternalTrack(
                        external_id=vid,
                        title=title,
                        artist="",
                        source="youtube",
                        candidate_kind=self._classify_candidate_kind(title, "youtube"),
                        playback_url=f"https://www.youtube.com/embed/{vid}?autoplay=1&rel=0",
                    )
                )
            return out

        bili_tracks, yt_tracks = run_parallel(
            [("bilibili", fetch_bili), ("youtube", fetch_youtube)],
            default=[],
        )
        tracks: list[ExternalTrack] = [*(bili_tracks or []), *(yt_tracks or [])]
        seen: set[tuple[str, str]] = set()
        unique: list[ExternalTrack] = []
        for track in tracks:
            key = (track.source, track.external_id)
            if key in seen:
                continue
            seen.add(key)
            unique.append(track)
        return unique[:top_k]

    async def search_videos_async(self, query: str, top_k: int = 5) -> list[ExternalTrack]:
        try:
            bili_items, youtube_items = await asyncio.gather(
                bilibili_source.asearch_bilibili_many(query, limit=min(top_k, 5)),
                youtube_source.asearch_youtube_many(query, limit=min(top_k, 3)),
            )
        except Exception:
            logger.debug("async video search failed; falling back to sync path for query=%s", query, exc_info=True)
            return await asyncio.to_thread(self._sync_search_videos, query=query, top_k=top_k)
        tracks = [
            ExternalTrack(
                external_id=item["bvid"],
                title=item["title"],
                artist=item.get("author", ""),
                source="bilibili",
                candidate_kind=self._classify_candidate_kind(item["title"], "bilibili"),
                playback_url=f"https://player.bilibili.com/player.html?bvid={item['bvid']}&autoplay=0&high_quality=1&danmaku=0",
            )
            for item in bili_items
        ]
        for item in youtube_items:
            title = item.get("title") or await youtube_source.afetch_youtube_title(item["video_id"])
            tracks.append(
                ExternalTrack(
                    external_id=item["video_id"],
                    title=title,
                    artist="",
                    source="youtube",
                    candidate_kind=self._classify_candidate_kind(title, "youtube"),
                    playback_url=f"https://www.youtube.com/embed/{item['video_id']}?autoplay=1&rel=0",
                )
            )
        selected = self._dedupe_tracks(tracks)[:top_k]
        if selected:
            return selected
        return await asyncio.to_thread(self._sync_search_videos, query=query, top_k=top_k)

    def search_artist_info(self, query: str) -> list[dict[str, str]]:
        return web_search_source.search_web_info(
            query,
            max_results=5,
            api_key=settings.tavily_api_key,
        )

    async def search_artist_info_async(self, query: str) -> list[dict[str, str]]:
        try:
            result = await web_search_source.asearch_web_info(
                query,
                max_results=5,
                api_key=settings.tavily_api_key,
            )
        except Exception:
            logger.debug(
                "async artist info search failed; falling back to sync path for query=%s", query, exc_info=True
            )
            return await asyncio.to_thread(self._sync_search_artist_info, query=query)
        if result:
            return result
        return await asyncio.to_thread(self._sync_search_artist_info, query=query)

    def classify_discover_query(self, query: str) -> dict[str, Any]:
        from app.graph.tag_rules import extract_tags

        raw = (query or "").strip()
        normalized = self._extract_search_query(raw).strip() or raw
        tags = extract_tags(raw)
        artist_cues = ("歌手", "艺人", "乐队", "组合", "artist", "band")
        explicit_artist = any(cue in raw.lower() for cue in artist_cues)
        normalized_key = self._normalize_match_text(normalized)
        artist_catalog: dict[str, set[str]] = {}
        for asset in self._list_assets():
            if not asset.artist:
                continue
            for artist_name in self._artist_credit_parts(asset.artist):
                artist_catalog.setdefault(artist_name, set()).update(self._artist_alias_keys(artist_name))

        exact_artist = next(
            (name for name, aliases in artist_catalog.items() if normalized_key and normalized_key in aliases),
            "",
        )
        exact_track = next(
            (
                asset.title
                for asset in self._list_assets()
                if normalized_key and self._normalize_match_text(asset.title) == normalized_key
            ),
            "",
        )

        if explicit_artist or exact_artist:
            canonical = exact_artist or normalized
            return {
                "kind": "artist",
                "normalized_query": canonical,
                "label": "歌手档案",
                "tags": tags,
                "confidence": 0.98 if exact_artist else 0.92,
                "matched_artist": exact_artist,
                "reason": "explicit_artist" if explicit_artist and not exact_artist else "library_artist_exact",
            }

        if exact_track:
            return {
                "kind": "track",
                "normalized_query": exact_track,
                "label": "歌曲搜索",
                "tags": tags,
                "confidence": 0.97,
                "reason": "library_track_exact",
            }

        category_order = (
            ("scenario", "scene", "场景电台"),
            ("mood", "mood", "情绪电台"),
            ("genre", "genre", "曲风探索"),
        )
        for tag_key, browse_category, label in category_order:
            if tags.get(tag_key):
                values = [*tags.get("genre", []), *tags.get("mood", []), *tags.get("scenario", [])]
                browse_value = " ".join(dict.fromkeys(values))
                return {
                    "kind": "category",
                    "normalized_query": normalized,
                    "label": label,
                    "browse_category": browse_category,
                    "browse_value": browse_value or tags[tag_key][0],
                    "tags": tags,
                    "confidence": 0.96,
                    "reason": f"tag:{tag_key}",
                }

        if len(normalized_key) >= 6 and re.search(r"[a-z]", normalized_key) and artist_catalog:
            scored: list[tuple[float, str]] = []
            for artist_name, aliases in artist_catalog.items():
                score = max((self._string_similarity(normalized_key, alias) for alias in aliases), default=0.0)
                scored.append((score, artist_name))
            scored.sort(key=lambda item: (-item[0], item[1].lower()))
            best_score, best_artist = scored[0]
            second_score = scored[1][0] if len(scored) > 1 else 0.0
            if best_score >= 88.0 and best_score - second_score >= 4.0:
                return {
                    "kind": "artist",
                    "normalized_query": best_artist,
                    "label": "歌手档案",
                    "tags": tags,
                    "confidence": round(best_score / 100.0, 3),
                    "matched_artist": best_artist,
                    "reason": "library_artist_fuzzy",
                }

        if self._looks_like_bare_artist_query(raw, normalized, tags):
            return {
                "kind": "artist",
                "normalized_query": normalized,
                "label": "歌手档案",
                "tags": tags,
                "confidence": 0.74,
                "matched_artist": "",
                "reason": "bare_artist_shape",
            }

        return {
            "kind": "track",
            "normalized_query": normalized,
            "label": "歌曲搜索",
            "tags": tags,
            "confidence": 0.55,
            "reason": "default_track_search",
        }
