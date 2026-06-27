from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Callable
from typing import Any

from app.config import settings
from app.graph.tag_rules import extract_tags
from app.models import ExternalTrack, ResourceTrack
from app.sources import bilibili as bilibili_source
from app.sources import youtube as youtube_source
from app.sources.mock_source import MockSource

logger = logging.getLogger(__name__)


class SearchService:
    def __init__(
        self,
        *,
        library: Any,
        source: Any,
        track_key: Callable[[Any], str],
        dedupe_tracks: Callable[[list[Any]], list[Any]],
        merge_search_queries: Callable[[str, list[str] | None], list[str]],
        classify_candidate_kind: Callable[[str, str], str],
        valid_external_track: Callable[[ExternalTrack, str], bool],
        sync_search_web_music: Callable[..., list[ExternalTrack]],
        search_videos: Callable[[str, int], list[ExternalTrack]],
        search_videos_async: Callable[[str, int], Any],
        search_bilibili_detail: Callable[[str], dict[str, Any] | None],
        search_youtube_video: Callable[[str], str | None],
        lexical_query_noise: set[str],
    ) -> None:
        self.library = library
        self.source = source
        self._track_key = track_key
        self._dedupe_tracks = dedupe_tracks
        self._merge_search_queries = merge_search_queries
        self._classify_candidate_kind = classify_candidate_kind
        self._valid_external_track = valid_external_track
        self._sync_search_web_music = sync_search_web_music
        self._search_videos = search_videos
        self._search_videos_async = search_videos_async
        self._search_bilibili_detail = search_bilibili_detail
        self._search_youtube_video = search_youtube_video
        self._lexical_query_noise = lexical_query_noise

    def search_web_music(
        self,
        query: str,
        top_k: int = 5,
        relevance_query: str = "",
        include_video_sources: bool = False,
        offset: int = 0,
        variants: list[str] | None = None,
    ) -> list[ExternalTrack]:
        query_list = self._merge_search_queries(query, variants)
        if len(query_list) > 1:
            from app.concurrency import run_parallel

            tasks = [
                (
                    f"search_variant:{idx}:{q}",
                    lambda q=q, idx=idx: self.search_web_music(
                        q,
                        top_k=max(top_k, 3),
                        relevance_query=relevance_query or query,
                        include_video_sources=include_video_sources,
                        offset=offset if idx == 0 else 0,
                        variants=None,
                    ),
                )
                for idx, q in enumerate(query_list)
            ]
            merged: list[ExternalTrack] = []
            for batch in run_parallel(tasks, timeout=8.0, default=[]):
                merged.extend(batch or [])
            selected = self._dedupe_tracks(merged)[:top_k]
            for track in selected:
                self.library.upsert_external(track)
            return selected

        tracks: list[ExternalTrack] = []
        try:
            from app.sources.netease import search_netease_many

            for meta in search_netease_many(query, limit=top_k, offset=offset):
                if not meta.get("title"):
                    continue
                tracks.append(
                    ExternalTrack(
                        external_id=meta["song_id"],
                        title=meta["title"],
                        artist=meta.get("artist", ""),
                        album=meta.get("album"),
                        cover_url=meta.get("cover"),
                        source="netease",
                        candidate_kind=self._classify_candidate_kind(meta["title"], "netease"),
                        playback_url=f"https://music.163.com/song?id={meta['song_id']}",
                    )
                )
        except Exception:
            logger.debug("NetEase web music search failed for query=%s", query, exc_info=True)

        if include_video_sources:
            self._extend_video_tracks(tracks, query, top_k)

        rel_q = relevance_query or query
        tracks = [track for track in tracks if self._valid_external_track(track, rel_q)]

        if len(tracks) < top_k:
            tracks.extend(self.dense_library_fallback(query=rel_q, existing=tracks, limit=top_k - len(tracks)))

        if len(tracks) < top_k:
            for candidate in self.source.search(query, limit=top_k - len(tracks)):
                fallback = candidate.model_copy(update={"source": f"{candidate.source}-fallback"})
                if self._valid_external_track(fallback, rel_q):
                    tracks.append(fallback)

        selected = self._unique_source_tracks(tracks)[:top_k]
        for track in selected:
            self.library.upsert_external(track)
        return selected

    async def search_web_music_async(
        self,
        query: str,
        top_k: int = 5,
        relevance_query: str = "",
        include_video_sources: bool = False,
        offset: int = 0,
        variants: list[str] | None = None,
    ) -> list[ExternalTrack]:
        query_list = self._merge_search_queries(query, variants)
        if len(query_list) > 1:
            batches = await asyncio.gather(
                *(
                    self.search_web_music_async(
                        item,
                        top_k=max(top_k, 3),
                        relevance_query=relevance_query or query,
                        include_video_sources=include_video_sources,
                        offset=offset if index == 0 else 0,
                        variants=None,
                    )
                    for index, item in enumerate(query_list)
                )
            )
            selected = self._dedupe_tracks([track for batch in batches for track in batch])[:top_k]
            await asyncio.gather(*(asyncio.to_thread(self.library.upsert_external, track) for track in selected))
            return selected

        from app.sources.netease import asearch_netease_many

        try:
            metadata = await asearch_netease_many(query, limit=top_k, offset=offset)
        except Exception:
            logger.debug("async web music search failed; falling back to sync path for query=%s", query, exc_info=True)
            return await asyncio.to_thread(
                self._sync_search_web_music,
                query=query,
                top_k=top_k,
                relevance_query=relevance_query,
                include_video_sources=include_video_sources,
                offset=offset,
                variants=variants,
            )
        tracks = [
            ExternalTrack(
                external_id=item["song_id"],
                title=item["title"],
                artist=item.get("artist", ""),
                album=item.get("album"),
                cover_url=item.get("cover"),
                source="netease",
                candidate_kind=self._classify_candidate_kind(item["title"], "netease"),
                playback_url=f"https://music.163.com/song?id={item['song_id']}",
            )
            for item in metadata
            if item.get("title")
        ]
        if include_video_sources and len(tracks) < top_k:
            video_tracks = await self._search_videos_async(query, top_k=top_k - len(tracks))
            tracks.extend(video_tracks)
        rel_q = relevance_query or query
        tracks = [track for track in tracks if self._valid_external_track(track, rel_q)]
        if len(tracks) < top_k:
            tracks.extend(await asyncio.to_thread(self.dense_library_fallback, rel_q, tracks, top_k - len(tracks)))
        if len(tracks) < top_k and isinstance(self.source, MockSource):
            for candidate in self.source.search(query, limit=top_k - len(tracks)):
                fallback = candidate.model_copy(update={"source": f"{candidate.source}-fallback"})
                if self._valid_external_track(fallback, rel_q):
                    tracks.append(fallback)
        selected = self._dedupe_tracks(tracks)[:top_k]
        if not selected:
            return await asyncio.to_thread(
                self._sync_search_web_music,
                query=query,
                top_k=top_k,
                relevance_query=relevance_query,
                include_video_sources=include_video_sources,
                offset=offset,
                variants=variants,
            )
        await asyncio.gather(*(asyncio.to_thread(self.library.upsert_external, track) for track in selected))
        return selected

    def dense_library_fallback(self, query: str, existing: list[ExternalTrack], limit: int = 5) -> list[ExternalTrack]:
        if limit <= 0:
            return []
        try:
            existing_keys = {self._track_key(track) for track in existing}
            hits = self.library.semantic_search(
                query,
                limit=max(limit * 2, limit),
                min_score=settings.dense_recall_min_score,
            )
            if not hits:
                hits = self.lexical_resource_fallback(query, limit=max(limit * 2, limit))
            out: list[ExternalTrack] = []
            for item in hits:
                track = ExternalTrack(
                    external_id=item.source_id or f"library:{item.title}:{item.artist}",
                    title=item.title,
                    artist=item.artist,
                    genre=item.genre,
                    mood=item.mood,
                    playback_url=item.playback_url,
                    source=item.source,
                    candidate_kind="track",
                )
                if self._track_key(track) in existing_keys:
                    continue
                out.append(track)
                existing_keys.add(self._track_key(track))
                if len(out) >= limit:
                    break
            return out
        except Exception:
            logger.debug("dense library fallback failed for query=%s", query, exc_info=True)
            return []

    def lexical_resource_fallback(self, query: str, limit: int = 10) -> list[ResourceTrack]:
        tags = extract_tags(query)
        wanted_genres = {item.lower() for item in tags["genre"]}
        wanted_moods = {item.lower() for item in tags["mood"]}
        wanted_scenarios = {item.lower() for item in tags["scenario"]}
        terms = {
            item.lower()
            for item in re.findall(r"[A-Za-z0-9&'-]+|[一-鿿㐀-䶿]{2,}", query or "")
            if item.lower() not in self._lexical_query_noise
        }
        scenario_moods = {
            "深夜": {"放松", "宁静", "孤独", "慵懒", "治愈"},
            "睡眠": {"放松", "宁静", "舒缓"},
            "学习": {"专注", "宁静", "放松"},
            "工作": {"专注", "放松"},
        }
        for scenario in wanted_scenarios:
            wanted_moods.update(item.lower() for item in scenario_moods.get(scenario, set()))

        ranked: list[tuple[float, ResourceTrack]] = []
        for track in self.library.list_tracks(1500, verified_only=True):
            genres = {item.lower() for item in track.genre}
            moods = {item.lower() for item in track.mood}
            searchable = " ".join([track.title, track.artist, *track.genre, *track.mood]).lower()
            score = len(wanted_genres & genres) * 4.0 + len(wanted_moods & moods) * 3.0
            score += sum(1.0 for term in terms if term in searchable)
            if score > 0:
                ranked.append((score, track))
        ranked.sort(key=lambda item: (-item[0], item[1].exposure_count, item[1].title.lower()))
        return [track for _, track in ranked[:limit]]

    def _extend_video_tracks(self, tracks: list[ExternalTrack], query: str, top_k: int) -> None:
        if len(tracks) < top_k:
            try:
                bili = self._search_bilibili_detail(query)
                if bili and bili.get("title"):
                    tracks.append(
                        ExternalTrack(
                            external_id=bili["bvid"],
                            title=bili["title"],
                            artist=bili.get("author", ""),
                            source="bilibili",
                            candidate_kind=self._classify_candidate_kind(bili["title"], "bilibili"),
                            playback_url=f"https://player.bilibili.com/player.html?bvid={bili['bvid']}&autoplay=0&high_quality=1&danmaku=0",
                        )
                    )
            except Exception:
                logger.debug("Bilibili web music search failed for query=%s", query, exc_info=True)

        if len(tracks) < top_k:
            try:
                video_id = self._search_youtube_video(query)
                if video_id:
                    url = f"https://www.youtube.com/watch?v={video_id}"
                    title = youtube_source.fetch_youtube_title(url)
                    if title:
                        tracks.append(
                            ExternalTrack(
                                external_id=video_id,
                                title=title,
                                artist="",
                                source="youtube",
                                candidate_kind=self._classify_candidate_kind(title, "youtube"),
                                playback_url=f"https://www.youtube.com/embed/{video_id}?autoplay=1&rel=0",
                            )
                        )
            except Exception:
                logger.debug("YouTube web music search failed for query=%s", query, exc_info=True)

    @staticmethod
    def _unique_source_tracks(tracks: list[ExternalTrack]) -> list[ExternalTrack]:
        seen: set[tuple[str, str]] = set()
        unique: list[ExternalTrack] = []
        for track in tracks:
            key = (track.source, track.external_id)
            if key in seen:
                continue
            seen.add(key)
            unique.append(track)
        return unique
