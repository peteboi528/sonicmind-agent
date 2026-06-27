from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable
from typing import Any

from app.llm.protocol import LLMError
from app.llm.structured import extract_json_dict, extract_json_list
from app.models import Asset, ExternalTrack, Playlist
from app.prompts import AUTO_PLAYLIST_TEMPLATE, GENERATE_PLAYLIST_TEMPLATE

logger = logging.getLogger(__name__)


class PlaylistService:
    def __init__(
        self,
        *,
        store: Any,
        memory: Any,
        llm: Any,
        list_assets: Callable[[], list[Asset]],
        search_web_music: Callable[..., list[ExternalTrack]],
        source: Any,
        summarize_taste: Callable[..., str],
        query_has_entity: Callable[[str], bool],
    ) -> None:
        self.store = store
        self.memory = memory
        self.llm = llm
        self._list_assets = list_assets
        self._search_web_music = search_web_music
        self.source = source
        self._summarize_taste = summarize_taste
        self._query_has_entity = query_has_entity

    def generate_playlist(
        self,
        user_id: str,
        instruction: str,
        *,
        seed_tracks: list[Asset | ExternalTrack] | None = None,
        target_count: int | None = None,
        infer_playlist_count: Callable[[str], int | None],
        playlist_candidates_builder: Callable[[str, list[Asset], list[Asset | ExternalTrack], int], list[Asset | ExternalTrack]],
        extract_search_query: Callable[[str], str],
        track_key: Callable[[Any], str],
        dedupe_tracks: Callable[[list[Asset | ExternalTrack]], list[Asset | ExternalTrack]],
        is_quality_track: Callable[[Any], bool],
        is_playlist_context_compatible: Callable[[str, Any], bool],
        query_requests_variant_content: Callable[[str], bool],
    ) -> Playlist:
        target_count = target_count or infer_playlist_count(instruction) or 12
        target_count = max(1, min(target_count, 100))
        seed_tracks = seed_tracks or []
        library = self._list_assets()
        candidates = playlist_candidates_builder(instruction, library, seed_tracks, target_count)
        lib_desc = "\n".join(
            f"- {asset.asset_id}: {asset.title} - {asset.artist or '?'} "
            f"({', '.join(asset.genre)}, {', '.join(asset.mood)}, energy={asset.energy_level})"
            for asset in library[:120]
        )
        candidate_desc = "\n".join(
            f"- {track.title} - {getattr(track, 'artist', '') or '?'} ({getattr(track, 'source', 'local')})"
            for track in candidates[:120]
        )
        memory = self.memory.get_memory(user_id)
        explicit_artist = self._query_has_entity(extract_search_query(instruction))
        taste_summary = self._summarize_taste(user_id, include_artists=explicit_artist) if memory.taste_profile else ""
        exclusion_rules = memory.exclusion_rules or None

        prompt = GENERATE_PLAYLIST_TEMPLATE(
            instruction=instruction,
            library_size=len(library),
            lib_desc=lib_desc,
            target_count=target_count,
            candidate_desc=candidate_desc,
            taste_summary=taste_summary,
            exclusion_rules=exclusion_rules,
        )
        try:
            result = self.llm.generate(prompt)
            data = extract_json_dict(result)
        except LLMError:
            logger.debug("Playlist generation LLM call failed; using fallback", exc_info=True)
            data = None
        if not data:
            return self.fallback_playlist(
                user_id,
                instruction,
                library,
                target_count=target_count,
                candidates=candidates,
                save_playlist=self.save_playlist,
                is_quality_track=is_quality_track,
                query_requests_variant_content=query_requests_variant_content,
                fill_tracks=self._fill_tracks,
            )

        asset_map = {asset.asset_id: asset for asset in library}
        candidate_map = {track_key(track): track for track in candidates}
        tracks: list[Asset | ExternalTrack] = []
        for item in data.get("tracks", []):
            asset_id = item.get("asset_id")
            if asset_id and asset_id in asset_map:
                tracks.append(asset_map[asset_id])
            elif track_key(item) in candidate_map:
                tracks.append(candidate_map[track_key(item)])
        tracks = dedupe_tracks(tracks)
        allow_variants = query_requests_variant_content(instruction)
        tracks = [
            track for track in tracks
            if is_quality_track(track, allow_variants=allow_variants)
            and is_playlist_context_compatible(instruction, track)
        ]
        clean_candidates = [
            track for track in candidates
            if is_quality_track(track, allow_variants=allow_variants)
            and is_playlist_context_compatible(instruction, track)
        ]
        tracks = self._fill_tracks(tracks, clean_candidates, target_count)

        playlist = Playlist(
            playlist_id=hashlib.sha1(f"{user_id}-{instruction}".encode()).hexdigest()[:8],
            user_id=user_id,
            name=data.get("name", instruction),
            description=data.get("description", ""),
            tracks=tracks[:target_count],
            generated_by="llm",
        )
        self.save_playlist(user_id, playlist)
        return playlist

    def auto_playlists(self, user_id: str, *, fallback_auto_playlists: Callable[[str, list[Asset]], list[Playlist]]) -> list[Playlist]:
        library = self._list_assets()
        if not library:
            return []
        lib_desc = "\n".join(
            f"- {asset.asset_id}: {asset.title} - {asset.artist or '?'} ({', '.join(asset.genre)}, {', '.join(asset.mood)})"
            for asset in library
        )
        prompt = AUTO_PLAYLIST_TEMPLATE(library_size=len(library), lib_desc=lib_desc)
        try:
            result = self.llm.generate(prompt)
            raw = extract_json_list(result)
        except LLMError:
            logger.debug("Auto playlist LLM call failed; using fallback", exc_info=True)
            raw = None
        if not raw:
            return fallback_auto_playlists(user_id, library)
        asset_map = {asset.asset_id: asset for asset in library}
        playlists: list[Playlist] = []
        for item in raw:
            tracks = [asset_map[track_id] for track_id in item.get("track_ids", []) if track_id in asset_map]
            playlist = Playlist(
                playlist_id=hashlib.sha1(f"{user_id}-{item.get('name', '')}".encode()).hexdigest()[:8],
                user_id=user_id,
                name=item.get("name", ""),
                description=item.get("description", ""),
                tracks=tracks,
                generated_by="auto",
            )
            self.save_playlist(user_id, playlist)
            playlists.append(playlist)
        return playlists

    def save_playlist(self, user_id: str, playlist: Playlist) -> None:
        self.store.write_model("playlists", f"{user_id}_{playlist.playlist_id}", playlist)

    def list_playlists(self, user_id: str) -> list[Playlist]:
        playlists: list[Playlist] = []
        for key in self.store.list_keys("playlists"):
            if not key.startswith(f"{user_id}_"):
                continue
            try:
                playlist = self.store.read_model("playlists", key, Playlist)
            except Exception:
                logger.warning("Skipping unreadable playlist %s (stale schema?)", key, exc_info=True)
                continue
            if playlist:
                playlists.append(playlist)
        return playlists

    def delete_playlist(self, user_id: str, playlist_id: str) -> bool:
        return self.store.delete_key("playlists", f"{user_id}_{playlist_id}")

    def playlist_candidates(
        self,
        instruction: str,
        library: list[Asset],
        seed_tracks: list[Asset | ExternalTrack],
        target_count: int,
        *,
        playlist_search_terms: Callable[[str], list[str]],
        extract_search_query: Callable[[str], str],
        query_requests_variant_content: Callable[[str], bool],
        is_quality_track: Callable[[Any], bool],
        is_playlist_context_compatible: Callable[[str, Any], bool],
        is_scenario_playlist_instruction: Callable[[str], bool],
        curated_playlist_query: Callable[[str], str],
        playlist_online_queries: Callable[[list[str]], list[str]],
        playlist_match_score: Callable[[Asset, list[str]], float],
        dedupe_tracks: Callable[[list[Asset | ExternalTrack]], list[Asset | ExternalTrack]],
    ) -> list[Asset | ExternalTrack]:
        search_terms = playlist_search_terms(instruction)
        relevance_core = extract_search_query(instruction) or instruction
        allow_variants = query_requests_variant_content(instruction)
        clean_seed_tracks = [
            track for track in seed_tracks
            if is_quality_track(track, allow_variants=allow_variants)
            and is_playlist_context_compatible(instruction, track)
        ]
        external: list[ExternalTrack] = []

        if is_scenario_playlist_instruction(instruction):
            try:
                from app.search.netease_playlist import search_and_extract

                curated = search_and_extract(
                    curated_playlist_query(instruction),
                    max_playlists=3,
                    tracks_per_playlist=max(target_count, 12),
                )
                external.extend(
                    track for track in curated
                    if is_quality_track(track, allow_variants=allow_variants)
                    and is_playlist_context_compatible(instruction, track)
                )
            except Exception:
                logger.debug("curated playlist recall failed for %s", instruction, exc_info=True)

        for online_query in playlist_online_queries(search_terms):
            if len(dedupe_tracks([*clean_seed_tracks, *external])) >= target_count:
                break
            batch = self._search_web_music(
                online_query,
                top_k=min(max(target_count, 8), 25),
                relevance_query=relevance_core,
            )
            external.extend(
                track for track in batch
                if is_quality_track(track, allow_variants=allow_variants)
                and is_playlist_context_compatible(instruction, track)
            )

        if len(dedupe_tracks([*clean_seed_tracks, *external])) < target_count:
            source_tracks = self.source.get_recommendations(
                seed_genres=["流行", "民谣", "R&B", "说唱", "电子"],
                seed_moods=["放松", "治愈", "浪漫", "伤感"],
                limit=max(target_count * 2, 40),
            )
            external.extend(
                track for track in source_tracks
                if is_quality_track(track, allow_variants=allow_variants)
                and is_playlist_context_compatible(instruction, track)
            )

        library_ranked = sorted(
            [
                track for track in library
                if is_quality_track(track, allow_variants=allow_variants)
                and is_playlist_context_compatible(instruction, track)
            ],
            key=lambda asset: playlist_match_score(asset, search_terms),
            reverse=True,
        )
        ordered: list[Asset | ExternalTrack] = [*clean_seed_tracks, *external, *library_ranked]
        return dedupe_tracks(ordered)

    def fallback_playlist(
        self,
        user_id: str,
        instruction: str,
        library: list[Asset],
        *,
        target_count: int | None,
        candidates: list[Asset | ExternalTrack] | None,
        save_playlist: Callable[[str, Playlist], None],
        is_quality_track: Callable[[Any], bool],
        query_requests_variant_content: Callable[[str], bool],
        fill_tracks: Callable[[list[Asset | ExternalTrack], list[Asset | ExternalTrack], int], list[Asset | ExternalTrack]],
    ) -> Playlist:
        target_count = target_count or 12
        keywords = instruction.lower().split()
        matched = [
            asset for asset in library
            if any(term in f"{asset.title} {asset.artist or ''} {' '.join(asset.genre)} {' '.join(asset.mood)}".lower()
                   for term in keywords)
        ]
        if not matched:
            matched = sorted(library, key=lambda asset: (asset.energy_level or 0.0, asset.updated_at), reverse=True)
        allow_variants = query_requests_variant_content(instruction)
        matched = [track for track in matched if is_quality_track(track, allow_variants=allow_variants)]
        clean_candidates = [track for track in (candidates or []) if is_quality_track(track, allow_variants=allow_variants)]
        tracks = fill_tracks(matched, clean_candidates, target_count)
        playlist = Playlist(
            playlist_id=hashlib.sha1(f"{user_id}-{instruction}".encode()).hexdigest()[:8],
            user_id=user_id,
            name=instruction or "Agent 歌单",
            description="离线回退歌单：根据你的音乐库和指令自动整理。",
            tracks=tracks[:target_count],
            generated_by="fallback",
        )
        save_playlist(user_id, playlist)
        return playlist

    def fallback_auto_playlists(self, user_id: str, library: list[Asset]) -> list[Playlist]:
        grouped: dict[str, list[Asset]] = {}
        for asset in library:
            bucket = asset.genre[0] if asset.genre else "未分类"
            grouped.setdefault(bucket, []).append(asset)
        playlists: list[Playlist] = []
        for genre, tracks in list(grouped.items())[:4]:
            playlist = Playlist(
                playlist_id=hashlib.sha1(f"{user_id}-{genre}".encode()).hexdigest()[:8],
                user_id=user_id,
                name=f"{genre}精选",
                description=f"按 {genre} 风格自动整理的离线歌单。",
                tracks=tracks[:10],
                generated_by="fallback-auto",
            )
            self.save_playlist(user_id, playlist)
            playlists.append(playlist)
        return playlists

    @staticmethod
    def _fill_tracks(
        tracks: list[Asset | ExternalTrack],
        candidates: list[Asset | ExternalTrack],
        target_count: int,
    ) -> list[Asset | ExternalTrack]:
        out = list(tracks)
        seen = {PlaylistService._fallback_track_key(track) for track in out}
        for candidate in candidates:
            key = PlaylistService._fallback_track_key(candidate)
            if key in seen:
                continue
            out.append(candidate)
            seen.add(key)
            if len(out) >= target_count:
                break
        return out[:target_count]

    @staticmethod
    def _fallback_track_key(track: Any) -> str:
        asset_id = getattr(track, "asset_id", "") or ""
        external_id = getattr(track, "external_id", "") or ""
        title = getattr(track, "title", "") or ""
        artist = getattr(track, "artist", "") or ""
        if asset_id:
            return f"asset:{asset_id}"
        if external_id:
            return f"external:{getattr(track, 'source', '')}:{external_id}"
        return f"title:{title.lower()}:{artist.lower()}"
