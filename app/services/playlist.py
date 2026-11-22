from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable
from typing import Any
from uuid import uuid4

from app.intents import extract_content_negations
from app.llm.protocol import LLMError
from app.llm.structured import extract_json_dict, extract_json_list
from app.models import Asset, ExternalTrack, Playlist, PlaylistDraft, PlaylistRevisionResult, utc_now_iso
from app.prompts import AUTO_PLAYLIST_TEMPLATE, GENERATE_PLAYLIST_TEMPLATE
from app.rules.content import filter_content_exclusions
from app.rules.track import _track_key, _track_title_key

logger = logging.getLogger(__name__)


def _draft_track_keys(track: Asset | ExternalTrack) -> set[str]:
    recording = json.dumps(_track_title_key(track.title, track.artist), ensure_ascii=False)
    return {_track_key(track), f"recording:{recording}"}


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

    def build_playlist(
        self,
        user_id: str,
        instruction: str,
        *,
        seed_tracks: list[Asset | ExternalTrack] | None = None,
        target_count: int,
        eligible: Callable[[Asset | ExternalTrack], bool] | None = None,
        playlist_candidates_builder: Callable[
            [str, list[Asset], list[Asset | ExternalTrack], int], list[Asset | ExternalTrack]
        ],
        extract_search_query: Callable[[str], str],
        dedupe_tracks: Callable[[list[Asset | ExternalTrack]], list[Asset | ExternalTrack]],
        is_quality_track: Callable[[Any], bool],
        is_playlist_context_compatible: Callable[[str, Any], bool],
        query_requests_variant_content: Callable[[str], bool],
    ) -> Playlist:
        memory = self.memory.get_memory(user_id)
        exclusion_rules = list(
            dict.fromkeys([*(memory.exclusion_rules or []), *extract_content_negations(instruction)])
        )
        library = filter_content_exclusions(self._list_assets(), exclusion_rules)
        if eligible is not None:
            library = [track for track in library if eligible(track)]
        seeds = filter_content_exclusions(seed_tracks or [], exclusion_rules)
        candidates = playlist_candidates_builder(instruction, library, seeds, target_count)
        candidates = filter_content_exclusions(candidates, exclusion_rules)
        if eligible is not None:
            candidates = [track for track in candidates if eligible(track)]
        lib_desc = "\n".join(
            f"- {asset.asset_id}: {asset.title} - {asset.artist or '?'} "
            f"({', '.join(asset.genre)}, {', '.join(asset.mood)}, energy={asset.energy_level})"
            for asset in library[:120]
        )
        candidate_desc = "\n".join(
            f"- {track.title} - {getattr(track, 'artist', '') or '?'} ({getattr(track, 'source', 'local')})"
            for track in candidates[:120]
        )
        explicit_artist = self._query_has_entity(extract_search_query(instruction))
        taste_summary = self._summarize_taste(user_id, include_artists=explicit_artist) if memory.taste_profile else ""

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
        if not data or not isinstance(data.get("tracks"), list):
            return self.fallback_playlist(
                user_id,
                instruction,
                library if eligible is None else [],
                target_count=target_count,
                candidates=candidates,
                save_playlist=None,
                is_quality_track=is_quality_track,
                query_requests_variant_content=query_requests_variant_content,
                fill_tracks=self._fill_tracks,
            )

        asset_map = {asset.asset_id: asset for asset in library}
        candidate_map = {}
        for track in candidates:
            candidate_map.setdefault(_track_title_key(track.title, track.artist), track)
        tracks: list[Asset | ExternalTrack] = []
        for item in data["tracks"]:
            if not isinstance(item, dict):
                continue
            asset_id = item.get("asset_id")
            if isinstance(asset_id, str) and asset_id in asset_map:
                tracks.append(asset_map[asset_id])
            else:
                key = _track_title_key(str(item.get("title") or ""), str(item.get("artist") or ""))
                if key in candidate_map:
                    tracks.append(candidate_map[key])
        tracks = dedupe_tracks(tracks)
        allow_variants = query_requests_variant_content(instruction)
        tracks = [
            track
            for track in tracks
            if is_quality_track(track, allow_variants=allow_variants)
            and is_playlist_context_compatible(instruction, track)
        ]
        clean_candidates = [
            track
            for track in candidates
            if is_quality_track(track, allow_variants=allow_variants)
            and is_playlist_context_compatible(instruction, track)
        ]
        tracks = dedupe_tracks([*tracks, *clean_candidates])[:target_count]
        return Playlist(
            playlist_id=hashlib.sha1(f"{user_id}-{instruction}".encode()).hexdigest()[:8],
            user_id=user_id,
            name=str(data.get("name") or instruction),
            description=str(data.get("description") or ""),
            tracks=tracks,
            generated_by="llm",
        )

    def save_draft(self, playlist: Playlist, instruction: str, target_count: int) -> PlaylistDraft:
        draft = PlaylistDraft(
            **playlist.model_dump(exclude={"playlist_id"}),
            playlist_id=uuid4().hex,
            instruction=instruction,
            target_count=target_count,
        )
        with self.store.lock("playlist_drafts", draft.user_id):
            self.store.write_model("playlist_drafts", draft.user_id, draft)
        return draft

    def _read_draft(self, user_id: str, draft_id: str) -> PlaylistDraft:
        draft = self.store.read_model("playlist_drafts", user_id, PlaylistDraft)
        if draft is None or draft.user_id != user_id or draft.playlist_id != draft_id:
            raise LookupError("草稿已更新，请重新预览")
        return draft

    def _assert_unsaved(self, user_id: str, draft_id: str) -> None:
        if self.store.read_model("playlists", f"{user_id}_{draft_id}", Playlist) is not None:
            raise LookupError("这份草稿已保存，请生成新草稿")

    def revise_draft(
        self,
        user_id: str,
        draft_id: str,
        kept_indices: list[int],
        adjustment: str,
        *,
        generate: Callable[[str, int, Callable[[Asset | ExternalTrack], bool], str], Playlist],
    ) -> PlaylistRevisionResult:
        with self.store.lock("playlist_drafts", user_id):
            draft = self._read_draft(user_id, draft_id)
            self._assert_unsaved(user_id, draft_id)
        if any(i < 0 or i >= len(draft.tracks) for i in kept_indices):
            raise ValueError("请选择草稿中的曲目")
        kept = [draft.tracks[i] for i in sorted(set(kept_indices))]
        target_count = draft.target_count if draft.instruction else (len(draft.tracks) or 12)
        remaining = target_count - len(kept)
        if remaining <= 0:
            raise ValueError("请取消勾选要换掉的曲目")

        seen_keys = draft.seen_track_keys.union(*(_draft_track_keys(track) for track in draft.tracks))

        def eligible(track: Asset | ExternalTrack) -> bool:
            return seen_keys.isdisjoint(_draft_track_keys(track))

        adjustments = draft.adjustments
        if adjustment and (not adjustments or adjustment != adjustments[-1]):
            adjustments = [*adjustments, adjustment][-10:]
        scene = draft.instruction or draft.name
        instruction = scene
        if adjustments:
            instruction += "\n调整要求（冲突时以最后一条为准）：" + "；".join(adjustments)
        if len(filter_content_exclusions(kept, extract_content_negations(instruction))) != len(kept):
            raise ValueError("保留的曲目与排除要求冲突，请调整选择")
        replacement = generate(instruction, remaining, eligible, adjustments[-1] if adjustments else "")

        with self.store.lock("playlist_drafts", user_id):
            # 召回期间允许读取或保存。写回前检查版本，避免覆盖新草稿或已确认的版本。
            self._read_draft(user_id, draft_id)
            self._assert_unsaved(user_id, draft_id)
            if not replacement.tracks:
                return PlaylistRevisionResult(draft=draft, added_count=0)
            revised = draft.model_copy(
                update={
                    "playlist_id": uuid4().hex,
                    "tracks": [*kept, *replacement.tracks],
                    "instruction": scene,
                    "target_count": target_count,
                    "adjustments": adjustments,
                    "seen_track_keys": seen_keys,
                    "updated_at": utc_now_iso(),
                }
            )
            self.store.write_model("playlist_drafts", user_id, revised)
            return PlaylistRevisionResult(draft=revised, added_count=len(replacement.tracks))

    def confirm_draft(self, user_id: str, draft_id: str, indices: list[int], name: str) -> Playlist:
        with self.store.lock("playlist_drafts", user_id):
            draft = self._read_draft(user_id, draft_id)
            if not indices or any(i < 0 or i >= len(draft.tracks) for i in indices):
                raise ValueError("请选择草稿中的曲目")
            tracks = [draft.tracks[i] for i in sorted(set(indices))]
            selected = Playlist(
                **draft.model_dump(include=set(Playlist.model_fields) - {"name", "tracks", "description"}),
                name=name.strip() or draft.name,
                tracks=tracks,
                description=f"从场景草稿中选定的 {len(tracks)} 首",
            )
            saved = self.store.read_model("playlists", f"{user_id}_{draft_id}", Playlist)
            if saved is not None:
                if saved == selected:
                    return saved
                raise LookupError("这份草稿已保存，请生成新草稿")
            self.save_playlist(user_id, selected)
            return selected

    def auto_playlists(
        self, user_id: str, *, fallback_auto_playlists: Callable[[str, list[Asset]], list[Playlist]]
    ) -> list[Playlist]:
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

    def create_playlist_from_assets(
        self,
        user_id: str,
        name: str,
        asset_ids: list[str],
        *,
        description: str = "",
    ) -> Playlist:
        """从用户库内指定曲目直接建歌单（不走 LLM）：按传入顺序保留命中的曲目。"""
        name = (name or "").strip() or "我的歌单"
        asset_map = {asset.asset_id: asset for asset in self._list_assets()}
        tracks: list[Asset | ExternalTrack] = []
        seen: set[str] = set()
        for asset_id in asset_ids:
            asset = asset_map.get(asset_id)
            if asset and asset.asset_id not in seen:
                tracks.append(asset)
                seen.add(asset.asset_id)
        if not tracks:
            raise ValueError("no valid tracks for playlist")
        # id 含曲目签名：同名但选不同曲目不互相覆盖。
        sig = ",".join(sorted(seen))
        playlist = Playlist(
            playlist_id=hashlib.sha1(f"{user_id}-{name}-{sig}".encode()).hexdigest()[:8],
            user_id=user_id,
            name=name,
            description=description or f"手动挑选的 {len(tracks)} 首",
            tracks=tracks,
            generated_by="manual",
        )
        self.save_playlist(user_id, playlist)
        return playlist

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
            if playlist and playlist.user_id == user_id:
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
        eligible: Callable[[Asset | ExternalTrack], bool] | None = None,
        recall_hint: str = "",
        playlist_search_terms: Callable[[str], str],
        extract_search_query: Callable[[str], str],
        query_requests_variant_content: Callable[[str], bool],
        is_quality_track: Callable[[Any], bool],
        is_playlist_context_compatible: Callable[[str, Any], bool],
        is_scenario_playlist_instruction: Callable[[str], bool],
        curated_playlist_query: Callable[[str], str],
        playlist_online_queries: Callable[[str], list[str]],
        playlist_match_score: Callable[[Asset, str], float],
        dedupe_tracks: Callable[[list[Asset | ExternalTrack]], list[Asset | ExternalTrack]],
    ) -> list[Asset | ExternalTrack]:
        search_terms = playlist_search_terms(instruction)
        relevance_core = extract_search_query(instruction) or instruction
        allow_variants = query_requests_variant_content(instruction)
        clean_seed_tracks = [
            track
            for track in seed_tracks
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
                    track
                    for track in curated
                    if is_quality_track(track, allow_variants=allow_variants)
                    and is_playlist_context_compatible(instruction, track)
                    and (eligible is None or eligible(track))
                )
            except Exception:
                logger.debug("curated playlist recall failed for %s", instruction, exc_info=True)

        queries = playlist_online_queries(search_terms)
        searches = [(query, 0) for query in queries if query != recall_hint]
        if recall_hint:
            searches.insert(0, (recall_hint, 0))
        if eligible is not None and searches:
            searches.append((searches[0][0], 25))
        for index, (online_query, offset) in enumerate(searches):
            preferred = bool(recall_hint and index == 0)
            if not preferred and len(dedupe_tracks([*clean_seed_tracks, *external])) >= target_count:
                break
            batch = self._search_web_music(
                online_query,
                top_k=25 if eligible is not None else min(max(target_count, 8), 25),
                relevance_query=(extract_search_query(online_query) or online_query) if recall_hint else relevance_core,
                **({"offset": offset} if offset else {}),
            )
            usable = [
                track
                for track in batch
                if is_quality_track(track, allow_variants=allow_variants)
                and is_playlist_context_compatible(instruction, track)
                and (eligible is None or eligible(track))
            ]
            if preferred:
                external = [*usable, *external]
            else:
                external.extend(usable)

        if len(dedupe_tracks([*clean_seed_tracks, *external])) < target_count:
            source_tracks = self.source.get_recommendations(
                seed_genres=["流行", "民谣", "R&B", "说唱", "电子"],
                seed_moods=["放松", "治愈", "浪漫", "伤感"],
                limit=max(target_count * 2, 40),
            )
            external.extend(
                track
                for track in source_tracks
                if is_quality_track(track, allow_variants=allow_variants)
                and is_playlist_context_compatible(instruction, track)
                and (eligible is None or eligible(track))
            )

        library_ranked = sorted(
            [
                track
                for track in library
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
        save_playlist: Callable[[str, Playlist], None] | None,
        is_quality_track: Callable[[Any], bool],
        query_requests_variant_content: Callable[[str], bool],
        fill_tracks: Callable[
            [list[Asset | ExternalTrack], list[Asset | ExternalTrack], int], list[Asset | ExternalTrack]
        ],
    ) -> Playlist:
        target_count = target_count or 12
        keywords = instruction.lower().split()
        matched = [
            asset
            for asset in library
            if any(
                term in f"{asset.title} {asset.artist or ''} {' '.join(asset.genre)} {' '.join(asset.mood)}".lower()
                for term in keywords
            )
        ]
        if not matched:
            matched = sorted(library, key=lambda asset: (asset.energy_level or 0.0, asset.updated_at), reverse=True)
        allow_variants = query_requests_variant_content(instruction)
        matched = [track for track in matched if is_quality_track(track, allow_variants=allow_variants)]
        clean_candidates = [
            track for track in (candidates or []) if is_quality_track(track, allow_variants=allow_variants)
        ]
        tracks = fill_tracks(matched, clean_candidates, target_count)
        playlist = Playlist(
            playlist_id=hashlib.sha1(f"{user_id}-{instruction}".encode()).hexdigest()[:8],
            user_id=user_id,
            name=instruction or "Agent 歌单",
            description="离线回退歌单：根据你的音乐库和指令自动整理。",
            tracks=tracks[:target_count],
            generated_by="fallback",
        )
        if save_playlist is not None:
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
