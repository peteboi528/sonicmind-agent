from __future__ import annotations

import math
from typing import Any

from app.models import Asset, ExternalTrack, utc_now_iso


class JourneyService:
    def __init__(
        self,
        *,
        store: Any,
        memory: Any,
        library: Any,
        list_assets: Any,
        search_web_music: Any,
        rerank_tracks: Any,
        track_key: Any,
        dedupe_tracks: Any,
        is_recommendation_quality_track: Any,
    ) -> None:
        self.store = store
        self.memory = memory
        self.library = library
        self._list_assets = list_assets
        self._search_web_music = search_web_music
        self._rerank_tracks = rerank_tracks
        self._track_key = track_key
        self._dedupe_tracks = dedupe_tracks
        self._is_recommendation_quality_track = is_recommendation_quality_track

    def generate_music_journey(
        self,
        user_id: str,
        instruction: str,
        target_count: int | None = None,
        *,
        journey_phases: Any,
        record_journey_history: Any,
    ) -> dict[str, Any]:
        from app.concurrency import run_parallel
        from app.search.netease_playlist import search_and_extract

        memory = self.memory.get_memory(user_id)
        phases = journey_phases(instruction, memory.taste_profile)
        out = {"user_id": user_id, "instruction": instruction, "phases": []}
        per_phase = 4
        if target_count:
            per_phase = max(1, math.ceil(target_count / max(1, len(phases))))
        recent = set(memory.journey_history[-160:])
        rotation = len(memory.journey_history) // max(1, per_phase)

        def recall_phase(phase: dict[str, Any]) -> list[ExternalTrack]:
            tracks = search_and_extract(
                phase["queries"][rotation % len(phase["queries"])],
                max_playlists=4,
                tracks_per_playlist=per_phase * 4,
            )
            if tracks:
                return tracks
            fallback: list[ExternalTrack] = []
            for query in phase["queries"]:
                batch = self._search_web_music(
                    query,
                    top_k=per_phase * 4,
                    relevance_query=phase["query"],
                )
                fallback.extend(batch)
                if len(self._dedupe_tracks(fallback)) >= per_phase * 2:
                    break
            return self._dedupe_tracks(fallback)

        tasks = [
            (
                f"journey:{phase['name']}",
                lambda phase=phase: recall_phase(phase),
            )
            for phase in phases
        ]
        batches = run_parallel(tasks, timeout=15.0, default=[])
        seen: set[str] = set()
        local_fallback = [
            track
            for track in self._list_assets()
            if self._is_recommendation_quality_track(track) and not self.library.is_disliked(user_id, track)
        ]
        journey_tracks: list[Asset | ExternalTrack] = []

        for phase, batch in zip(phases, batches, strict=False):
            pool: list[Asset | ExternalTrack] = []
            for track in self._dedupe_tracks([*(batch or []), *local_fallback]):
                key = self._track_key(track)
                if key in seen or key in recent or self.library.is_disliked(user_id, track):
                    continue
                if not self._is_recommendation_quality_track(track):
                    continue
                pool.append(track)
            ranked = self._rerank_tracks(
                user_id,
                phase["query"],
                pool,
                top_k=max(per_phase * 2, per_phase),
            )
            candidates = [track for track, _ in ranked[:per_phase]]
            if len(candidates) < per_phase:
                refill_pool = [
                    track
                    for track in self._dedupe_tracks([*(batch or []), *local_fallback])
                    if self._track_key(track) not in seen
                    and self._is_recommendation_quality_track(track)
                    and not self.library.is_disliked(user_id, track)
                ]
                refill = self._rerank_tracks(user_id, phase["query"], refill_pool, top_k=per_phase)
                for track, _ in refill:
                    if self._track_key(track) not in {self._track_key(item) for item in candidates}:
                        candidates.append(track)
                    if len(candidates) >= per_phase:
                        break
            for track in candidates:
                seen.add(self._track_key(track))
            journey_tracks.extend(candidates)
            self.library.record_exposure(candidates)
            out["phases"].append(
                {
                    "name": phase["name"],
                    "goal": phase["goal"],
                    "transition": phase["transition"],
                    "energy": phase["energy"],
                    "tracks": [self._serialize_journey_track(track) for track in candidates],
                }
            )
        record_journey_history(user_id, journey_tracks)
        return out

    def record_journey_history(self, user_id: str, tracks: list[Asset | ExternalTrack]) -> None:
        keys = [self._track_key(track) for track in tracks]
        if not keys:
            return
        with self.store.lock("memory", user_id):
            memory = self.memory.get_memory(user_id)
            memory.journey_history = [*memory.journey_history, *keys][-240:]
            memory.updated_at = utc_now_iso()
            self.store.write_model("memory", user_id, memory)

    @staticmethod
    def _serialize_journey_track(track: Asset | ExternalTrack) -> dict[str, Any]:
        if isinstance(track, ExternalTrack):
            return track.model_dump(mode="json")
        return ExternalTrack(
            external_id=track.external_id or track.asset_id,
            title=track.title,
            artist=track.artist or "未知",
            album=track.album,
            genre=track.genre,
            mood=track.mood,
            tempo_bpm=track.tempo_bpm,
            energy_level=track.energy_level,
            cover_url=track.cover_url,
            playback_url=track.source_url,
            source="local",
        ).model_dump(mode="json")
