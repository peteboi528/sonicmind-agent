from __future__ import annotations

import math
from typing import Any

from app.models import Asset, ExternalTrack, TasteProfile


def score_track(
    track: Asset | ExternalTrack,
    taste: TasteProfile,
    time_moods: list[str],
    recent_ids: set[str],
) -> float:
    genre = track.genre if hasattr(track, "genre") else []
    mood = track.mood if hasattr(track, "mood") else []
    energy = track.energy_level or 0.5
    tempo = track.tempo_bpm or 110
    track_id = track.asset_id if isinstance(track, Asset) else track.external_id

    taste_genres = {g for g, _ in taste.top_genres}
    taste_moods = {m for m, _ in taste.top_moods}

    genre_match = len(set(genre) & taste_genres) / max(len(taste_genres), 1)
    mood_match = len(set(mood) & set(time_moods)) / max(len(time_moods), 1)
    energy_prox = 1.0 - abs(energy - taste.preferred_energy)
    tempo_center = sum(taste.preferred_tempo_range) / 2
    tempo_fit = math.exp(-((tempo - tempo_center) ** 2) / (2 * 30 ** 2))
    novelty = 0.2 if track_id not in recent_ids else 0.0

    return (
        0.30 * genre_match
        + 0.25 * mood_match
        + 0.20 * energy_prox
        + 0.15 * tempo_fit
        + 0.10 * novelty
    )


def compute_taste_profile(
    assets: list[Asset],
    listening_history: list[Any],
    ratings: list[Any] | None = None,
) -> TasteProfile:
    genre_counts: dict[str, float] = {}
    mood_counts: dict[str, float] = {}
    energy_sum = 0.0
    energy_n = 0
    tempo_values: list[int] = []

    # 基础权重：库中每首歌贡献 1
    for asset in assets:
        for g in asset.genre:
            genre_counts[g] = genre_counts.get(g, 0) + 1
        for m in asset.mood:
            mood_counts[m] = mood_counts.get(m, 0) + 1
        if asset.energy_level is not None:
            energy_sum += asset.energy_level
            energy_n += 1
        if asset.tempo_bpm is not None:
            tempo_values.append(asset.tempo_bpm)

    # 评分加权：10.0=+4, 8.0=+2.4, 5.0=0, 3.0=-0.8, 0.0=-2
    if ratings:
        asset_map = {a.asset_id: a for a in assets}
        for rating in ratings:
            weight = (rating.score - 5.0) * 0.5  # 10→2.5, 8→1.5, 5→0, 3→-1, 0→-2.5
            genres = rating.genre or (asset_map.get(rating.asset_id, None) and asset_map[rating.asset_id].genre) or []
            moods = rating.mood or (asset_map.get(rating.asset_id, None) and asset_map[rating.asset_id].mood) or []
            for g in genres:
                genre_counts[g] = genre_counts.get(g, 0) + weight
            for m in moods:
                mood_counts[m] = mood_counts.get(m, 0) + weight
            if rating.score >= 4:
                rated_asset = asset_map.get(rating.asset_id)
                if rated_asset and rated_asset.energy_level is not None:
                    energy_sum += rated_asset.energy_level * weight
                    energy_n += abs(weight)

    # 过滤掉负权重
    genre_counts = {k: max(v, 0) for k, v in genre_counts.items() if v > 0}
    mood_counts = {k: max(v, 0) for k, v in mood_counts.items() if v > 0}

    top_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:6]
    top_moods = sorted(mood_counts.items(), key=lambda x: x[1], reverse=True)[:6]
    preferred_energy = energy_sum / energy_n if energy_n else 0.5
    if tempo_values:
        tempo_range = [min(tempo_values), max(tempo_values)]
    else:
        tempo_range = [80, 140]

    return TasteProfile(
        top_genres=[[g, round(w, 1)] for g, w in top_genres],
        top_moods=[[m, round(w, 1)] for m, w in top_moods],
        preferred_energy=round(preferred_energy, 2),
        preferred_tempo_range=tempo_range,
        discovery_openness=0.3,
    )


class RecommendEngine:
    def rank_tracks(
        self,
        candidates: list[Asset | ExternalTrack],
        taste: TasteProfile,
        time_moods: list[str],
        recent_ids: set[str],
    ) -> list[tuple[Asset | ExternalTrack, float]]:
        scored = [(t, score_track(t, taste, time_moods, recent_ids)) for t in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored
