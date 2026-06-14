from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from app.models import Asset, ExternalTrack, TasteProfile


@dataclass(frozen=True)
class ScoringWeights:
    genre: float = 0.30
    mood: float = 0.25
    energy: float = 0.20
    tempo: float = 0.15
    novelty: float = 0.10
    behavior: float = 0.25


DEFAULT_SCORING_WEIGHTS = ScoringWeights()


def score_track(
    track: Asset | ExternalTrack,
    taste: TasteProfile,
    time_moods: list[str],
    recent_ids: set[str],
    behavior_scores: dict[str, float] | None = None,
    weights: ScoringWeights = DEFAULT_SCORING_WEIGHTS,
) -> float:
    genre = track.genre if hasattr(track, "genre") else []
    mood = track.mood if hasattr(track, "mood") else []
    energy = track.energy_level or 0.5
    tempo = track.tempo_bpm or 110
    track_id = track.asset_id if isinstance(track, Asset) else track.external_id

    taste_genres = {g for g, _ in taste.top_genres}

    genre_match = len(set(genre) & taste_genres) / max(len(taste_genres), 1)
    mood_match = len(set(mood) & set(time_moods)) / max(len(time_moods), 1)
    energy_prox = 1.0 - abs(energy - taste.preferred_energy)
    tempo_center = sum(taste.preferred_tempo_range) / 2
    tempo_fit = math.exp(-((tempo - tempo_center) ** 2) / (2 * 30 ** 2))
    novelty = 0.2 if track_id not in recent_ids else 0.0

    base = (
        weights.genre * genre_match
        + weights.mood * mood_match
        + weights.energy * energy_prox
        + weights.tempo * tempo_fit
        + weights.novelty * novelty
    )

    # Spotify BaRT 思想：把真实收听行为（听完/秒跳）作为奖励信号回灌排序。
    # 行为分压缩到 [-1,1] 后乘权重，让被反复听完的曲目排名上升、被秒跳的下沉。
    if behavior_scores and track_id in behavior_scores:
        raw = behavior_scores[track_id]
        reward = max(-1.0, min(1.0, raw / 3.0))  # 约 3 次听完即达上限
        base += weights.behavior * reward
    return base


def compute_taste_profile(
    assets: list[Asset],
    listening_history: list[Any],
    ratings: list[Any] | None = None,
) -> TasteProfile:
    genre_counts: dict[str, float] = {}
    mood_counts: dict[str, float] = {}
    artist_counts: dict[str, float] = {}  # 艺术家偏好追踪
    energy_sum = 0.0
    energy_n = 0
    tempo_values: list[int] = []

    # 基础权重：库中每首歌贡献 1
    for asset in assets:
        for g in asset.genre:
            genre_counts[g] = genre_counts.get(g, 0) + 1
        for m in asset.mood:
            mood_counts[m] = mood_counts.get(m, 0) + 1
        # 艺术家计数
        if asset.artist:
            artist_key = asset.artist.lower().strip()
            artist_counts[artist_key] = artist_counts.get(artist_key, 0) + 1
        if asset.energy_level is not None:
            energy_sum += asset.energy_level
            energy_n += 1
        if asset.tempo_bpm is not None:
            tempo_values.append(asset.tempo_bpm)

    # 评分加权：大幅增强信号
    # 旧：(score - 5.0) * 0.5 → 10分才+2.5，太弱
    # 新：(score - 5.0) * 2.0 → 10分+10, 8分+6, 5分0, 3分-4, 1分-8
    # 评分比单纯的库存在权重高得多，真正反映用户喜好
    if ratings:
        asset_map = {a.asset_id: a for a in assets}
        for rating in ratings:
            weight = (rating.score - 5.0) * 2.0  # 10→10, 8→6, 5→0, 3→-4, 1→-8
            genres = rating.genre or (asset_map.get(rating.asset_id, None) and asset_map[rating.asset_id].genre) or []
            moods = rating.mood or (asset_map.get(rating.asset_id, None) and asset_map[rating.asset_id].mood) or []
            for g in genres:
                genre_counts[g] = genre_counts.get(g, 0) + weight
            for m in moods:
                mood_counts[m] = mood_counts.get(m, 0) + weight
            # 高分艺术家偏好加强
            if rating.score >= 7:
                artist_name = rating.artist or ""
                if artist_name:
                    artist_key = artist_name.lower().strip()
                    artist_counts[artist_key] = artist_counts.get(artist_key, 0) + weight * 0.5
            # 能量偏好
            if rating.score >= 4:
                rated_asset = asset_map.get(rating.asset_id)
                if rated_asset and rated_asset.energy_level is not None:
                    energy_sum += rated_asset.energy_level * weight
                    energy_n += abs(weight)

    # 过滤掉负权重
    genre_counts = {k: max(v, 0) for k, v in genre_counts.items() if v > 0}
    mood_counts = {k: max(v, 0) for k, v in mood_counts.items() if v > 0}
    artist_counts = {k: max(v, 0) for k, v in artist_counts.items() if v > 0}

    top_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:6]
    top_moods = sorted(mood_counts.items(), key=lambda x: x[1], reverse=True)[:6]
    top_artists = sorted(artist_counts.items(), key=lambda x: x[1], reverse=True)[:8]
    preferred_energy = energy_sum / energy_n if energy_n else 0.5
    if tempo_values:
        tempo_range = [min(tempo_values), max(tempo_values)]
    else:
        tempo_range = [80, 140]

    openness = _adapt_discovery_openness(assets, listening_history, top_genres)

    return TasteProfile(
        top_genres=[(g, round(w, 1)) for g, w in top_genres],
        top_moods=[(m, round(w, 1)) for m, w in top_moods],
        top_artists=[(a, round(w, 1)) for a, w in top_artists],
        preferred_energy=round(preferred_energy, 2),
        preferred_tempo_range=tempo_range,
        discovery_openness=openness,
    )


def _adapt_discovery_openness(
    assets: list[Asset],
    listening_history: list[Any],
    top_genres: list[tuple[str, float]],
) -> float:
    """根据用户对"探索性收听"的真实反馈动态调整 discovery_openness（BaRT 奖励闭环）。

    每个 genre 的"参与度" = 库存广度（每首贡献 1）+ 该曲目的真实行为分（听完/秒跳）。
    参与度高于均值的 genre 视为已确立偏好；被收听但参与度低于均值的曲目视为探索项。
    探索项多被听完 → 用户乐于探索，调高 openness；多被秒跳 → 调低。
    无足够信号时回退默认 0.3，clamp 到 [0.1, 0.6]。
    """
    from app.memory import compute_behavior_scores

    default = 0.3
    if not assets or not listening_history:
        return default

    durations = {a.asset_id: a.duration_seconds for a in assets}
    behavior = compute_behavior_scores(listening_history, durations)
    if not behavior:
        return default

    asset_map = {a.asset_id: a for a in assets}

    # genre 参与度 = 库存广度 + 真实收听奖励
    genre_engagement: dict[str, float] = {}
    for asset in assets:
        for g in asset.genre:
            genre_engagement[g] = genre_engagement.get(g, 0.0) + 1.0 + behavior.get(asset.asset_id, 0.0)
    if len(genre_engagement) < 2:
        return default  # 至少要有两个 genre 才能区分主流 vs 探索

    mean_eng = sum(genre_engagement.values()) / len(genre_engagement)
    dominant = {g for g, e in genre_engagement.items() if e >= mean_eng}

    # 探索性收听 = 被收听、但 genre 全部落在主流之外的曲目
    explore_reward = 0.0
    explore_n = 0
    for asset_id, score in behavior.items():
        asset = asset_map.get(asset_id)
        if asset is None or not asset.genre:
            continue
        if not (set(asset.genre) & dominant):
            explore_reward += score
            explore_n += 1

    if explore_n == 0:
        return default
    # 归一化：约 3 次听完/秒跳达到满信号
    avg = max(-1.0, min(1.0, (explore_reward / explore_n) / 3.0))
    adjusted = default + (0.3 * avg if avg > 0 else 0.2 * avg)
    return round(max(0.1, min(0.6, adjusted)), 2)


class RecommendEngine:
    def rank_tracks(
        self,
        candidates: list[Asset | ExternalTrack],
        taste: TasteProfile,
        time_moods: list[str],
        recent_ids: set[str],
        behavior_scores: dict[str, float] | None = None,
    ) -> list[tuple[Asset | ExternalTrack, float]]:
        scored = [
            (t, score_track(t, taste, time_moods, recent_ids, behavior_scores))
            for t in candidates
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored
