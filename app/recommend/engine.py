from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from app.models import Asset, ExternalTrack, TasteProfile
from app.recommend.features import estimate_energy, estimate_tempo


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
    track_id = track.asset_id if isinstance(track, Asset) else track.external_id

    # tempo/energy：优先用已存储值（真实测量或导入/回填写入的估算值）；为 None 时按
    # genre/mood 标签即时估算（ExternalTrack 候选同样带 genre/mood，吃同一条路径）。
    # 仍为 None → 该维度无信号，下面丢弃该项并重归一化，而不是塞默认值制造常数项。
    energy = track.energy_level
    if energy is None:
        energy = estimate_energy(mood)
    tempo = track.tempo_bpm
    if tempo is None:
        tempo = estimate_tempo(genre)

    taste_genres = {g for g, _ in taste.top_genres}

    genre_match = len(set(genre) & taste_genres) / max(len(taste_genres), 1)
    mood_match = len(set(mood) & set(time_moods)) / max(len(time_moods), 1)
    novelty = 0.2 if track_id not in recent_ids else 0.0

    # 仅在维度有值时计入 energy_prox/tempo_fit；缺失则丢弃该项权重，并对剩余项重归一化到
    # 1.0（floor 防除零）。否则全库 tempo/energy 缺失时该两项对每首都是常数，稀释 genre/mood 的区分度。
    terms = weights.genre * genre_match + weights.mood * mood_match + weights.novelty * novelty
    remaining = weights.genre + weights.mood + weights.novelty
    if energy is not None:
        energy_prox = 1.0 - abs(energy - taste.preferred_energy)
        terms += weights.energy * energy_prox
        remaining += weights.energy
    if tempo is not None:
        tempo_center = sum(taste.preferred_tempo_range) / 2
        tempo_fit = math.exp(-((tempo - tempo_center) ** 2) / (2 * 30 ** 2))
        terms += weights.tempo * tempo_fit
        remaining += weights.tempo
    base = terms / max(remaining, 1e-6)

    # Spotify BaRT 思想：把真实收听行为（听完/秒跳）作为奖励信号回灌排序。
    # 行为分作为加性奖惩（不参与上方重归一化，保留 BaRT 语义），压缩到 [-1,1] 后乘权重，
    # 让被反复听完的曲目排名上升、被秒跳的下沉。
    if behavior_scores and track_id in behavior_scores:
        raw = behavior_scores[track_id]
        reward = max(-1.0, min(1.0, raw / 3.0))  # 约 3 次听完即达上限
        base += weights.behavior * reward
    return base


_ARTIST_SPLIT_RE = re.compile(r"[、,/;&]|\b(?:feat|ft|featuring|with)\b\.?", re.IGNORECASE)


def _split_artists(name: str | None) -> list[str]:
    """把多艺人字符串拆成个体（小写、去空）。

    网易云导入把多歌手拼成 'a、b、c' 一串；计数前必须拆开，否则合作歌手无法单独提出，
    画像/推荐里会看到 'kanye west、ye' 这种合并条目。分隔符：、 , / ; & 以及
    feat./ft./featuring/with（末尾可选的点会被一起消耗，避免留 '. b' 残段）。
    """
    if not name:
        return []
    return [p.strip().strip(".").strip().lower() for p in _ARTIST_SPLIT_RE.split(name) if p and p.strip()]


def _pstdev(values: list[float]) -> float:
    """总体标准差（小样本用 population std；仅用于评分中心化的 spread 估计）。"""
    n = len(values)
    if n == 0:
        return 0.0
    mean = sum(values) / n
    return (sum((v - mean) ** 2 for v in values) / n) ** 0.5


def compute_taste_profile(
    assets: list[Asset],
    listening_history: list[Any],
    ratings: list[Any] | None = None,
) -> TasteProfile:
    genre_counts: dict[str, float] = {}
    mood_counts: dict[str, float] = {}
    artist_counts: dict[str, float] = {}
    energy_sum = 0.0
    energy_n = 0
    tempo_values: list[int] = []

    # 行为信号：把 compute_behavior_scores 的收听奖惩叠加进基础权重。
    # 听完 → +1（奖励），秒跳 → -1（惩罚），中途 → 按比例。多次听指数衰减累加。
    # 这样品味随使用时间真正流动——库里没听过的歌贡献 1，听完的贡献 2，秒跳的贡献 0。
    from app.memory import compute_behavior_scores  # 延迟导入避免循环依赖
    asset_durations = {a.asset_id: a.duration_seconds for a in assets if a.duration_seconds}
    behavior = compute_behavior_scores(listening_history, asset_durations)

    # 基础权重：库中每首歌贡献 1，叠加行为分（上限 3，下限 0.1，确保不被完全消除）
    for asset in assets:
        base = 1.0 + max(-0.9, min(2.0, behavior.get(asset.asset_id, 0.0)))
        for g in asset.genre:
            genre_counts[g] = genre_counts.get(g, 0) + base
        for m in asset.mood:
            mood_counts[m] = mood_counts.get(m, 0) + base
        for _a in _split_artists(asset.artist):
            artist_counts[_a] = artist_counts.get(_a, 0) + base
        if asset.energy_level is not None:
            energy_sum += asset.energy_level * base
            energy_n += base
        if asset.tempo_bpm is not None:
            tempo_values.append(asset.tempo_bpm)

    # 评分加权：相对中心化（治选择偏差导致的评分膨胀）。
    # 用户只入库喜欢的歌 → 评分天然偏高（实测 web_user：10×123 / 8×36 / 6×2，均值 9.5）。
    # 旧公式 (score-5)*2 让所有评分都成正激励（10→+10, 8→+6, 6→+2），零梯度，推荐分不出"最喜欢什么"。
    # 改用相对自己均值的 z-score：10 分在均值 9.5 的库里只是基准（+1.6，"喜欢"已由入库 +1 表达），
    # 8/6 分才是真负信号（比我常态少）。全员同分时 std→0 → 统一归零（诚实：相同评分无区分信息）。
    # spread 设 0.5 地板，避免极度集中的评分把微小偏差放大成噪声；×3 保持与入库 base(1.0) 同量级。
    if ratings:
        asset_map = {a.asset_id: a for a in assets}
        scores = [r.score for r in ratings]
        center = sum(scores) / len(scores)
        spread = max(_pstdev(scores), 0.5)
        for rating in ratings:
            weight = ((rating.score - center) / spread) * 3.0
            genres = rating.genre or (asset_map.get(rating.asset_id, None) and asset_map[rating.asset_id].genre) or []
            moods = rating.mood or (asset_map.get(rating.asset_id, None) and asset_map[rating.asset_id].mood) or []
            for g in genres:
                genre_counts[g] = genre_counts.get(g, 0) + weight
            for m in moods:
                mood_counts[m] = mood_counts.get(m, 0) + weight
            # 高于个人常态的评分 → 抬升艺人偏好（合作歌手拆开各自加权；低于常态的不抬升）
            if weight > 0:
                for _a in _split_artists(rating.artist):
                    artist_counts[_a] = artist_counts.get(_a, 0) + weight * 0.5
            # 能量偏好：只用相对偏爱的歌（weight>0）拉 preferred_energy，避免负权重污染均值
            if weight > 0:
                rated_asset = asset_map.get(rating.asset_id)
                if rated_asset and rated_asset.energy_level is not None:
                    energy_sum += rated_asset.energy_level * weight
                    energy_n += weight

    # 过滤掉负权重
    genre_counts = {k: max(v, 0) for k, v in genre_counts.items() if v > 0}
    mood_counts = {k: max(v, 0) for k, v in mood_counts.items() if v > 0}
    artist_counts = {k: max(v, 0) for k, v in artist_counts.items() if v > 0}

    top_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:6]
    top_moods = sorted(mood_counts.items(), key=lambda x: x[1], reverse=True)[:6]
    top_artists = sorted(artist_counts.items(), key=lambda x: x[1], reverse=True)[:8]
    preferred_energy = energy_sum / energy_n if energy_n else 0.5
    if tempo_values:
        tempo_values.sort()
        n = len(tempo_values)
        p25 = tempo_values[max(0, int(n * 0.25))]
        p75 = tempo_values[min(n - 1, int(n * 0.75))]
        # p25==p75 时（库太小）展开 ±15 BPM 避免过窄
        if p25 == p75:
            p25, p75 = max(60, p25 - 15), min(220, p75 + 15)
        tempo_range = [p25, p75]
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
