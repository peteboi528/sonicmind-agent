"""把证据组装成结构化、可解释的画像（计划 §4-8, §17, §18）。

设计铁律（延续项目防幻觉传统）：
- 只承载可追溯证据。证据不足的维度返回空 + 低置信，绝不编造分数或艺人关系。
- 纯确定性映射：genre/mood → 声音维度 / 情绪坐标都是固定查表，可单测、可解释。
- 每个分数都能用证据回答「为什么是这个值」，对应前端「为什么在这里？」。
"""

from __future__ import annotations

from datetime import UTC, datetime

from app.models import UserMemory
from app.profile.evidence import (
    SCENE_LABELS,
    ProfileEvidence,
    collect_profile_evidence,
)
from app.profile.models import (
    ArtistRelation,
    DiscoveryStyle,
    MoodLandscape,
    MoodPoint,
    ScenePreference,
    SoundDimension,
    SoundFingerprint,
    TasteSummary,
    UserProfileResponse,
)

# ── 声音指纹维度定义（计划 §4.1）────────────────────────────────────────────
# 每维 = 若干 genre/mood 的加权贡献。归一化后映射到 [0,1]。
_SOUND_DIMENSIONS: list[tuple[str, str]] = [
    ("groove", "律动感"),
    ("melody", "旋律性"),
    ("energy", "能量"),
    ("softness", "柔和度"),
    ("electronic", "电子感"),
    ("vocal_intimacy", "人声亲密度"),
    ("experimental", "实验性"),
    ("acoustic", "原声质感"),
]

# genre → {维度: 贡献}。贡献是该 genre 对维度的「典型强度」。
_GENRE_TO_SOUND: dict[str, dict[str, float]] = {
    "R&B": {"groove": 0.9, "melody": 0.7, "vocal_intimacy": 0.8, "softness": 0.5},
    "说唱": {"groove": 0.95, "energy": 0.6, "vocal_intimacy": 0.5, "experimental": 0.3},
    "电子": {"electronic": 0.95, "groove": 0.7, "energy": 0.7, "experimental": 0.4},
    "流行": {"melody": 0.9, "vocal_intimacy": 0.5, "energy": 0.5},
    "摇滚": {"energy": 0.85, "experimental": 0.4, "melody": 0.5},
    "金属": {"energy": 0.95, "experimental": 0.6},
    "民谣": {"acoustic": 0.95, "melody": 0.8, "softness": 0.8, "vocal_intimacy": 0.7},
    "古典": {"acoustic": 0.9, "melody": 0.85, "softness": 0.7, "experimental": 0.3},
    "爵士": {"acoustic": 0.7, "groove": 0.6, "experimental": 0.6, "softness": 0.5},
    "国风": {"melody": 0.8, "acoustic": 0.6, "experimental": 0.3},
}

# mood → {维度: 贡献}。
_MOOD_TO_SOUND: dict[str, dict[str, float]] = {
    "律动": {"groove": 0.9, "energy": 0.5},
    "激昂": {"energy": 0.95},
    "欢快": {"energy": 0.6, "melody": 0.5},
    "放松": {"softness": 0.9, "energy": 0.0},
    "治愈": {"softness": 0.8, "vocal_intimacy": 0.6, "melody": 0.5},
    "宁静": {"softness": 0.95},
    "梦幻": {"electronic": 0.6, "experimental": 0.6, "softness": 0.5},
    "暗黑": {"experimental": 0.7, "electronic": 0.4},
    "性感": {"vocal_intimacy": 0.8, "groove": 0.5, "softness": 0.4},
    "浪漫": {"vocal_intimacy": 0.7, "melody": 0.6, "softness": 0.5},
    "伤感": {"vocal_intimacy": 0.6, "softness": 0.6, "melody": 0.5},
    "励志": {"energy": 0.7, "melody": 0.5},
}

# mood → 情绪地图坐标 valence(明亮↔阴郁) × arousal(平静↔激昂)，计划 §5.2。
_MOOD_COORDS: dict[str, tuple[float, float]] = {
    "放松": (0.3, -0.5),
    "治愈": (0.7, -0.2),
    "欢快": (0.8, 0.55),
    "伤感": (-0.7, -0.3),
    "浪漫": (0.5, 0.05),
    "激昂": (0.4, 0.9),
    "宁静": (0.25, -0.8),
    "梦幻": (0.3, -0.25),
    "律动": (0.5, 0.5),
    "暗黑": (-0.6, 0.2),
    "性感": (0.25, 0.15),
    "励志": (0.7, 0.7),
}

_SCENE_STRATEGY: dict[str, str] = {
    "运动": "优先旋律清晰、节奏明确、情绪积极的歌曲；不必一味追高 BPM 或 DJ 串烧。",
    "学习": "降低人声与歌词密度、减少情绪波动，偏稳定节奏与氛围声。",
    "睡眠": "选低刺激、柔和、节奏舒缓的曲目，避免强鼓点与突兀动态。",
    "开车": "中等能量、律动稳定、适合长时间播放的曲目。",
    "通勤": "节奏明快但不过度刺激，适合碎片时间。",
    "派对": "高能量、律动强、节奏带动气氛的曲目。",
    "咖啡": "轻盈、温暖、旋律性强的背景型曲目。",
}


def _normalize(weights: dict[str, float]) -> dict[str, float]:
    """把权重字典归一化到 [0,1]（按最大值）。空则返回空。"""
    if not weights:
        return {}
    peak = max(weights.values())
    if peak <= 0:
        return {}
    return {k: max(0.0, v) / peak for k, v in weights.items()}


def compute_confidence(
    *,
    evidence_count: int,
    explicit: bool,
    recency: float,
    consistency: float,
    contradiction: float,
) -> float:
    """置信度计算（计划 §18）。各子项 ∈ [0,1]，contradiction 为惩罚。"""
    evidence_score = min(1.0, evidence_count / 4.0)  # 约 4 条证据达满
    explicit_score = 1.0 if explicit else 0.4
    value = (
        0.2 * evidence_score
        + 0.3 * explicit_score
        + 0.2 * max(0.0, min(1.0, recency))
        + 0.2 * max(0.0, min(1.0, consistency))
        - 0.2 * max(0.0, min(1.0, contradiction))
    )
    return round(max(0.0, min(1.0, value)), 2)


def confidence_band(value: float) -> str:
    if value >= 0.66:
        return "high"
    if value >= 0.4:
        return "medium"
    return "low"


# 时效衰减窗口：最近一条信号距今 90 天内线性降到 0。品味是粘性的，窗口给宽一点。
RECENCY_DECAY_DAYS = 90


def _parse_iso_epoch(ts: str) -> float:
    """把 ISO 时间字符串解析成 epoch 秒；失败/空返回 0。"""
    if not ts:
        return 0.0
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
    except (ValueError, TypeError):
        return 0.0


def _compute_recency(ev: ProfileEvidence) -> float:
    """最近一条「品味信号」的新鲜度 ∈ [0,1]：90 天内线性衰减到 0。

    只看真正的品味信号时间戳：listening / ratings / structured_preferences。
    故意不用 memory.updated_at——它在任何写入（巩固画像、情景记忆等）都会刷新，
    会把陈旧品味误判成新鲜。仅有 taste_profile（无行为/明确偏好时间戳）的用户无
    候选时间戳 → 返回 0.5 中性，不因「未知」而惩罚。
    """
    candidates = [_parse_iso_epoch(e.timestamp) for e in ev.listening]
    candidates += [_parse_iso_epoch(r.timestamp) for r in ev.ratings]
    candidates += [_parse_iso_epoch(entry.last_used) for entry, _ in ev.scored_preferences]
    latest = max(candidates) if candidates else 0.0
    if latest <= 0:
        return 0.5  # 无时间戳信号 → 中性，不惩罚
    age_days = (datetime.now(UTC).timestamp() - latest) / 86400.0
    return round(max(0.0, min(1.0, 1.0 - age_days / RECENCY_DECAY_DAYS)), 2)


def _build_sound_fingerprint(ev: ProfileEvidence) -> SoundFingerprint:
    g_norm = _normalize(ev.genre_weights)
    m_norm = _normalize(ev.mood_weights)
    if not g_norm and not m_norm and not ev.taste.top_genres:
        return SoundFingerprint()

    raw: dict[str, float] = {key: 0.0 for key, _ in _SOUND_DIMENSIONS}
    contributors: dict[str, list[str]] = {key: [] for key, _ in _SOUND_DIMENSIONS}

    for genre, w in g_norm.items():
        for dim, strength in _GENRE_TO_SOUND.get(genre, {}).items():
            raw[dim] += w * strength
            if w * strength >= 0.25:
                contributors[dim].append(genre)
    for mood, w in m_norm.items():
        for dim, strength in _MOOD_TO_SOUND.get(mood, {}).items():
            raw[dim] += w * strength
            if w * strength >= 0.25:
                contributors[dim].append(mood)

    # energy 维度同时受 taste.preferred_energy 直接驱动。
    raw["energy"] += ev.taste.preferred_energy
    # softness 与 energy 互补：高能量的用户柔和度相应降低。
    if ev.taste.preferred_energy:
        raw["softness"] += (1.0 - ev.taste.preferred_energy) * 0.4

    norm = _normalize(raw)
    dims: list[SoundDimension] = []
    for key, label in _SOUND_DIMENSIONS:
        value = round(norm.get(key, 0.0), 2)
        names = list(dict.fromkeys(contributors[key]))[:3]
        if names:
            explanation = f"主要来自你偏好的 {('、'.join(names))}。"
        elif key == "energy" and ev.taste.preferred_energy:
            explanation = "依据你收听/评分曲目的整体能量水平估算。"
        else:
            explanation = "证据较少，仅作初步估计。"
        dims.append(SoundDimension(key=key, label=label, value=value, explanation=explanation))

    top = sorted(dims, key=lambda d: d.value, reverse=True)[:3]
    top_labels = "、".join(d.label for d in top if d.value > 0)
    summary = (
        f"你的声音指纹偏向 {top_labels}：更喜欢有结构、有质感的声音，而非单一维度的极端。"
        if top_labels
        else "声音指纹还在形成中。"
    )
    return SoundFingerprint(dimensions=dims, explanation=summary)


def _build_mood_landscape(ev: ProfileEvidence) -> MoodLandscape:
    m_norm = _normalize(ev.mood_weights)
    points: list[MoodPoint] = []
    for mood, weight in sorted(m_norm.items(), key=lambda x: x[1], reverse=True):
        coord = _MOOD_COORDS.get(mood)
        if not coord:
            continue
        valence, arousal = coord
        points.append(
            MoodPoint(
                mood=mood,
                valence=valence,
                arousal=arousal,
                weight=round(weight, 2),
                evidence_count=int(ev.mood_weights.get(mood, 0) > 0),
            )
        )
    if not points:
        return MoodLandscape(summary="还没有足够的情绪信号来描绘你的情绪地图。")

    # 加权重心 → 一句话描述常驻区域。
    tot = sum(p.weight for p in points) or 1.0
    cx = sum(p.valence * p.weight for p in points) / tot
    cy = sum(p.arousal * p.weight for p in points) / tot
    bright = "明亮" if cx >= 0.15 else "阴郁" if cx <= -0.15 else "中性"
    energy = "激昂" if cy >= 0.25 else "平静" if cy <= -0.25 else "中等能量"
    summary = (
        f"你最常停留在「{bright} + {energy}」的情绪区域，"
        f"说明你偏好{'积极、有节奏感' if cx >= 0 else '内省、克制'}的音乐，"
        "而非极端兴奋或极端低落。"
    )
    return MoodLandscape(global_points=points, summary=summary)


def _build_scenes(ev: ProfileEvidence) -> list[ScenePreference]:
    if not ev.scene_hits:
        return []
    top_genres = [g for g, _ in sorted(ev.genre_weights.items(), key=lambda x: x[1], reverse=True)][:4]
    top_moods = [m for m, _ in sorted(ev.mood_weights.items(), key=lambda x: x[1], reverse=True)][:3]
    scenes: list[ScenePreference] = []
    for scene, hits in sorted(ev.scene_hits.items(), key=lambda x: x[1], reverse=True):
        if hits <= 0:
            continue
        conf = compute_confidence(
            evidence_count=hits,
            explicit=hits >= 2,
            recency=0.6,
            consistency=0.6,
            contradiction=0.0,
        )
        avoid = list(ev.exclusions)[:4]
        scenes.append(
            ScenePreference(
                scene=scene,
                label=SCENE_LABELS.get(scene, scene),
                preferred_genres=top_genres,
                preferred_moods=top_moods,
                avoid_features=avoid,
                recommendation_strategy=_SCENE_STRATEGY.get(scene, "按你整体品味挑选合适曲目。"),
                confidence=conf,
                examples=[],
            )
        )
    return scenes


def _build_artists(ev: ProfileEvidence) -> list[ArtistRelation]:
    a_norm = _normalize(ev.artist_weights)
    if not a_norm and not ev.dislikes:
        return []
    ordered = sorted(a_norm.items(), key=lambda x: x[1], reverse=True)
    relations: list[ArtistRelation] = []
    dislike_keys = {d.strip().lower() for d in ev.dislikes}
    for idx, (artist, weight) in enumerate(ordered):
        if artist in dislike_keys:
            relation = "avoid"
            reasons = ["你曾对这位艺人/相关曲目表达过不感兴趣。"]
        elif idx < 3 and weight >= 0.6:
            relation = "core"
            reasons = ["在你高分/高频收听的曲目中反复出现。"]
        elif weight >= 0.35:
            relation = "occasional"
            reasons = ["在你的收听记录里出现过几次。"]
        else:
            relation = "occasional"
            reasons = ["弱信号，出现次数较少。"]
        conf = compute_confidence(
            evidence_count=int(weight * 4),
            explicit=False,
            recency=0.5,
            consistency=weight,
            contradiction=0.0,
        )
        relations.append(
            ArtistRelation(
                artist=artist,
                relation_type=relation,
                reasons=reasons,
                evidence_tracks=[],
                confidence=conf,
            )
        )
    # 明确不喜欢但未出现在权重表里的艺人，单列为 avoid。
    seen = {r.artist for r in relations}
    for dislike in ev.dislikes:
        key = dislike.strip().lower()
        if key and key not in seen:
            relations.append(
                ArtistRelation(
                    artist=dislike,
                    relation_type="avoid",
                    reasons=["你明确表达过不想要这类。"],
                    confidence=0.7,
                )
            )
            seen.add(key)
    return relations[:12]


def _build_discovery_style(ev: ProfileEvidence) -> DiscoveryStyle:
    openness = float(ev.taste.discovery_openness or 0.3)
    # 语言开放度：中英文分布 + 排除项。"不要中文/英文" 强烈拉低对应开放度。
    zh = ev.language_counts.get("zh", 0)
    en = ev.language_counts.get("en", 0)
    total_lang = zh + en
    lang_balance = 1.0 - abs(zh - en) / total_lang if total_lang else 0.5
    excl_text = " ".join(ev.exclusions).lower()
    if "中文" in excl_text or "国语" in excl_text or "华语" in excl_text:
        lang_balance = min(lang_balance, 0.35)
    novelty = round(min(1.0, openness / 0.6), 2)  # openness 上限 0.6 → 满
    mainstream = round(1.0 - novelty * 0.7, 2)
    niche = round(novelty * 0.8, 2)
    if novelty >= 0.66:
        label = "探索型"
        explanation = "你愿意尝试陌生风格和冷门艺人，对新声音接受度高。"
    elif novelty >= 0.4:
        label = "平衡探索型"
        explanation = "你通常喜欢熟悉的声音，但在情绪相近、节奏舒适时也愿意接受新艺人。"
    else:
        label = "保守型"
        explanation = "你更偏好熟悉的艺人和风格，新东西需要更贴近你的口味才会买账。"
    return DiscoveryStyle(
        label=label,
        novelty_tolerance=novelty,
        mainstream_preference=mainstream,
        niche_openness=niche,
        language_openness=round(lang_balance, 2),
        explanation=explanation,
    )


def _build_summary(
    ev: ProfileEvidence,
    sound: SoundFingerprint,
    mood: MoodLandscape,
    discovery: DiscoveryStyle,
    overall_conf: float,
) -> TasteSummary:
    top_genres = [g for g, _ in sorted(ev.genre_weights.items(), key=lambda x: x[1], reverse=True)][:3]
    top_sound = [d.label for d in sorted(sound.dimensions, key=lambda d: d.value, reverse=True)[:2] if d.value > 0]
    chips: list[str] = []
    chips.extend(top_sound)
    if top_genres:
        chips.append("/".join(top_genres[:2]))
    chips.append(discovery.label)
    chips = [c for c in dict.fromkeys(chips) if c][:4]

    if ev.consolidated:
        headline = ev.consolidated
    elif top_genres:
        headline = (
            f"你最近偏向 {('、'.join(top_genres))} 的声音，"
            f"{'有律动但不压迫、旋律清晰' if top_sound else '风格逐渐清晰'}。"
        )
    else:
        headline = "你的音乐品味正在形成，多给一些反馈我会更懂你。"

    core: list[str] = []
    if top_sound:
        core.append(f"声音上偏好{('、'.join(top_sound))}")
    top_moods = [p.mood for p in mood.global_points[:2]]
    if top_moods:
        core.append(f"情绪上常出现{('、'.join(top_moods))}")
    core.append(f"探索风格为{discovery.label}")

    hint = "推荐时不应只追求 BPM 或能量值，更应兼顾旋律性与情绪舒适度。"
    if ev.exclusions:
        hint += f" 同时严格避开你排除的：{('、'.join(ev.exclusions[:3]))}。"
    return TasteSummary(
        headline=headline,
        core_preferences=core,
        recommendation_hint=hint,
        chips=chips,
        confidence=overall_conf,
    )


def _empty_profile(user_id: str) -> UserProfileResponse:
    return UserProfileResponse(
        user_id=user_id,
        is_empty=True,
        empty_hint=(
            "SonicMind 还在认识你。你可以导入歌单、对推荐点喜欢/不喜欢、"
            "直接告诉我你想听什么，或让我生成几次不同场景的歌单来加速建立画像。"
        ),
        summary=TasteSummary(
            headline="SonicMind 还在认识你。",
            core_preferences=[],
            recommendation_hint="多给一些反馈，我会逐渐理解你的音乐品味。",
        ),
    )


class UserProfileBuilder:
    """从一份 UserMemory 构建完整画像（计划 §16.1）。

    无状态、可复用：调用方负责读 memory（API 层已 get_memory 过），builder 不读盘。
    """

    def build_from_memory(self, memory: UserMemory) -> UserProfileResponse:
        ev = collect_profile_evidence(memory)
        if not ev.has_any_signal:
            return _empty_profile(memory.user_id)

        sound = _build_sound_fingerprint(ev)
        mood = _build_mood_landscape(ev)
        scenes = _build_scenes(ev)
        artists = _build_artists(ev)
        discovery = _build_discovery_style(ev)

        # 整体证据充分度 → 顶层置信。
        evidence_strength = round(min(1.0, ev.total_signal_count / 8.0), 2)
        overall_conf = compute_confidence(
            evidence_count=ev.total_signal_count,
            explicit=ev.explicit_signal_count > 0,
            recency=_compute_recency(ev),
            # consistency / contradiction 暂为占位：真实跨信号一致性/矛盾（如 dislikes、
            # exclusions 与 top 风格冲突）需单独逻辑，属后续优化，不在本次范围。
            consistency=0.6,
            contradiction=0.0,
        )
        summary = _build_summary(ev, sound, mood, discovery, overall_conf)

        return UserProfileResponse(
            user_id=memory.user_id,
            is_empty=False,
            summary=summary,
            sound_fingerprint=sound,
            mood_landscape=mood,
            scenes=scenes,
            artists=artists,
            discovery_style=discovery,
            insights=[],  # 由 insights.generate_profile_insights 填充
            hard_constraints=list(ev.exclusions),
            evidence_strength=evidence_strength,
        )
