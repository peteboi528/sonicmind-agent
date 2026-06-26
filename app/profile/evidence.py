"""画像证据收集（计划 §10, §17）。

纯确定性、零 LLM、零网络：从 UserMemory / listening_history / ratings /
taste_profile 把信号收成一个 ProfileEvidence，供 builder 组装。

权重原则（计划 §10.2）：明确拒绝 > 明确喜欢 > 多次请求 > 播放/收藏 >
单次聊天提及 > 模型推断。这里在收集阶段就给不同来源不同权重，下游据此算置信度。

刻意不调 LLM：画像页是高频读接口，per-request 不能随库/记忆增长引入昂贵开销
（见记忆「稳定性天花板」）。所有归一化/映射都是 O(信号数) 的查表与计数。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.graph.tag_rules import extract_genre, extract_mood, extract_scenario
from app.memory import score_entries
from app.models import ListeningEvent, RatingEntry, TasteProfile, UserMemory

_CJK_RE = re.compile(r"[一-鿿]")
_LATIN_RE = re.compile(r"[A-Za-z]")

# 来源可信度权重（计划 §10.2）。明确表达 > 检索/收藏 > 模型推断。
_SOURCE_WEIGHT: dict[str, float] = {
    "user_event": 1.0,
    "auto_explicit": 1.0,
    "migrated": 0.8,
    "from_search_result": 0.5,
    "llm_extract": 0.4,
    "inferred_from_result": 0.3,
}

# 场景中文标签（与 tag_rules._SCENARIO_RULES 的 key 对齐）。
SCENE_LABELS: dict[str, str] = {
    "运动": "运动 / 跑步",
    "学习": "学习 / 专注",
    "睡眠": "睡前 / 助眠",
    "开车": "开车 / 兜风",
    "通勤": "通勤 / 路上",
    "派对": "派对 / 聚会",
    "咖啡": "咖啡 / 下午茶",
}


@dataclass
class ProfileEvidence:
    """画像的全部原始证据，已做来源加权与去重，但未组装成可读维度。"""

    user_id: str
    # 已按 frequency × 时间衰减打分并排序的结构化偏好 (text, weight)。
    scored_preferences: list[tuple[str, float]] = field(default_factory=list)
    raw_preferences: list[str] = field(default_factory=list)
    dislikes: list[str] = field(default_factory=list)
    exclusions: list[str] = field(default_factory=list)
    ratings: list[RatingEntry] = field(default_factory=list)
    listening: list[ListeningEvent] = field(default_factory=list)
    taste: TasteProfile = field(default_factory=TasteProfile)
    consolidated: str = ""
    episodes: list[str] = field(default_factory=list)
    # 聚合后的加权信号。
    genre_weights: dict[str, float] = field(default_factory=dict)
    mood_weights: dict[str, float] = field(default_factory=dict)
    artist_weights: dict[str, float] = field(default_factory=dict)
    scene_hits: dict[str, int] = field(default_factory=dict)
    language_counts: dict[str, int] = field(default_factory=dict)
    explicit_signal_count: int = 0  # 明确表达过的偏好/排除条数
    behavior_signal_count: int = 0  # 评分 + 收听条数

    @property
    def total_signal_count(self) -> int:
        # taste_profile 是「库 + 评分」加权聚合的稳定品味信号：导入歌单/收听都会落到
        # 这里。把它计入证据，否则只有 taste_profile（无评分/排除/明确偏好）的用户
        # 会被判成「零证据」、置信度卡在地板（见 builder.compute_confidence）。
        # 与 ratings 存在不同粒度的轻度重合计（ratings 会喂进 taste_profile），
        # 可接受：下游 evidence_score/evidence_strength 都有自然封顶防膨胀。
        taste = (
            len(self.taste.top_genres)
            + len(self.taste.top_artists)
            + len(self.taste.top_moods)
        )
        return (
            len(self.scored_preferences)
            + len(self.ratings)
            + len(self.listening)
            + len(self.exclusions)
            + len(self.dislikes)
            + taste
        )

    @property
    def has_any_signal(self) -> bool:
        return bool(
            self.scored_preferences
            or self.raw_preferences
            or self.ratings
            or self.listening
            or self.exclusions
            or self.dislikes
            or self.taste.top_genres
            or self.taste.top_artists
            or self.consolidated
        )


def detect_language(text: str) -> str:
    """粗判文本主语言：zh / en / other（用于语言开放度与加权）。"""
    if not text:
        return "other"
    cjk = len(_CJK_RE.findall(text))
    latin = len(_LATIN_RE.findall(text))
    if cjk and cjk >= latin:
        return "zh"
    if latin and latin > cjk:
        return "en"
    return "other"


def _bump(d: dict[str, float], key: str, amount: float) -> None:
    if key and amount:
        d[key] = d.get(key, 0.0) + amount


def collect_profile_evidence(memory: UserMemory) -> ProfileEvidence:
    """从一份 UserMemory 收集画像证据（不读盘，调用方传入 memory）。"""
    ev = ProfileEvidence(user_id=memory.user_id)
    ev.raw_preferences = list(memory.preferences)
    ev.dislikes = list(memory.dislikes)
    ev.exclusions = list(memory.exclusion_rules)
    ev.ratings = list(memory.ratings)
    ev.listening = list(memory.listening_history)
    ev.taste = memory.taste_profile or TasteProfile()
    ev.consolidated = memory.consolidated_profile or ""
    ev.episodes = [e.text for e in memory.episodic_memory][-12:]
    ev.scored_preferences = score_entries(memory.structured_preferences)

    # 1) taste_profile 已是「库 + 评分」加权后的稳定信号，作为基底。
    for genre, weight in ev.taste.top_genres:
        _bump(ev.genre_weights, genre, float(weight))
    for mood, weight in ev.taste.top_moods:
        _bump(ev.mood_weights, mood, float(weight))
    for artist, weight in ev.taste.top_artists:
        _bump(ev.artist_weights, artist, float(weight))

    # 2) 结构化偏好：按来源可信度 × 打分加权，抽 genre/mood/scene。
    for entry, weight in ev.scored_preferences:
        source = getattr(entry, "source", "") or ""
        src_w = _SOURCE_WEIGHT.get(source, 0.5)
        text = entry.text or ""
        if src_w >= 1.0:
            ev.explicit_signal_count += 1
        signal = weight * src_w
        for g in extract_genre(text):
            _bump(ev.genre_weights, g, signal)
        for m in extract_mood(text):
            _bump(ev.mood_weights, m, signal)
        for scene in extract_scenario(text):
            ev.scene_hits[scene] = ev.scene_hits.get(scene, 0) + 1
        lang = detect_language(text)
        ev.language_counts[lang] = ev.language_counts.get(lang, 0) + 1

    # 3) 评分：高分曲目的 genre/mood/artist 加权（评分是强信号）。
    for rating in ev.ratings:
        ev.behavior_signal_count += 1
        delta = (rating.score - 5.0) / 5.0  # [-1, 1]
        if delta <= 0:
            continue
        for g in (rating.genre or []):
            _bump(ev.genre_weights, g, delta)
        for m in (rating.mood or []):
            _bump(ev.mood_weights, m, delta)
        if rating.artist and rating.score >= 7:
            _bump(ev.artist_weights, rating.artist.strip().lower(), delta)
        if rating.title or rating.artist:
            lang = detect_language(f"{rating.title} {rating.artist}")
            ev.language_counts[lang] = ev.language_counts.get(lang, 0) + 1

    # 4) 收听上下文 / 情景记忆 / 目标：抽场景命中（per-scene 证据）。
    ev.behavior_signal_count += len(ev.listening)
    scene_texts: list[str] = []
    scene_texts.extend(e.context or "" for e in ev.listening)
    scene_texts.extend(ev.episodes)
    scene_texts.extend(ev.raw_preferences)
    scene_texts.extend(memory.common_goals)
    for text in scene_texts:
        for scene in extract_scenario(text):
            ev.scene_hits[scene] = ev.scene_hits.get(scene, 0) + 1

    # 5) 排除项也参与语言判断（"不要中文歌" → 明确降低中文开放度）。
    for rule in ev.exclusions:
        ev.explicit_signal_count += 1
        for scene in extract_scenario(rule):
            ev.scene_hits.setdefault(scene, ev.scene_hits.get(scene, 0))

    return ev
