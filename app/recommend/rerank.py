"""Phase 1：三锚归一化精排 + 4 维 Jaccard 个性化 + MMR 多样性重排。

借鉴 SoulTuner 的三锚精排思想，但落到本项目的轻量栈上：
- 语义锚 semantic：sentence-transformers 可用时走 dense 向量，否则回退 TF 词项重叠。
- 个性化锚 personalize：4 维 Jaccard（genre/mood/scenario/theme）对用户偏好集合。
- 行为锚 behavior：BaRT 收听奖励（听完/秒跳），替代 SoulTuner 的 GPU 声学锚。

三锚权重自动归一化；某锚缺失（如无语义模型、无行为数据）时，其权重重分配给其余锚，
不让缺项把分数拉平。每个候选产出 RankingBreakdown 透明打分明细。

MMR 在三锚精排之后做多样性重排：mmr = λ·rel − (1−λ)·max_overlap，
overlap 用候选间 4 维标签 Jaccard 度量，避免连续推荐高度同质的歌。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from app.config import settings
from app.models import Asset, ExternalTrack, RankingBreakdown, TasteProfile

# 4 维个性化 Jaccard 权重（对齐 SoulTuner Graph Affinity 四维）。
DIM_WEIGHTS = {"genre": 0.30, "mood": 0.30, "scenario": 0.25, "theme": 0.15}


@dataclass
class PreferenceProfile:
    """用户偏好集合，供个性化锚做 Jaccard。"""
    genres: set[str]
    moods: set[str]
    scenarios: set[str]
    themes: set[str]

    @classmethod
    def from_taste(
        cls,
        taste: TasteProfile | None,
        scenarios: set[str] | None = None,
        themes: set[str] | None = None,
    ) -> "PreferenceProfile":
        genres = {g for g, _ in taste.top_genres} if taste else set()
        moods = {m for m, _ in taste.top_moods} if taste else set()
        return cls(
            genres={g.lower() for g in genres},
            moods={m.lower() for m in moods},
            scenarios={s.lower() for s in (scenarios or set())},
            themes={t.lower() for t in (themes or set())},
        )


def _tokens(text: str) -> set[str]:
    """轻量分词：英文按词、中文按字 bigram，用于 TF 回退语义。"""
    text = text.lower()
    en = set(re.findall(r"[a-z0-9]+", text))
    zh_chars = re.findall(r"[一-鿿]", text)
    bigrams = {"".join(pair) for pair in zip(zh_chars, zh_chars[1:])}
    return en | set(zh_chars) | bigrams


# 同义词组：跨语言/跨维度的情绪-风格关联，让 TF 回退不再纯随机。
# 每组内的词互为同义，query 和 track 分别命中间组的不同词时给 boost。
_SYNONYM_GROUPS: list[set[str]] = [
    # 情绪/氛围
    {"chill", "放松", "慵懒", "lofi", "relax", "relaxing", "舒缓"},
    {"伤感", "sad", "悲伤", "忧郁", "低落", "emo"},
    {"欢快", "happy", "开心", "轻松", "upbeat"},
    {"治愈", "healing", "温暖", "温柔", "comfort"},
    {"激昂", "热血", "energetic", "pump", "intense", "兴奋"},
    {"浪漫", "romantic", "浪漫的", "甜蜜", "romance"},
    {"梦幻", "dreamy", "dream", "空灵", "ethereal"},
    {"孤独", "lonely", "alone", "寂寞"},
    {"律动", "groove", "groovy", "rhythmic", "节奏"},
    {"暗黑", "dark", "暗夜", "midnight"},
    # 曲风
    {"说唱", "rap", "hip-hop", "hiphop", "hip hop", "trap"},
    {"r&b", "rnb", "soul", "neo-soul", "neosoul", "r and b"},
    {"电子", "electronic", "edm", "techno", "house", "electro"},
    {"摇滚", "rock", "rocknroll", "punk"},
    {"爵士", "jazz", "jazzy", "swing"},
    {"民谣", "folk", "acoustic", "indie folk"},
    {"流行", "pop", "popular", "mainstream"},
    {"古典", "classical", "classic", "orchestra"},
    {"国风", "古风", "中国风", "chinese traditional"},
    {"金属", "metal", "heavy metal", "metalcore"},
    # 场景
    {"运动", "跑步", "workout", "running", "exercise", "gym", "健身"},
    {"学习", "study", "studying", "专注", "focus", "concentrate"},
    {"睡眠", "sleep", "睡前", "asleep", "助眠"},
    {"派对", "party", "club", "聚会", "蹦迪"},
]


def _synonym_boost(query_tokens: set[str], track_tokens: set[str]) -> float:
    """检查 query 和 track 是否有同义词组重叠，有则返回 boost 分。"""
    q_lower = {t.lower() for t in query_tokens}
    t_lower = {t.lower() for t in track_tokens}
    for group in _SYNONYM_GROUPS:
        q_hit = bool(q_lower & group)
        t_hit = bool(t_lower & group)
        if q_hit and t_hit:
            return 0.3  # 同义词组重叠 → 温和 boost
    return 0.0


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def detect_language(track: Any) -> str:
    """判断一首歌的主语言：'zh'（中文为主）或 'en'（非中文/英文为主）。

    启发式：看标题+歌手里 CJK 字符 vs 拉丁字母的占比。两者都没有时归到 'en'
    （多数纯符号/数字标题是西文曲目）。够用、零依赖。
    """
    text = f"{getattr(track, 'title', '') or ''} {getattr(track, 'artist', '') or ''}"
    cjk = len(re.findall(r"[一-鿿]", text))
    latin = len(re.findall(r"[a-zA-Z]", text))
    if cjk == 0 and latin == 0:
        return "en"
    return "zh" if cjk >= latin else "en"


def language_distribution(tracks: list[Any]) -> dict[str, float]:
    """统计一批曲目的语言占比，返回 {'zh': x, 'en': y}（和为 1）。

    空库返回均衡分布，避免冷启动时把任一语言压到 0。
    """
    if not tracks:
        return {"zh": 0.5, "en": 0.5}
    counts = {"zh": 0, "en": 0}
    for t in tracks:
        counts[detect_language(t)] += 1
    total = counts["zh"] + counts["en"]
    return {"zh": counts["zh"] / total, "en": counts["en"] / total}


def _apply_exclusion_filter(tracks: list[Any], rules: list[str]) -> list[Any]:
    """排除命中用户排除规则的候选。规则作为子串匹配候选的 title+artist+genre+mood。"""
    filtered: list[Any] = []
    for t in tracks:
        title = (getattr(t, "title", "") or "").lower()
        artist = (getattr(t, "artist", "") or "").lower()
        genres = " ".join(getattr(t, "genre", []) or []).lower()
        moods = " ".join(getattr(t, "mood", []) or []).lower()
        combined = f"{title} {artist} {genres} {moods}"
        excluded = any(rule.lower() in combined for rule in rules)
        if not excluded:
            filtered.append(t)
    return filtered


def _track_tags(track: Any) -> dict[str, set[str]]:
    genre = {g.lower() for g in (getattr(track, "genre", []) or [])}
    mood = {m.lower() for m in (getattr(track, "mood", []) or [])}
    return {"genre": genre, "mood": mood, "scenario": set(), "theme": set()}


def _track_text(track: Any) -> str:
    parts = [
        getattr(track, "title", "") or "",
        getattr(track, "artist", "") or "",
        " ".join(getattr(track, "genre", []) or []),
        " ".join(getattr(track, "mood", []) or []),
    ]
    return " ".join(p for p in parts if p)


def _track_id(track: Any) -> str:
    if isinstance(track, Asset):
        return track.asset_id
    return getattr(track, "external_id", "") or getattr(track, "title", "")


def _semantic_anchor(query: str, tracks: list[Any]) -> tuple[list[float], bool]:
    """返回 (每个候选的语义分[0,1], 是否可用)。

    优先 dense 向量；不可用时回退 TF 词项 Jaccard + 同义词 boost，并标记
    available=False（TF 回退质量较弱，作为弱锚——但仍参与，不直接丢权重）。
    """
    texts = [_track_text(t) for t in tracks]
    try:
        from app.retrieval.embeddings import semantic_scores

        dense = semantic_scores(query, texts)
    except Exception:
        dense = None
    if dense is not None:
        return dense, True
    # TF 回退 + 同义词 boost
    q_tokens = _tokens(query)
    scores: list[float] = []
    for text in texts:
        base = _jaccard(q_tokens, _tokens(text))
        boost = _synonym_boost(q_tokens, _tokens(text))
        scores.append(min(base + boost, 1.0))
    return scores, False


def _personalize_anchor(tracks: list[Any], profile: PreferenceProfile) -> list[float]:
    scores: list[float] = []
    for track in tracks:
        tags = _track_tags(track)
        weighted = (
            DIM_WEIGHTS["genre"] * _jaccard(profile.genres, tags["genre"])
            + DIM_WEIGHTS["mood"] * _jaccard(profile.moods, tags["mood"])
            + DIM_WEIGHTS["scenario"] * _jaccard(profile.scenarios, tags["scenario"])
            + DIM_WEIGHTS["theme"] * _jaccard(profile.themes, tags["theme"])
        )
        scores.append(weighted)
    return scores


def _behavior_anchor(tracks: list[Any], behavior_scores: dict[str, float] | None) -> tuple[list[float], bool]:
    """BaRT 行为奖励归一化到 [0,1]：约 3 次听完达上限。无行为数据则 available=False。"""
    if not behavior_scores:
        return [0.0] * len(tracks), False
    out: list[float] = []
    any_hit = False
    for track in tracks:
        raw = behavior_scores.get(_track_id(track), 0.0)
        if raw:
            any_hit = True
        reward = max(-1.0, min(1.0, raw / 3.0))
        out.append((reward + 1.0) / 2.0)  # [-1,1] → [0,1]
    return out, any_hit


def _normalized_weights(semantic_ok: bool, behavior_ok: bool) -> tuple[float, float, float]:
    """三锚权重归一化 + 缺项重分配（对齐 SoulTuner 缺声学锚时的降级）。"""
    w_sem = settings.tri_anchor_w_semantic if semantic_ok else 0.0
    w_beh = settings.tri_anchor_w_behavior if behavior_ok else 0.0
    w_per = settings.tri_anchor_w_personal
    total = w_sem + w_beh + w_per
    if total <= 0:
        return 0.0, 1.0, 0.0  # 全缺时只靠个性化
    return w_sem / total, w_per / total, w_beh / total


def _language_multiplier(track: Any, lang_pref: dict[str, float] | None) -> float:
    """按曲库语言分布给候选一个温和的乘性权重（不排斥任何语言）。

    思路：曲库里某语言占比越高，该语言候选越被偏好，但只是"加权"不是"过滤"——
    占比低的语言仍保留可观权重，避免一刀切。映射：multiplier = 0.85 + 0.30*share，
    即占比 0 的语言仍有 0.85，占比 1 的语言到 1.15，差距温和（约 35%）。
    """
    if not lang_pref:
        return 1.0
    share = lang_pref.get(detect_language(track), 0.5)
    return 0.85 + 0.30 * share


def tri_anchor_rerank(
    query: str,
    tracks: list[Any],
    profile: PreferenceProfile,
    behavior_scores: dict[str, float] | None = None,
    lang_pref: dict[str, float] | None = None,
) -> list[tuple[Any, RankingBreakdown]]:
    """三锚归一化精排。返回按 final_score 降序的 (track, breakdown) 列表。

    lang_pref：曲库语言分布 {'zh':x,'en':y}。给定时按分布对候选做温和的语言加权
    （英文歌多则多推英文，但中文仍保留），实现用户"按曲库语言偏好推荐"的需求。
    """
    if not tracks:
        return []
    semantic, semantic_ok = _semantic_anchor(query, tracks)
    personalize = _personalize_anchor(tracks, profile)
    behavior, behavior_ok = _behavior_anchor(tracks, behavior_scores)
    w_sem, w_per, w_beh = _normalized_weights(semantic_ok, behavior_ok)

    scored: list[tuple[Any, RankingBreakdown]] = []
    for i, track in enumerate(tracks):
        base = w_sem * semantic[i] + w_per * personalize[i] + w_beh * behavior[i]
        lang_mult = _language_multiplier(track, lang_pref)
        # 乘性加权在 base>0 时倾斜同语言候选；额外加一个很小的加性语言先验，
        # 让 base≈0（冷启动/泛查询无任何锚信号）时语言偏好仍能打破平局，
        # 但量级（≤0.05）远小于真实相关性差异，不会盖过口味/语义。
        lang_prior = 0.05 * lang_pref.get(detect_language(track), 0.5) if lang_pref else 0.0
        final = base * lang_mult + lang_prior
        breakdown = RankingBreakdown(
            title=getattr(track, "title", ""),
            source=getattr(track, "source", "local"),
            score=round(final, 4),
            reason=_reason(semantic[i], personalize[i], behavior[i], w_sem, w_per, w_beh),
            components={
                "semantic": round(semantic[i], 4),
                "personalize": round(personalize[i], 4),
                "behavior": round(behavior[i], 4),
                "w_semantic": round(w_sem, 3),
                "w_personalize": round(w_per, 3),
                "w_behavior": round(w_beh, 3),
                "lang_mult": round(lang_mult, 3),
            },
        )
        scored.append((track, breakdown))
    scored.sort(key=lambda x: x[1].score, reverse=True)
    return scored


def _reason(sem: float, per: float, beh: float, w_sem: float, w_per: float, w_beh: float) -> str:
    contributions = {
        "语义匹配": w_sem * sem,
        "口味契合": w_per * per,
        "收听行为": w_beh * beh,
    }
    top = max(contributions, key=contributions.get)
    return f"{top}主导（语义{sem:.2f}/口味{per:.2f}/行为{beh:.2f}）"


def mmr_rerank(
    scored: list[tuple[Any, RankingBreakdown]],
    top_k: int,
    lambda_: float | None = None,
) -> list[tuple[Any, RankingBreakdown]]:
    """MMR 多样性重排：在相关性与多样性间平衡，避免连续同质推荐。"""
    if not scored:
        return []
    lam = settings.mmr_lambda if lambda_ is None else lambda_
    remaining = list(scored)
    selected: list[tuple[Any, RankingBreakdown]] = [remaining.pop(0)]
    sel_tagsets = [_all_tags(selected[0][0])]

    while remaining and len(selected) < top_k:
        best_idx, best_score = 0, -float("inf")
        for idx, (track, bd) in enumerate(remaining):
            rel = bd.score
            cand_tags = _all_tags(track)
            max_overlap = max((_jaccard(cand_tags, st) for st in sel_tagsets), default=0.0)
            # 多样性惩罚按候选自身相关性缩放：低相关的垃圾候选不该靠"多样"翻盘到
            # 高相关候选前面（否则 rel≈0 的噪声会反超 rel 明显更高的同质好候选）。
            penalty = (1 - lam) * max_overlap * rel
            mmr = lam * rel - penalty
            if mmr > best_score:
                best_score, best_idx = mmr, idx
        chosen = remaining.pop(best_idx)
        selected.append(chosen)
        sel_tagsets.append(_all_tags(chosen[0]))
    return selected


def _all_tags(track: Any) -> set[str]:
    tags = _track_tags(track)
    return tags["genre"] | tags["mood"] | tags["scenario"] | tags["theme"]


def rerank_candidates(
    query: str,
    tracks: list[Any],
    taste: TasteProfile | None,
    behavior_scores: dict[str, float] | None = None,
    scenarios: set[str] | None = None,
    top_k: int = 5,
    apply_mmr: bool = True,
    lang_pref: dict[str, float] | None = None,
    exclusion_rules: list[str] | None = None,
) -> list[tuple[Any, RankingBreakdown]]:
    """精排管线入口：排除过滤 → 三锚精排 → MMR 多样性重排 → 取 top_k。

    exclusion_rules：用户排除规则（如"抖音热歌"），候选匹配则丢弃。
    lang_pref：曲库语言分布，传入时按分布对候选做温和语言加权。
    """
    # 排除过滤：先于精排，命中排除规则的候选直接丢弃
    if exclusion_rules:
        tracks = _apply_exclusion_filter(tracks, exclusion_rules)

    profile = PreferenceProfile.from_taste(taste, scenarios=scenarios)
    scored = tri_anchor_rerank(query, tracks, profile, behavior_scores, lang_pref=lang_pref)
    if apply_mmr:
        return mmr_rerank(scored, top_k=top_k)
    return scored[:top_k]
