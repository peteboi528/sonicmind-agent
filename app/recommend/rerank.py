"""Phase 1：三锚归一化精排 + 4 维 Jaccard 个性化 + MMR 多样性重排。

借鉴 SoulTuner 的三锚精排思想，但落到本项目的轻量栈上：
- 语义锚 semantic：sentence-transformers 可用时走 dense 向量，否则回退 TF 词项重叠。
- 个性化锚 personalize：4 维 Jaccard（genre/mood/scenario/theme）对用户偏好集合。
- 行为锚 behavior：BaRT 收听奖励（听完/秒跳），替代 SoulTuner 的 GPU 声学锚。

三锚权重自动归一化；某锚缺失（如无行为数据）时，其权重重分配给其余锚，
不让缺项把分数拉平。语义锚无 dense 向量时回退 TF 词项重叠——仍作为有效弱锚参与
（不再清零），避免三锚退化成单锚。每个候选产出 RankingBreakdown 透明打分明细。

MMR 在三锚精排之后做多样性重排：mmr = λ·rel − (1−λ)·max_overlap，
overlap 用候选间 4 维标签 Jaccard 度量，避免连续推荐高度同质的歌。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from app.config import settings
from app.models import Asset, RankingBreakdown, TasteProfile

# 4 维个性化 Jaccard 权重（对齐 SoulTuner Graph Affinity 四维）。
DIM_WEIGHTS = {"genre": 0.30, "mood": 0.30, "scenario": 0.25, "theme": 0.15}

# dense 语义分对比归一化的最小区间阈值：batch 内 max-min 小于此值时视为高度同质，
# 不做对比拉伸（避免把一批都相关的候选里最低那个误压到 0）。
_CONTRASTIVE_MIN_SPREAD = 0.05


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
    ) -> PreferenceProfile:
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
    bigrams = {"".join(pair) for pair in zip(zh_chars, zh_chars[1:], strict=False)}
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


def _artist_matches(track: Any, names: set[str]) -> bool:
    """候选 artist 是否命中画像艺人集合（子串匹配，候选 artist 形如「A、B」）。"""
    if not names:
        return False
    artist = (getattr(track, "artist", "") or "").lower()
    if not artist:
        return False
    return any(n in artist for n in names)


def apply_profile_artist_adjust(
    scored: list[tuple[Any, RankingBreakdown]],
    boost: set[str] | None,
    penalty: set[str] | None,
) -> list[tuple[Any, RankingBreakdown]]:
    """画像艺人关系（core/rising 加分，avoid 减分）作为三锚之外的小幅调整。

    就地改 breakdown.score 后按分数重排。量级（+0.06/-0.12）远小于真实相关性差异，
    只起「同分段时倾向画像偏好艺人、避开画像 dislike 艺人」的微调，不盖过语义/口味锚。
    MMR 读 bd.score，故调整自然影响最终排序。boost/penalty 为空时原样返回（保持旧行为）。
    """
    boost_n = {b.lower() for b in (boost or [])}
    penalty_n = {p.lower() for p in (penalty or [])}
    if not boost_n and not penalty_n:
        return scored
    for _track, bd in scored:
        delta = 0.0
        label = ""
        if _artist_matches(_track, penalty_n):
            delta, label = -0.12, "profile_avoid_artist"
        elif _artist_matches(_track, boost_n):
            delta, label = 0.06, "profile_core_artist"
        if delta:
            bd.score = round(max(0.0, bd.score + delta), 4)
            bd.components[label] = round(delta, 3)
    scored.sort(key=lambda x: x[1].score, reverse=True)
    return scored


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


def _contrastive_normalize(scores: list[float]) -> list[float]:
    """dense 余弦在多语言模型下被压进高位窄区间（中文短语对任意 query 都 0.7+），
    绝对值失真：无关垃圾也能拿 0.77，被 MMR 多样性抬到相关候选之前。

    解法：减去 batch 均值做对比基线，负差截断到 0——低于平均相关度的候选（真正的
    离群垃圾）语义分归零，高于均值的保留其相对差值。不除以 spread（不强行拉满
    [0,1]），保留差异的绝对量级，避免把「都相关、仅 query 语言偏置」的微小差放大成
    一边倒，从而盖过语言/口味加权。区间过窄（候选高度同质）时不动，原样返回。
    """
    if len(scores) < 2:
        return scores
    if max(scores) - min(scores) < _CONTRASTIVE_MIN_SPREAD:
        return scores  # 高度同质，保留原始绝对分
    mean = sum(scores) / len(scores)
    return [max(0.0, s - mean) for s in scores]


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
        return _contrastive_normalize(dense), True
    # TF 回退 + 同义词 boost：无 sentence-transformers 时仍作为有效弱锚参与精排。
    # 旧实现返回 available=False，导致 _normalized_weights 把语义权重清零、重分配给
    # 个性化锚——正确的语义匹配（如 query「说唱」命中 Eminem）被整体丢弃，三锚退化成单锚。
    q_tokens = _tokens(query)
    scores: list[float] = []
    for text in texts:
        base = _jaccard(q_tokens, _tokens(text))
        boost = _synonym_boost(q_tokens, _tokens(text))
        scores.append(min(base + boost, 1.0))
    return scores, True


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
    # 有历史但当前候选一个都没命中时，0.5 只是数学上的“中性值”，不能在 UI 中
    # 冒充行为证据，更不能参与实验熟悉度。明确归零并关闭该锚。
    return (out, True) if any_hit else ([0.0] * len(tracks), False)


def _normalized_weights(
    semantic_ok: bool,
    behavior_ok: bool,
    collaborative_ok: bool = False,
    explore_ok: bool = False,
) -> tuple[float, float, float, float, float]:
    """多锚权重归一化 + 缺项重分配（对齐 SoulTuner 缺声学锚时的降级）。

    返回 (w_semantic, w_personalize, w_behavior, w_collaborative, w_explore)。
    某锚不可用（无语义模型/无行为数据/无 CF 共现/无 TS 探索分）时其权重置 0，
    其余锚归一化吸收——缺 CF/TS 时退回旧精排行为。
    """
    w_sem = settings.tri_anchor_w_semantic if semantic_ok else 0.0
    w_beh = settings.tri_anchor_w_behavior if behavior_ok else 0.0
    w_col = settings.tri_anchor_w_collaborative if collaborative_ok else 0.0
    w_exp = settings.tri_anchor_w_explore if explore_ok else 0.0
    w_per = settings.tri_anchor_w_personal
    total = w_sem + w_beh + w_col + w_exp + w_per
    if total <= 0:
        return 0.0, 1.0, 0.0, 0.0, 0.0  # 全缺时只靠个性化
    return w_sem / total, w_per / total, w_beh / total, w_col / total, w_exp / total


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
    collaborative_scores: list[float] | None = None,
    collaborative_ok: bool = False,
    ts_scores: dict[str, float] | None = None,
) -> list[tuple[Any, RankingBreakdown]]:
    """四锚归一化精排（语义/个性化/行为 + 可选协同）。返回按 final_score 降序的 (track, breakdown)。

    lang_pref：曲库语言分布 {'zh':x,'en':y}。给定时按分布对候选做温和的语言加权
    （英文歌多则多推英文，但中文仍保留），实现用户"按曲库语言偏好推荐"的需求。
    collaborative_scores：可选 CF 第四锚（已归一到 [0,1]，与 tracks 等长）。
    collaborative_ok=False 时该锚权重重分配给其余锚，行为与三锚一致。
    ts_scores：可选 Thompson Sampling 探索分（ResourceLibrary.sample_ts_scores）。
    """
    if not tracks:
        return []
    semantic, semantic_ok = _semantic_anchor(query, tracks)
    personalize = _personalize_anchor(tracks, profile)
    behavior, behavior_ok = _behavior_anchor(tracks, behavior_scores)
    collab = collaborative_scores if (collaborative_ok and collaborative_scores and len(collaborative_scores) == len(tracks)) else None
    collab_ok = collab is not None
    explore_ok = bool(settings.enable_explore and ts_scores)
    w_sem, w_per, w_beh, w_col, w_exp = _normalized_weights(semantic_ok, behavior_ok, collab_ok, explore_ok)

    scored: list[tuple[Any, RankingBreakdown]] = []
    for i, track in enumerate(tracks):
        cf_i = collab[i] if collab is not None else 0.0
        exp_i = _ts_score_for_track(ts_scores, track) if explore_ok else 0.0
        base = w_sem * semantic[i] + w_per * personalize[i] + w_beh * behavior[i] + w_col * cf_i + w_exp * exp_i
        lang_mult = _language_multiplier(track, lang_pref)
        # 乘性加权在 base>0 时倾斜同语言候选；额外加一个很小的加性语言先验，
        # 让 base≈0（冷启动/泛查询无任何锚信号）时语言偏好仍能打破平局，
        # 但量级（≤0.05）远小于真实相关性差异，不会盖过口味/语义。
        lang_prior = 0.05 * lang_pref.get(detect_language(track), 0.5) if lang_pref else 0.0
        final = base * lang_mult + lang_prior
        components = {
            "semantic": round(semantic[i], 4),
            "personalize": round(personalize[i], 4),
            "behavior": round(behavior[i], 4),
            "w_semantic": round(w_sem, 3),
            "w_personalize": round(w_per, 3),
            "w_behavior": round(w_beh, 3),
            "lang_mult": round(lang_mult, 3),
        }
        if collab_ok:
            components["collaborative"] = round(cf_i, 4)
            components["w_collaborative"] = round(w_col, 3)
        if explore_ok:
            components["explore"] = round(exp_i, 4)
            components["w_explore"] = round(w_exp, 3)
        breakdown = RankingBreakdown(
            title=getattr(track, "title", ""),
            source=getattr(track, "source", "local"),
            score=round(final, 4),
            reason=_reason(semantic[i], personalize[i], behavior[i], w_sem, w_per, w_beh, cf_i, w_col, exp_i, w_exp),
            components=components,
        )
        scored.append((track, breakdown))
    scored.sort(key=lambda x: x[1].score, reverse=True)
    return scored


def _ts_score_for_track(ts_scores: dict[str, float] | None, track: Any) -> float:
    if not ts_scores:
        return 0.0
    source = getattr(track, "source", "") or "local"
    source_id = getattr(track, "source_id", None) or getattr(track, "external_id", None) or getattr(track, "asset_id", "") or ""
    title = (getattr(track, "title", "") or "").strip().lower()
    artist = (getattr(track, "artist", "") or "").strip().lower()
    keys = [
        f"{source}|{source_id}|{title}|{artist}",
        getattr(track, "external_id", "") or getattr(track, "asset_id", ""),
        getattr(track, "title", ""),
    ]
    for key in keys:
        if key in ts_scores:
            return max(0.0, min(1.0, float(ts_scores[key])))
    return 0.0


def _reason(
    sem: float,
    per: float,
    beh: float,
    w_sem: float,
    w_per: float,
    w_beh: float,
    cf: float = 0.0,
    w_col: float = 0.0,
    exp: float = 0.0,
    w_exp: float = 0.0,
) -> str:
    contributions = {
        "语义匹配": w_sem * sem,
        "口味契合": w_per * per,
        "收听行为": w_beh * beh,
    }
    if w_col > 0:
        contributions["协同推荐"] = w_col * cf
    if w_exp > 0:
        contributions["探索潜力"] = w_exp * exp
    top = max(contributions, key=contributions.get)
    tail = f"/协同{cf:.2f}" if w_col > 0 else ""
    explore_tail = f"/探索{exp:.2f}" if w_exp > 0 else ""
    return f"{top}主导（语义{sem:.2f}/口味{per:.2f}/行为{beh:.2f}{tail}{explore_tail}）"


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


def bandit_select(
    scored: list[tuple[Any, RankingBreakdown]],
    top_k: int,
    explore_ratio: float | None = None,
) -> list[tuple[Any, RankingBreakdown]]:
    """Exploit + explore 混合选择：多数位置给综合分，少数位置给 TS 探索潜力。

    这一步发生在候选已通过真实来源验证和排除规则之后，因此 explore 不是造歌，
    而是在可信候选里主动留出探索槽。
    """
    if not scored or top_k <= 0:
        return []
    ratio = settings.explore_ratio if explore_ratio is None else explore_ratio
    ratio = max(0.0, min(1.0, ratio))
    if ratio <= 0:
        return mmr_rerank(scored, top_k=top_k)
    explore_n = int(round(top_k * ratio))
    if top_k > 1 and explore_n <= 0:
        explore_n = 1
    explore_n = min(explore_n, top_k, len(scored))
    exploit_n = max(0, min(top_k - explore_n, len(scored)))

    selected = mmr_rerank(scored, top_k=exploit_n) if exploit_n else []
    selected_ids = {_track_identity(t) for t, _ in selected}
    remaining = [(t, bd) for t, bd in scored if _track_identity(t) not in selected_ids]
    remaining.sort(key=lambda item: (item[1].components.get("explore", 0.0), item[1].score), reverse=True)
    for item in remaining:
        if len(selected) >= top_k:
            break
        selected.append(item)
        selected_ids.add(_track_identity(item[0]))
    return selected[:top_k]


def _all_tags(track: Any) -> set[str]:
    tags = _track_tags(track)
    return tags["genre"] | tags["mood"] | tags["scenario"] | tags["theme"]


def _track_identity(track: Any) -> tuple[str, str, str, str]:
    return (
        getattr(track, "source", "") or "local",
        getattr(track, "source_id", None) or getattr(track, "external_id", None) or getattr(track, "asset_id", "") or "",
        (getattr(track, "title", "") or "").strip().lower(),
        (getattr(track, "artist", "") or "").strip().lower(),
    )


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
    collaborative_scores: list[float] | None = None,
    collaborative_ok: bool = False,
    ts_scores: dict[str, float] | None = None,
    profile_boost_artists: set[str] | None = None,
    profile_penalty_artists: set[str] | None = None,
) -> list[tuple[Any, RankingBreakdown]]:
    """精排管线入口：排除过滤 → 四锚精排 → 画像艺人微调 → MMR 多样性重排 → 取 top_k。

    exclusion_rules：用户排除规则（如"抖音热歌"），候选匹配则丢弃。
    lang_pref：曲库语言分布，传入时按分布对候选做温和语言加权。
    collaborative_scores/collaborative_ok：可选 CF 第四锚（须与过滤后 tracks 对齐）。
    profile_boost_artists/penalty_artists：画像仪表盘艺人关系（core/rising 加分、avoid 减分），
        与 memory.taste_profile（频次）互补。空集时不调整，行为与旧版一致。
    ts_scores：可选 TS 探索锚，开启 ENABLE_EXPLORE 时用于留出探索槽。
    """
    # 排除过滤：先于精排，命中排除规则的候选直接丢弃
    if exclusion_rules:
        before = tracks
        tracks = _apply_exclusion_filter(tracks, exclusion_rules)
        # 排除改变了候选集合，CF 分数会错位——失配时安全关闭 CF 锚。
        if collaborative_scores is not None and len(tracks) != len(before):
            collaborative_scores, collaborative_ok = None, False

    profile = PreferenceProfile.from_taste(taste, scenarios=scenarios)
    scored = tri_anchor_rerank(
        query, tracks, profile, behavior_scores, lang_pref=lang_pref,
        collaborative_scores=collaborative_scores, collaborative_ok=collaborative_ok,
        ts_scores=ts_scores,
    )
    # 画像艺人微调（core/rising 加分、avoid 减分）：小幅、可选，空集时原样返回。
    scored = apply_profile_artist_adjust(scored, profile_boost_artists, profile_penalty_artists)
    if apply_mmr:
        if settings.enable_explore and ts_scores:
            return bandit_select(scored, top_k=top_k)
        return mmr_rerank(scored, top_k=top_k)
    return scored[:top_k]
