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


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


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

    优先 dense 向量；不可用时回退 TF 词项 Jaccard，并标记 available=False
    （TF 回退质量较弱，作为弱锚——但仍参与，不直接丢权重）。
    """
    texts = [_track_text(t) for t in tracks]
    try:
        from app.retrieval.embeddings import semantic_scores

        dense = semantic_scores(query, texts)
    except Exception:
        dense = None
    if dense is not None:
        return dense, True
    # TF 回退
    q_tokens = _tokens(query)
    return [_jaccard(q_tokens, _tokens(text)) for text in texts], False


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


def tri_anchor_rerank(
    query: str,
    tracks: list[Any],
    profile: PreferenceProfile,
    behavior_scores: dict[str, float] | None = None,
) -> list[tuple[Any, RankingBreakdown]]:
    """三锚归一化精排。返回按 final_score 降序的 (track, breakdown) 列表。"""
    if not tracks:
        return []
    semantic, semantic_ok = _semantic_anchor(query, tracks)
    personalize = _personalize_anchor(tracks, profile)
    behavior, behavior_ok = _behavior_anchor(tracks, behavior_scores)
    w_sem, w_per, w_beh = _normalized_weights(semantic_ok, behavior_ok)

    scored: list[tuple[Any, RankingBreakdown]] = []
    for i, track in enumerate(tracks):
        final = w_sem * semantic[i] + w_per * personalize[i] + w_beh * behavior[i]
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
            mmr = lam * rel - (1 - lam) * max_overlap
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
) -> list[tuple[Any, RankingBreakdown]]:
    """精排管线入口：三锚精排 → MMR 多样性重排 → 取 top_k。"""
    profile = PreferenceProfile.from_taste(taste, scenarios=scenarios)
    scored = tri_anchor_rerank(query, tracks, profile, behavior_scores)
    if apply_mmr:
        return mmr_rerank(scored, top_k=top_k)
    return scored[:top_k]
