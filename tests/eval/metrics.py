"""Eval 指标：确定性能跑（mock 模式）的部分 + 可选 LLM 相关性。

四指标（对应计划 P1-D）：
  intent_hit        预期意图是否命中 agent_trace（子串匹配）。确定性。
  anti_halluc_pass  must_not_mention 禁词零泄漏。确定性（反幻觉硬约束）。
  must_mention_hit  必现关键词命中率 [0,1]。确定性。
  relevance         LLM judge 综合分（0-5）；无 key 时回退 None。

diversity 现在既可在 A/B rerank 层度量，也可基于 AgentAnswer.recommended_tracks
做端到端推荐多样性。
"""
from __future__ import annotations

from typing import Any

from app.models import AgentAnswer
from tests.eval.cases import EvalCase


def intent_hit(case: EvalCase, answer: AgentAnswer) -> bool | None:
    """预期意图是否作为子串出现在 agent_trace 里。expected_intent 为 None 时返回 None（不检查）。"""
    if not case.expected_intent:
        return None
    blob = " ".join(answer.agent_trace).lower()
    return case.expected_intent.lower() in blob


def anti_halluc_pass(case: EvalCase, answer: AgentAnswer) -> bool:
    """must_not_mention 禁词是否零泄漏（反幻觉/兜底检测，确定性硬约束）。"""
    return all(kw not in answer.answer for kw in case.must_not_mention)


def junk_rate(case: EvalCase, answer: AgentAnswer) -> float:
    """禁词/脏结果泄漏率。0=完全干净，1=禁词全命中。"""
    if not case.must_not_mention:
        return 0.0
    leaked = sum(1 for kw in case.must_not_mention if kw in answer.answer)
    return leaked / len(case.must_not_mention)


def must_mention_hit(case: EvalCase, answer: AgentAnswer) -> float:
    if not case.must_mention:
        return 1.0
    hit = sum(1 for kw in case.must_mention if kw in answer.answer)
    return hit / len(case.must_mention)


def local_ratio(answer: AgentAnswer) -> float | None:
    tracks = answer.recommended_tracks or []
    if not tracks:
        return None
    local_n = sum(1 for track in tracks if str(getattr(track, "source", "") or "").startswith("local"))
    return local_n / len(tracks)


def compute_metrics(
    case: EvalCase, answer: AgentAnswer, relevance: float | None = None
) -> dict[str, Any]:
    """汇总单个 case 的指标字典。relevance 由调用方传入（有 LLM key 时取 judge 分）。"""
    diversity = None
    if answer.recommended_tracks:
        diversity = intra_list_diversity(answer.recommended_tracks)
    ratio = local_ratio(answer)
    prompt_signature = ", ".join(
        f"{key}={value}" for key, value in sorted(answer.prompt_versions.items())
    )
    return {
        "intent_hit": intent_hit(case, answer),
        "anti_halluc_pass": anti_halluc_pass(case, answer),
        "junk_rate": round(junk_rate(case, answer), 3),
        "must_mention_hit": round(must_mention_hit(case, answer), 3),
        "local_ratio": round(ratio, 3) if ratio is not None else None,
        "local_ratio_ok": (ratio <= case.max_local_ratio) if ratio is not None and case.max_local_ratio is not None else None,
        "answer_len": len(answer.answer),
        "trace_steps": len(answer.agent_trace),
        "recommended_tracks": len(answer.recommended_tracks),
        "diversity": diversity,
        "prompt_signature": prompt_signature,
        "fallback": bool(answer.fallback_reason),
        "relevance": round(relevance, 3) if relevance is not None else None,
    }


def aggregate(per_case: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """跨 case 聚合：意图命中率 / 反幻觉通过率 / 平均 must_mention / 平均 relevance。

    intent_hit 为 None 的 case（未标 expected_intent）不计入意图命中率分母。
    skipped（requires_llm 且在 mock 模式跳过）的 case 完全不计入任何聚合。
    """
    cases = [v for k, v in per_case.items() if not k.startswith("__") and not v.get("skipped")]
    intent_checked = [c for c in cases if c["intent_hit"] is not None]
    rel = [c["relevance"] for c in cases if c["relevance"] is not None]
    return {
        "n_total_cases": len([v for k, v in per_case.items() if not k.startswith("__")]),
        "n_skipped_cases": len([v for k, v in per_case.items() if not k.startswith("__") and v.get("skipped")]),
        "intent_hit_rate": (
            sum(1 for c in intent_checked if c["intent_hit"]) / len(intent_checked)
            if intent_checked else None
        ),
        "anti_halluc_rate": sum(1 for c in cases if c["anti_halluc_pass"]) / len(cases) if cases else None,
        "avg_junk_rate": (
            sum(c.get("junk_rate", 0.0) for c in cases) / len(cases) if cases else None
        ),
        "avg_must_mention_hit": (
            sum(c["must_mention_hit"] for c in cases) / len(cases) if cases else None
        ),
        "avg_local_ratio": (
            sum(c["local_ratio"] for c in cases if c["local_ratio"] is not None) /
            len([c for c in cases if c["local_ratio"] is not None])
            if any(c["local_ratio"] is not None for c in cases) else None
        ),
        "local_ratio_pass_rate": (
            sum(1 for c in cases if c["local_ratio_ok"]) /
            len([c for c in cases if c["local_ratio_ok"] is not None])
            if any(c["local_ratio_ok"] is not None for c in cases) else None
        ),
        "avg_diversity": (
            sum(c["diversity"] for c in cases if c["diversity"] is not None) /
            len([c for c in cases if c["diversity"] is not None])
            if any(c["diversity"] is not None for c in cases) else None
        ),
        "avg_relevance": (sum(rel) / len(rel)) if rel else None,
        "n_cases": len(cases),
    }


def _track_tagset(track: Any) -> set[str]:
    """genre ∪ mood 的小写标签集合（与 rerank 多样性度量口径一致）。"""
    genre = {g.lower() for g in (getattr(track, "genre", []) or [])}
    mood = {m.lower() for m in (getattr(track, "mood", []) or [])}
    return genre | mood


def intra_list_diversity(tracks: list[Any]) -> float:
    """候选列表的「1 − 平均两两标签 Jaccard」。0=完全同质，1=完全多样。

    空列表/单元素返回 1.0（无同质问题）。用于 A/B rerank 对比。
    """
    if len(tracks) < 2:
        return 1.0
    tagsets = [_track_tagset(t) for t in tracks]
    n = len(tagsets)
    total = 0.0
    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            a, b = tagsets[i], tagsets[j]
            union = len(a | b)
            total += 1.0 - (len(a & b) / union if union else 0.0)
            pairs += 1
    return round(total / pairs, 3) if pairs else 1.0
