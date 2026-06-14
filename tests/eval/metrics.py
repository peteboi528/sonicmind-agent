"""Eval 指标：确定性能跑（mock 模式）的部分 + 可选 LLM 相关性。

四指标（对应计划 P1-D）：
  intent_hit        预期意图是否命中 agent_trace（子串匹配）。确定性。
  anti_halluc_pass  must_not_mention 禁词零泄漏。确定性（反幻觉硬约束）。
  must_mention_hit  必现关键词命中率 [0,1]。确定性。
  relevance         LLM judge 综合分（0-5）；无 key 时回退 None。

diversity 在 A/B rerank 层度量（AgentAnswer 不暴露推荐曲目，端到端暂不可算；
后续给 AgentAnswer 加 recommended_tracks 字段即可补齐端到端多样性）。
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


def must_mention_hit(case: EvalCase, answer: AgentAnswer) -> float:
    if not case.must_mention:
        return 1.0
    hit = sum(1 for kw in case.must_mention if kw in answer.answer)
    return hit / len(case.must_mention)


def compute_metrics(
    case: EvalCase, answer: AgentAnswer, relevance: float | None = None
) -> dict[str, Any]:
    """汇总单个 case 的指标字典。relevance 由调用方传入（有 LLM key 时取 judge 分）。"""
    return {
        "intent_hit": intent_hit(case, answer),
        "anti_halluc_pass": anti_halluc_pass(case, answer),
        "must_mention_hit": round(must_mention_hit(case, answer), 3),
        "answer_len": len(answer.answer),
        "trace_steps": len(answer.agent_trace),
        "fallback": bool(answer.fallback_reason),
        "relevance": round(relevance, 3) if relevance is not None else None,
    }


def aggregate(per_case: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """跨 case 聚合：意图命中率 / 反幻觉通过率 / 平均 must_mention / 平均 relevance。

    intent_hit 为 None 的 case（未标 expected_intent）不计入意图命中率分母。
    """
    cases = [v for k, v in per_case.items() if not k.startswith("__")]
    intent_checked = [c for c in cases if c["intent_hit"] is not None]
    rel = [c["relevance"] for c in cases if c["relevance"] is not None]
    return {
        "intent_hit_rate": (
            sum(1 for c in intent_checked if c["intent_hit"]) / len(intent_checked)
            if intent_checked else None
        ),
        "anti_halluc_rate": sum(1 for c in cases if c["anti_halluc_pass"]) / len(cases) if cases else None,
        "avg_must_mention_hit": (
            sum(c["must_mention_hit"] for c in cases) / len(cases) if cases else None
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
