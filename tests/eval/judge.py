"""LLM-as-judge：用强模型对 agent 回复打分。

Usage:
    judge = LLMJudge(llm)
    score = judge.evaluate(case, agent_answer)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.llm.protocol import LLMProvider
from app.llm.structured import extract_json_dict
from tests.eval.cases import EvalCase


@dataclass
class JudgeScore:
    case_id: str
    overall: float  # 0-5
    per_criterion: dict[str, float] = field(default_factory=dict)
    mention_hit: float = 0.0  # must_mention 命中率
    mention_miss: float = 0.0  # must_not_mention 是否泄漏（违规扣分）
    rationale: str = ""

    @property
    def passed(self) -> bool:
        return self.overall >= 3.0 and self.mention_miss == 0.0


JUDGE_SYSTEM = """\
你是一个严格的 Agent 评估专家。根据用户场景和评分维度，对 Agent 的回复打分。

打分原则：
- 每个维度独立打 0-5 分（0=完全没做到，5=非常出色）
- 严格但公平。如果回复抽象、套话、没有具体内容 → 不超过 3 分
- 如果回复明显与用户意图不符 → 不超过 2 分
- 输出必须是合法 JSON
"""


def _build_judge_prompt(case: EvalCase, agent_answer: str) -> str:
    history_text = ""
    if case.history:
        history_text = "对话历史：\n" + "\n".join(
            f"{m['role']}: {m['content']}" for m in case.history
        ) + "\n\n"

    criteria_text = "\n".join(
        f"  {i + 1}. {c}" for i, c in enumerate(case.criteria)
    )

    return f"""场景：{case.description}

{history_text}用户当前提问：{case.query}

Agent 回复：
{agent_answer}

请按以下维度打分（0-5）：
{criteria_text}

输出 JSON：
{{
  "scores": {{"维度1原文": 分数, "维度2原文": 分数, ...}},
  "overall": 综合分（0-5，可小数）,
  "rationale": "简短评语（30字内）"
}}
"""


class LLMJudge:
    def __init__(self, llm: LLMProvider) -> None:
        self.llm = llm

    def evaluate(self, case: EvalCase, agent_answer: str) -> JudgeScore:
        # must_mention / must_not_mention 用代码硬判，不交给 LLM
        hit = sum(1 for kw in case.must_mention if kw in agent_answer)
        mention_hit = hit / len(case.must_mention) if case.must_mention else 1.0
        violations = sum(1 for kw in case.must_not_mention if kw in agent_answer)

        score = JudgeScore(case_id=case.case_id, overall=0.0)
        score.mention_hit = round(mention_hit, 3)
        score.mention_miss = float(violations)

        # 走 LLM judge
        prompt = _build_judge_prompt(case, agent_answer)
        try:
            raw = self.llm.generate(prompt, system=JUDGE_SYSTEM, temperature=0.0)
            data = extract_json_dict(raw)
            if data:
                score.overall = float(data.get("overall", 0))
                scores_dict = data.get("scores", {})
                if isinstance(scores_dict, dict):
                    score.per_criterion = {k: float(v) for k, v in scores_dict.items()}
                score.rationale = str(data.get("rationale", ""))[:200]
        except Exception as exc:
            score.rationale = f"judge 失败: {exc}"

        # mention 违规直接拉低 overall
        if violations > 0:
            score.overall = min(score.overall, 1.5)
        # must_mention 没命中也扣分
        if case.must_mention and mention_hit < 1.0:
            score.overall = score.overall * (0.5 + 0.5 * mention_hit)

        return score
