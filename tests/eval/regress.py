"""Eval 回归护栏：跑 golden case，计算指标，对比 baseline，打印 before/after diff。

无 LLM key 也能跑（确定性子集：intent_hit / anti_halluc_pass / must_mention_hit）。
有 key 时 relevance 升级为 LLM judge 综合分。

每次改 prompt / 模型 / ranking 策略前后跑一遍，客观看到质量变化——
"能量化的 agent 才是工程，不能量化的只是 demo"。

Usage:
    python -m tests.eval.regress                    # 对比 baseline.json，打印 diff
    python -m tests.eval.regress --update-baseline  # 用当前结果覆写 baseline.json
    python -m tests.eval.regress --case recommend_basic
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from app.agent import AudioVisualAgent
from app.config import settings
from app.llm.client import OpenAICompatibleLLM
from app.models import AgentAnswer, Asset
from app.storage import JsonStore
from tests.eval.cases import EVAL_CASES, EvalCase
from tests.eval.judge import LLMJudge
from tests.eval.metrics import aggregate, compute_metrics

BASELINE_PATH = Path(__file__).parent / "baseline.json"

_SEED_ASSETS = [
    Asset(asset_id="a_seed1", source_url="https://eval/1", title="夜的钢琴曲",
          duration_seconds=240, artist="石进", genre=["古典"], mood=["治愈", "宁静"],
          tempo_bpm=72, energy_level=0.3, status="analyzed"),
    Asset(asset_id="a_seed2", source_url="https://eval/2", title="海阔天空",
          duration_seconds=326, artist="Beyond", genre=["摇滚"], mood=["励志"],
          tempo_bpm=85, energy_level=0.8, status="analyzed"),
]


def _setup(agent: AudioVisualAgent, case: EvalCase) -> None:
    for s in _SEED_ASSETS:
        agent.store.write_model("assets", s.asset_id, s)
    for action in case.setup_actions:
        if action.get("type") == "listen":
            agent.record_listen(case.user_id, action["asset_id"],
                                duration=action.get("duration", 100),
                                completed=action.get("completed", True))
        elif action.get("type") == "rate":
            agent.rate_asset(case.user_id, action["asset_id"], action["score"])


def _run_case(case: EvalCase, judge: LLMJudge | None) -> tuple[AgentAnswer, dict[str, Any]]:
    agent = AudioVisualAgent(JsonStore(Path(tempfile.mkdtemp())))
    _setup(agent, case)
    history = [{"role": m["role"], "content": m["content"]} for m in case.history] or None
    answer = asyncio.run(agent.chat_async(case.user_id, case.query, history=history))
    relevance = judge.evaluate(case, answer.answer).overall if judge else None
    return answer, compute_metrics(case, answer, relevance)


def collect(case_filter: str | None = None) -> dict[str, Any]:
    """跑（过滤后的）cases，返回 {case_id: metrics} + '__aggregate__'。"""
    judge = None
    if not settings.mock_mode:
        judge_llm = OpenAICompatibleLLM(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=os.getenv("JUDGE_MODEL", settings.llm_model),
        )
        judge = LLMJudge(judge_llm)
    cases = [c for c in EVAL_CASES if not case_filter or c.case_id == case_filter]
    per_case: dict[str, Any] = {}
    for i, case in enumerate(cases, 1):
        answer, metrics = _run_case(case, judge)
        per_case[case.case_id] = metrics
        mode = "LLM" if judge else "mock"
        print(f"  [{i}/{len(cases)}] {case.case_id}: intent={metrics['intent_hit']} "
              f"anti_halluc={metrics['anti_halluc_pass']} mm_hit={metrics['must_mention_hit']} "
              f"pv={metrics['prompt_signature'] or '-'} ({mode}, {len(answer.agent_trace)} steps)")
    per_case["__aggregate__"] = aggregate(per_case)
    return per_case


def _fmt(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "pass" if v else "FAIL"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def _delta(cur: Any, base: Any) -> str:
    if cur is None or base is None:
        return ""
    if isinstance(cur, bool) or isinstance(base, bool):
        return "" if cur == base else "  (changed!)"
    if isinstance(cur, (int, float)) and isinstance(base, (int, float)):
        d = cur - base
        return f"  ({d:+.3f})" if abs(d) > 1e-9 else ""
    return ""


def print_diff(current: dict[str, Any], baseline: dict[str, Any]) -> None:
    agg_cur = current.get("__aggregate__", {})
    agg_base = baseline.get("__aggregate__", {})
    print("\n" + "=" * 82)
    print(f"{'case':<22} {'intent':<16} {'anti_halluc':<14} {'mm_hit':<10} {'diversity':<12} {'relevance'}")
    print("-" * 82)
    for cid, m in current.items():
        if cid.startswith("__"):
            continue
        b = baseline.get(cid, {})
        print(
            f"{cid:<22} "
            f"{(_fmt(m['intent_hit']) + _delta(m['intent_hit'], b.get('intent_hit'))):<16} "
            f"{_fmt(m['anti_halluc_pass']):<14} "
            f"{(_fmt(m['must_mention_hit']) + _delta(m['must_mention_hit'], b.get('must_mention_hit'))):<10} "
            f"{(_fmt(m['diversity']) + _delta(m['diversity'], b.get('diversity'))):<12} "
            f"{_fmt(m['relevance']) + _delta(m['relevance'], b.get('relevance'))}"
        )
    print("-" * 82)
    print("AGGREGATE (current vs baseline):")
    for k in ("intent_hit_rate", "anti_halluc_rate", "avg_must_mention_hit", "avg_diversity", "avg_relevance"):
        print(f"  {k:<24} {_fmt(agg_cur.get(k))}{_delta(agg_cur.get(k), agg_base.get(k))}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Eval 回归护栏")
    parser.add_argument("--case", help="只跑某个 case_id")
    parser.add_argument("--update-baseline", action="store_true",
                        help="用当前结果覆写 baseline.json")
    args = parser.parse_args()

    mode = "mock（无 LLM key，仅确定性指标）" if settings.mock_mode else "LLM（含 judge 相关性）"
    print(f"运行模式: {mode}\n")

    current = collect(args.case)

    if args.update_baseline or not BASELINE_PATH.exists():
        BASELINE_PATH.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n✅ baseline 已写入 {BASELINE_PATH}")
        print("   （后续运行将与此对比；改完 prompt/模型/ranking 后重新 --update-baseline 快照）")
        print_diff(current, current)
    else:
        baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
        print_diff(current, baseline)


if __name__ == "__main__":
    main()
