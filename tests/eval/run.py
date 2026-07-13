"""端到端 eval 入口（独立于 pytest，要消耗 token）。

Usage:
    # 需要真实 LLM_API_KEY（OPENAI_API_KEY 或兼容接口）
    python -m tests.eval.run

    # 只跑某个 case
    python -m tests.eval.run --case recommend_basic

    # 用不同的 judge 模型
    JUDGE_MODEL=gpt-4o python -m tests.eval.run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

# --offline 必须在 app.config 导入前配置 env：settings 是构造一次的单例，后设 env 不生效。
# offline_fakes 模块级无 app 导入，可安全前置导入。
if "--offline" in sys.argv:
    from tests.offline_fakes import configure_offline_env

    configure_offline_env(keep_llm=True)  # 伪造音乐源，保留真实 LLM key

from app.agent import AudioVisualAgent
from app.config import settings
from app.llm.client import OpenAICompatibleLLM
from app.models import Asset
from app.storage import JsonStore
from tests.eval.cases import EVAL_CASES
from tests.eval.judge import JudgeScore, LLMJudge


def _setup_agent(case, agent: AudioVisualAgent) -> None:
    """根据 case.setup_actions 准备初始状态。"""
    # 写入两条种子素材，让 setup_actions 里的 asset_id 可用
    seeds = [
        Asset(
            asset_id="a_seed1",
            source_url="https://eval/1",
            title="夜的钢琴曲",
            duration_seconds=240,
            artist="石进",
            genre=["古典"],
            mood=["治愈", "宁静"],
            tempo_bpm=72,
            energy_level=0.3,
            status="analyzed",
        ),
        Asset(
            asset_id="a_seed2",
            source_url="https://eval/2",
            title="海阔天空",
            duration_seconds=326,
            artist="Beyond",
            genre=["摇滚"],
            mood=["励志"],
            tempo_bpm=85,
            energy_level=0.8,
            status="analyzed",
        ),
    ]
    for s in seeds:
        agent.store.write_model("assets", s.asset_id, s)

    for action in case.setup_actions:
        kind = action.get("type")
        if kind == "listen":
            agent.record_listen(
                case.user_id,
                action["asset_id"],
                duration=action.get("duration", 100),
                completed=action.get("completed", True),
            )
        elif kind == "rate":
            agent.rate_asset(case.user_id, action["asset_id"], action["score"])


def run_eval(case_id: str | None = None, store_root: str = "/tmp/musicagent_eval") -> dict:
    if settings.mock_mode:
        print("⚠️  当前是 mock mode，eval 没有实际意义。请设置 LLM_API_KEY 后再跑。")
        sys.exit(1)

    cases = [c for c in EVAL_CASES if not case_id or c.case_id == case_id]
    if not cases:
        print(f"未找到 case: {case_id}")
        sys.exit(1)

    # judge 用一个独立的强模型（默认走相同配置；可通过 env 覆盖）
    judge_model = os.getenv("JUDGE_MODEL", settings.llm_model)
    judge_llm = OpenAICompatibleLLM(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=judge_model,
    )
    judge = LLMJudge(judge_llm)

    scores: list[JudgeScore] = []
    case_reports: list[dict] = []
    for i, case in enumerate(cases, 1):
        # 每个 case 用独立 store 避免相互污染
        case_store = Path(store_root) / case.case_id
        case_store.mkdir(parents=True, exist_ok=True)
        agent = AudioVisualAgent(JsonStore(case_store))
        _setup_agent(case, agent)

        print(f"\n[{i}/{len(cases)}] {case.case_id}: {case.description}")
        print(f"  query: {case.query!r}")

        t0 = time.time()
        history = [{"role": m["role"], "content": m["content"]} for m in case.history] or None
        answer = asyncio.run(agent.chat_async(case.user_id, case.query, history=history))
        latency = time.time() - t0

        print(f"  answer ({latency:.1f}s, {len(answer.agent_trace)} steps):")
        print(f"    {answer.answer[:200]}")

        score = judge.evaluate(case, answer.answer)
        scores.append(score)
        case_reports.append(
            {
                "case_id": case.case_id,
                "latency_seconds": round(latency, 3),
                "has_final_answer": bool(answer.answer.strip()),
                "overall": score.overall,
                "passed": score.passed,
                "mention_hit": score.mention_hit,
                "mention_miss": score.mention_miss,
                "per_criterion": score.per_criterion,
                "rationale": score.rationale,
            }
        )
        status = "✅" if score.passed else "❌"
        print(
            f"  {status} overall={score.overall:.2f}  mention_hit={score.mention_hit}  violations={score.mention_miss}"
        )
        if score.rationale:
            print(f"  judge: {score.rationale}")

    print("\n" + "=" * 60)
    print("汇总:")
    avg = sum(s.overall for s in scores) / len(scores) if scores else 0
    passed = sum(1 for s in scores if s.passed)
    print(f"  平均分: {avg:.2f}/5.0")
    print(f"  通过率: {passed}/{len(scores)}")
    passed = sum(item["passed"] for item in case_reports)
    violations = sum(item["mention_miss"] for item in case_reports)
    valid_finals = sum(item["has_final_answer"] for item in case_reports)
    return {
        "model": settings.llm_model,
        "judge_model": judge_model,
        "cases": case_reports,
        "summary": {
            "total_cases": len(case_reports),
            "passed_cases": passed,
            "pass_rate": round(passed / len(case_reports), 4) if case_reports else 0.0,
            "hard_violations": violations,
            "valid_final_answers": valid_finals,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", help="只跑某个 case_id")
    parser.add_argument("--store", default="/tmp/musicagent_eval", help="临时 store 根目录")
    parser.add_argument("--json-output", help="将机器可读评估报告写入该路径")
    parser.add_argument("--min-pass-rate", type=float, default=0.8, help="通过率门槛，默认 0.8")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="离线门禁：伪造音乐源（固定候选）+ 真实 LLM/Judge，确定性、无平台噪声",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="只报告、不按通过率失败（在线冒烟用，永不阻塞 CI）",
    )
    args = parser.parse_args()
    stack = None
    if args.offline:
        from tests.offline_fakes import start_offline_patches

        stack = start_offline_patches()
    try:
        report = run_eval(case_id=args.case, store_root=args.store)
    finally:
        if stack is not None:
            stack.close()
    if args.json_output:
        output = Path(args.json_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = report["summary"]
    ok = (
        summary["total_cases"] > 0
        and summary["valid_final_answers"] == summary["total_cases"]
        and summary["hard_violations"] == 0
        and summary["pass_rate"] >= args.min_pass_rate
    )
    if args.offline:
        print("\n[offline] 离线门禁：候选来自假源（固定夹具），无真实平台访问。")
    if not ok:
        if args.report_only:
            print("\n⚠️  未达标，但 --report-only：仅报告，不失败 CI。")
            return
        raise SystemExit(1)


if __name__ == "__main__":
    main()
