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
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from tests.eval.cases import EVAL_CASES, EvalCase
from tests.eval.metrics import aggregate, compute_metrics
from tests.offline_fakes import configure_offline_env, seed_random, start_offline_patches

BASELINE_PATH = Path(__file__).parent / "baseline.json"


def _seed_assets():
    from app.models import Asset

    return [
        Asset(asset_id="a_seed1", source_url="https://eval/1", title="夜的钢琴曲",
              duration_seconds=240, artist="石进", genre=["古典"], mood=["治愈", "宁静"],
              tempo_bpm=72, energy_level=0.3, status="analyzed"),
        Asset(asset_id="a_seed2", source_url="https://eval/2", title="海阔天空",
              duration_seconds=326, artist="Beyond", genre=["摇滚"], mood=["励志"],
              tempo_bpm=85, energy_level=0.8, status="analyzed"),
    ]


def _setup(agent: Any, case: EvalCase) -> None:
    for s in _seed_assets():
        agent.store.write_model("assets", s.asset_id, s)
    for action in case.setup_actions:
        if action.get("type") == "seed_similar_library":
            from app.models import Asset

            artists = [
                "Drake", "Future", "Molly Santana", "A$AP Rocky", "BROCKHAMPTON",
                "Don Toliver", "Fetty Wap", "Lil Uzi Vert", "Metro Boomin",
                "PARTYNEXTDOOR", "Playboi Carti", "SZA", "Travis Scott",
            ]
            for idx, artist in enumerate(artists, 1):
                asset = Asset(
                    asset_id=f"a_similar_{idx}",
                    source_url=f"https://eval/similar/{idx}",
                    title=f"{artist} Song",
                    duration_seconds=200,
                    artist=artist,
                    genre=["说唱", "R&B"],
                    mood=["律动", "放松"],
                    status="analyzed",
                )
                agent.store.write_model("assets", asset.asset_id, asset)
        elif action.get("type") == "seed_local_library":
            from app.models import Asset

            for idx in range(1, 4):
                asset = Asset(
                    asset_id=f"a_local_seed_{idx}",
                    source_url=f"https://eval/local/{idx}",
                    title=f"Local Seed {idx}",
                    duration_seconds=200,
                    artist="Local Seed Artist",
                    genre=["R&B"],
                    mood=["放松"],
                    status="analyzed",
                )
                agent.store.write_model("assets", asset.asset_id, asset)
        elif action.get("type") == "seed_mixed_language_local_library":
            from app.models import Asset

            for idx, title in enumerate(("七里香", "晴天"), 1):
                asset = Asset(
                    asset_id=f"a_local_zh_{idx}",
                    source_url=f"https://eval/local/zh/{idx}",
                    title=title,
                    duration_seconds=200,
                    artist="周杰伦",
                    genre=["流行"],
                    mood=["放松"],
                    status="analyzed",
                )
                agent.store.write_model("assets", asset.asset_id, asset)
    agent._invalidate_assets_cache()
    agent.library.sync_assets(agent.list_assets())
    for action in case.setup_actions:
        if action.get("type") == "listen":
            agent.record_listen(case.user_id, action["asset_id"],
                                duration=action.get("duration", 100),
                                completed=action.get("completed", True))
        elif action.get("type") == "rate":
            agent.rate_asset(case.user_id, action["asset_id"], action["score"])
        elif action.get("type") == "chat":
            asyncio.run(agent.chat_async(case.user_id, action["query"]))


def _run_case(case: EvalCase, judge: Any | None) -> tuple[Any, dict[str, Any]]:
    from app.agent import AudioVisualAgent
    from app.storage import JsonStore

    agent = AudioVisualAgent(JsonStore(Path(tempfile.mkdtemp())))
    _setup(agent, case)
    history = [{"role": m["role"], "content": m["content"]} for m in case.history] or None
    answer = asyncio.run(agent.chat_async(case.user_id, case.query, history=history))
    relevance = judge.evaluate(case, answer.answer).overall if judge else None
    return answer, compute_metrics(case, answer, relevance)


def collect(case_filter: str | None = None, *, with_llm_judge: bool = False) -> dict[str, Any]:
    """跑（过滤后的）cases，返回 {case_id: metrics} + '__aggregate__'。"""
    from app.config import settings

    judge = None
    if with_llm_judge and not settings.mock_mode:
        from app.llm.client import OpenAICompatibleLLM
        from tests.eval.judge import LLMJudge

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
              f"anti_halluc={metrics['anti_halluc_pass']} local={metrics['local_ratio']} mm_hit={metrics['must_mention_hit']} "
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
    print(f"{'case':<22} {'intent':<16} {'anti_halluc':<14} {'junk':<8} {'local':<8} {'mm_hit':<10} {'diversity':<12} {'relevance'}")
    print("-" * 82)
    for cid, m in current.items():
        if cid.startswith("__"):
            continue
        b = baseline.get(cid, {})
        print(
            f"{cid:<22} "
            f"{(_fmt(m['intent_hit']) + _delta(m['intent_hit'], b.get('intent_hit'))):<16} "
            f"{_fmt(m['anti_halluc_pass']):<14} "
            f"{(_fmt(m.get('junk_rate')) + _delta(m.get('junk_rate'), b.get('junk_rate'))):<8} "
            f"{(_fmt(m.get('local_ratio')) + _delta(m.get('local_ratio'), b.get('local_ratio'))):<8} "
            f"{(_fmt(m['must_mention_hit']) + _delta(m['must_mention_hit'], b.get('must_mention_hit'))):<10} "
            f"{(_fmt(m['diversity']) + _delta(m['diversity'], b.get('diversity'))):<12} "
            f"{_fmt(m['relevance']) + _delta(m['relevance'], b.get('relevance'))}"
        )
    print("-" * 82)
    print("AGGREGATE (current vs baseline):")
    for k in ("intent_hit_rate", "anti_halluc_rate", "avg_junk_rate", "avg_local_ratio", "local_ratio_pass_rate", "avg_must_mention_hit", "avg_diversity", "avg_relevance"):
        print(f"  {k:<24} {_fmt(agg_cur.get(k))}{_delta(agg_cur.get(k), agg_base.get(k))}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Eval 回归护栏")
    parser.add_argument("--case", help="只跑某个 case_id")
    parser.add_argument("--update-baseline", action="store_true",
                        help="用当前结果覆写 baseline.json")
    parser.add_argument("--online", action="store_true",
                        help="允许真实外部源/embedding（默认关闭，保证离线确定）")
    parser.add_argument("--with-llm-judge", action="store_true",
                        help="有 LLM key 时启用 judge 相关性评分（默认关闭）")
    args = parser.parse_args()

    if not args.online:
        configure_offline_env()
    seed_random()

    from app.config import settings

    patches = nullcontext() if args.online else start_offline_patches()
    judge_mode = "LLM judge" if args.with_llm_judge and not settings.mock_mode else "确定性指标"
    source_mode = "online" if args.online else "offline"
    mode = f"{source_mode} / {judge_mode}"
    print(f"运行模式: {mode}\n")

    with patches:
        current = collect(args.case, with_llm_judge=args.with_llm_judge)

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
