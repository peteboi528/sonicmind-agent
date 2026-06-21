from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EVALS_DIR = ROOT / "evals"
ARTIFACT_DIR = ROOT / "artifacts"


def scenario_paths(name: str) -> tuple[Path, Path, Path]:
    """按场景名解析 scenario JSON 与报告输出路径。默认 realistic_memory 保持兼容。"""
    candidates = [EVALS_DIR / f"{name}.json", EVALS_DIR / f"{name}_dialogue.json"]
    scenario = next((p for p in candidates if p.exists()), candidates[0])
    report_json = ARTIFACT_DIR / f"{name}_eval_report.json"
    report_md = ARTIFACT_DIR / f"{name}_eval_report.md"
    return scenario, report_json, report_md


# 向后兼容：旧代码/测试按名引用默认场景路径。
SCENARIO_PATH = scenario_paths("realistic_memory")[0]


def configure_environment(runtime_dir: Path) -> None:
    os.environ["LLM_API_KEY"] = ""
    os.environ["EXTERNAL_SOURCE"] = "mock"
    os.environ["ENABLE_EMBEDDINGS"] = "false"
    os.environ["ENABLE_ONLINE_ENRICH"] = "false"
    os.environ["LASTFM_API_KEY"] = ""
    os.environ["TAVILY_API_KEY"] = ""
    os.environ["STORE_ROOT"] = str(runtime_dir / "store")
    os.environ["MEDIA_ROOT"] = str(runtime_dir / "media")
    os.environ["RESOURCE_LIBRARY_PATH"] = str(runtime_dir / "resource_library.sqlite")


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""


@dataclass
class TurnResult:
    index: int
    session: str
    message: str
    ok: bool
    checks: list[Check] = field(default_factory=list)
    answer_preview: str = ""
    trace: list[str] = field(default_factory=list)
    rationale: str = ""
    error: str = ""


def _contains(items: list[str], needle: str) -> bool:
    lowered = needle.lower()
    return any(lowered in str(item).lower() for item in items)


def evaluate_turn(agent: Any, user_id: str, turn: dict[str, Any], answer: Any) -> list[Check]:
    expect = turn.get("expect") or {}
    memory = agent.memory.get_memory(user_id)
    dialogue = agent.memory.get_dialogue_state(user_id)
    positive = [entry.text for entry in memory.structured_preferences]
    exclusions = list(memory.exclusion_rules)
    episodes = [entry.text for entry in memory.episodic_memory]
    trace = list(getattr(answer, "agent_trace", []) or [])
    trace_text = "\n".join(trace)
    checks: list[Check] = []

    if expect.get("answer_non_empty"):
        text = (getattr(answer, "answer", "") or "").strip()
        checks.append(Check("answer_non_empty", bool(text), text[:100]))
    if "tracks_min" in expect:
        count = len(getattr(answer, "recommended_tracks", []) or [])
        checks.append(Check("tracks_min", count >= int(expect["tracks_min"]), f"actual={count}"))
    if "shown_tracks_min" in expect:
        count = len(dialogue.shown_tracks)
        checks.append(Check("shown_tracks_min", count >= int(expect["shown_tracks_min"]), f"actual={count}"))

    groups = [
        ("positive_contains", positive, True),
        ("positive_not_contains", positive, False),
        ("exclusion_contains", exclusions, True),
        ("exclusion_not_contains", exclusions, False),
        ("episode_contains", episodes, True),
        ("dialogue_entities_contains", dialogue.entities, True),
        ("dialogue_entities_not_contains", dialogue.entities, False),
    ]
    for key, values, wanted in groups:
        for needle in expect.get(key, []):
            found = _contains(values, needle)
            checks.append(Check(f"{key}:{needle}", found is wanted, f"actual={values[:12]}"))

    for needle in expect.get("trace_contains", []):
        checks.append(Check(f"trace_contains:{needle}", needle in trace_text, trace_text[-500:]))
    for needle in expect.get("trace_not_contains", []):
        checks.append(Check(f"trace_not_contains:{needle}", needle not in trace_text, trace_text[-500:]))
    return checks


def run_scenario(agent: Any, scenario: dict[str, Any]) -> list[TurnResult]:
    user_id = scenario["user_id"]
    history: list[dict[str, str]] = []
    current_session = ""
    results: list[TurnResult] = []
    for index, turn in enumerate(scenario["turns"], start=1):
        session = turn["session"]
        if session != current_session:
            history = []
            current_session = session
        try:
            answer = asyncio.run(agent.chat_async(user_id, turn["message"], history=history))
            checks = evaluate_turn(agent, user_id, turn, answer)
            results.append(TurnResult(
                index=index,
                session=session,
                message=turn["message"],
                ok=all(check.ok for check in checks),
                checks=checks,
                answer_preview=(getattr(answer, "answer", "") or "")[:240],
                trace=list(getattr(answer, "agent_trace", []) or []),
                rationale=turn.get("rationale", ""),
            ))
            history.extend([
                {"role": "user", "content": turn["message"]},
                {"role": "assistant", "content": getattr(answer, "answer", "") or ""},
            ])
        except Exception as exc:
            results.append(TurnResult(
                index=index,
                session=session,
                message=turn["message"],
                ok=False,
                error=repr(exc),
                rationale=turn.get("rationale", ""),
            ))
    return results


def render_report(scenario: dict[str, Any], results: list[TurnResult]) -> str:
    passed = sum(item.ok for item in results)
    lines = [
        f"# Realistic Memory Eval: {scenario['name']}",
        "",
        scenario.get("description", ""),
        "",
        f"- Turns: {len(results)}",
        f"- Passed: {passed}",
        f"- Failed: {len(results) - passed}",
        f"- Score: {passed / max(1, len(results)):.1%}",
        "",
        "## Turn results",
        "",
    ]
    previous_session = ""
    for item in results:
        if item.session != previous_session:
            lines.extend([f"### Session: {item.session}", ""])
            previous_session = item.session
        mark = "PASS" if item.ok else "FAIL"
        lines.append(f"#### {mark} Turn {item.index}: {item.message}")
        if item.rationale:
            lines.append(f"- Why it matters: {item.rationale}")
        if item.error:
            lines.append(f"- Error: `{item.error}`")
        for check in item.checks:
            cmark = "PASS" if check.ok else "FAIL"
            lines.append(f"- {cmark} `{check.name}`" + (f": {check.detail}" if check.detail else ""))
        if item.answer_preview:
            lines.append(f"- Answer preview: {item.answer_preview}")
        lines.append("")

    failed = [item for item in results if not item.ok]
    lines.extend(["## Product gaps found", ""])
    if not failed:
        lines.append("No gap was reproduced in this deterministic run.")
    else:
        for item in failed:
            failed_names = ", ".join(check.name for check in item.checks if not check.ok)
            lines.append(f"- Turn {item.index}: {failed_names or item.error}")
            if item.rationale:
                lines.append(f"  - Expected product behavior: {item.rationale}")
    lines.append("")
    return "\n".join(lines)


def _install_library(library: str) -> None:
    """按场景声明的库选择假数据源。默认 small=install_fakes；large_messy=大而杂压测库。"""
    if library == "large_messy":
        from scripts.large_messy_library import install_large_messy

        install_large_messy()
    else:
        from scripts.long_dialogue_smoke import install_fakes

        install_fakes()


def run_eval(scenario_name: str = "realistic_memory") -> int:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    scenario_path, report_json, report_md = scenario_paths(scenario_name)
    runtime_dir = ARTIFACT_DIR / f"{scenario_name}_eval_runtime"
    configure_environment(runtime_dir)
    if runtime_dir.exists():
        shutil.rmtree(runtime_dir)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    scenario = json.loads(scenario_path.read_text(encoding="utf-8"))
    _install_library(scenario.get("library", "small"))

    from app.agent import AudioVisualAgent
    from app.storage import JsonStore

    agent = AudioVisualAgent(JsonStore(runtime_dir / "store"))
    results = run_scenario(agent, scenario)
    payload = {
        "scenario": scenario["name"],
        "score": sum(item.ok for item in results) / max(1, len(results)),
        "results": [
            {
                **item.__dict__,
                "checks": [check.__dict__ for check in item.checks],
            }
            for item in results
        ],
    }
    report_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report_md.write_text(render_report(scenario, results), encoding="utf-8")
    failed = [item for item in results if not item.ok]
    print(f"{scenario_name} eval: {len(results) - len(failed)}/{len(results)} passed")
    print(f"Markdown report: {report_md}")
    print(f"JSON report: {report_json}")
    return 1 if failed else 0


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Realistic / stress memory eval runner")
    parser.add_argument("--scenario", default="realistic_memory", help="evals/ 下的场景名（不含 .json）")
    args = parser.parse_args()
    return run_eval(args.scenario)


if __name__ == "__main__":
    raise SystemExit(main())
