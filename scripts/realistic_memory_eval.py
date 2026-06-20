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
SCENARIO_PATH = ROOT / "evals" / "realistic_memory_dialogue.json"
ARTIFACT_DIR = ROOT / "artifacts"
RUNTIME_DIR = ARTIFACT_DIR / "realistic_memory_eval_runtime"
REPORT_JSON = ARTIFACT_DIR / "realistic_memory_eval_report.json"
REPORT_MD = ARTIFACT_DIR / "realistic_memory_eval_report.md"


def configure_environment() -> None:
    os.environ["LLM_API_KEY"] = ""
    os.environ["EXTERNAL_SOURCE"] = "mock"
    os.environ["ENABLE_EMBEDDINGS"] = "false"
    os.environ["ENABLE_ONLINE_ENRICH"] = "false"
    os.environ["LASTFM_API_KEY"] = ""
    os.environ["TAVILY_API_KEY"] = ""
    os.environ["STORE_ROOT"] = str(RUNTIME_DIR / "store")
    os.environ["MEDIA_ROOT"] = str(RUNTIME_DIR / "media")
    os.environ["RESOURCE_LIBRARY_PATH"] = str(RUNTIME_DIR / "resource_library.sqlite")


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


def main() -> int:
    configure_environment()
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    if RUNTIME_DIR.exists():
        shutil.rmtree(RUNTIME_DIR)
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    # 复用常规 smoke 的稳定假数据源；评测关注对话和记忆，而不是外部 API 波动。
    from scripts.long_dialogue_smoke import install_fakes

    install_fakes()
    from app.agent import AudioVisualAgent
    from app.storage import JsonStore

    scenario = json.loads(SCENARIO_PATH.read_text(encoding="utf-8"))
    agent = AudioVisualAgent(JsonStore(RUNTIME_DIR / "store"))
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
    REPORT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    REPORT_MD.write_text(render_report(scenario, results), encoding="utf-8")
    failed = [item for item in results if not item.ok]
    print(f"Realistic memory eval: {len(results) - len(failed)}/{len(results)} passed")
    print(f"Markdown report: {REPORT_MD}")
    print(f"JSON report: {REPORT_JSON}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
