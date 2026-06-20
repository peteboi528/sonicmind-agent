from __future__ import annotations

import json
from pathlib import Path

from scripts.realistic_memory_eval import SCENARIO_PATH, render_report


def test_realistic_scenario_has_multiple_sessions_and_memory_conflicts():
    scenario = json.loads(Path(SCENARIO_PATH).read_text(encoding="utf-8"))
    turns = scenario["turns"]
    sessions = {turn["session"] for turn in turns}
    expectations = [turn.get("expect", {}) for turn in turns]

    assert len(turns) >= 15
    assert len(sessions) >= 3
    assert any("positive_not_contains" in expected for expected in expectations)
    assert any("exclusion_not_contains" in expected for expected in expectations)
    assert any("dialogue_entities_not_contains" in expected for expected in expectations)


def test_realistic_report_calls_out_failed_product_gap():
    from scripts.realistic_memory_eval import Check, TurnResult

    scenario = {"name": "demo", "description": "demo"}
    results = [
        TurnResult(
            index=1,
            session="s1",
            message="这轮不要中文歌",
            ok=False,
            checks=[Check("exclusion_not_contains:中文歌", False, "actual=['中文歌']")],
            rationale="临时约束不应成为长期记忆。",
        )
    ]

    report = render_report(scenario, results)

    assert "Product gaps found" in report
    assert "exclusion_not_contains:中文歌" in report
    assert "临时约束不应成为长期记忆" in report
