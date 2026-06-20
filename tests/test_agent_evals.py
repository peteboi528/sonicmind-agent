from __future__ import annotations

from pathlib import Path

from app.evals import load_golden_cases, run_golden_evals

DATASET = Path(__file__).parents[1] / "evals" / "golden_dialogues.json"


def test_golden_dialogue_dataset_is_valid_and_unique():
    cases = load_golden_cases(DATASET)
    assert len(cases) >= 8
    assert len({case.id for case in cases}) == len(cases)
    assert all(case.assertions for case in cases)


def test_golden_dialogue_eval_passes_deterministically():
    report = run_golden_evals(DATASET)
    assert report.score == 1.0, [failure for case in report.cases for failure in case.failures]
    assert report.passed is True
