from __future__ import annotations

import argparse
from pathlib import Path

from app.evals import run_golden_evals


def main() -> int:
    parser = argparse.ArgumentParser(description="Run SonicMind golden dialogue evaluations.")
    parser.add_argument("--dataset", default="evals/golden_dialogues.json")
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--live-llm", action="store_true", help="Use configured LLM instead of deterministic mode.")
    args = parser.parse_args()
    report = run_golden_evals(Path(args.dataset), deterministic=not args.live_llm)
    print(report.model_dump_json(indent=2))
    return 0 if report.score >= args.threshold else 1


if __name__ == "__main__":
    raise SystemExit(main())
