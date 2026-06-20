from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.library_baseline import LibraryBaseline, build_library_baseline, compare_library_baselines


def main() -> int:
    parser = argparse.ArgumentParser(description="Create or compare a read-only SonicMind library baseline.")
    parser.add_argument("--output", help="Optional JSON file to save the current baseline.")
    parser.add_argument("--compare", help="Compare the current library with a saved baseline JSON.")
    args = parser.parse_args()
    current = build_library_baseline()
    if args.compare:
        previous = LibraryBaseline.model_validate_json(Path(args.compare).read_text(encoding="utf-8"))
        result = compare_library_baselines(previous, current)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["unchanged"] else 1
    payload = current.model_dump_json(indent=2)
    print(payload)
    if args.output:
        Path(args.output).write_text(payload + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
