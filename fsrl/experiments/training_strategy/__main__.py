"""Explicit lifecycle commands for the prospective paired experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .locks import RUN_ROOT, write_artifact_lock, write_source_lock


def main(args=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(
            "smoke",
            "lock-implementation",
            "train-artifacts",
            "lock-artifacts",
            "evaluate",
            "report",
        ),
    )
    parser.add_argument(
        "--smoke-dir", type=Path, default=RUN_ROOT / "smoke" / "attempt-1"
    )
    parsed = parser.parse_args(args)
    if parsed.stage == "smoke":
        from .smoke import run_smoke

        result = run_smoke(parsed.smoke_dir)
    elif parsed.stage == "lock-implementation":
        result = write_source_lock(parsed.smoke_dir / "smoke.json")
    elif parsed.stage == "lock-artifacts":
        result = write_artifact_lock()
    elif parsed.stage == "evaluate":
        from .evaluation import evaluate_all

        result = evaluate_all()
    elif parsed.stage == "report":
        from .reporting import write_report

        result = write_report()
    else:
        from .training import train_all

        result = train_all()
    concise = {
        key: value
        for key, value in result.items()
        if key not in {"sources", "scientific_inputs", "runs"}
    }
    if "runs" in result:
        concise["registered_runs"] = sorted(result["runs"])
    print(json.dumps(concise, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
