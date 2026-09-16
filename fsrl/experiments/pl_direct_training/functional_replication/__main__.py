"""Lifecycle commands for clean no-time P/L functional replication."""

from __future__ import annotations

import argparse
import json


def main(args=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(
            "qualify",
            "lock-source",
            "train",
            "lock-artifacts",
            "evaluate",
            "report",
        ),
    )
    parsed = parser.parse_args(args)
    if parsed.stage == "qualify":
        from .qualification import run_qualification

        result = run_qualification()
    elif parsed.stage == "lock-source":
        from .locks import write_source_lock

        result = write_source_lock()
    elif parsed.stage == "train":
        from .training import train_cohort

        result = train_cohort()
    elif parsed.stage == "lock-artifacts":
        from .locks import write_artifact_lock

        result = write_artifact_lock()
    elif parsed.stage == "evaluate":
        from .evaluation import evaluate_cohort

        result = evaluate_cohort()
    elif parsed.stage == "report":
        from .reporting import write_report

        result = write_report()
    else:
        raise ValueError(f"unknown lifecycle stage: {parsed.stage}")
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
