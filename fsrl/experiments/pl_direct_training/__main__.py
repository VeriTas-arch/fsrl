"""Lifecycle commands for direct P/L training."""

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
            "train-development",
            "lock-development",
            "evaluate-development",
            "report-development",
            "train-confirmation",
            "lock-confirmation",
            "evaluate-confirmation",
            "report-confirmation",
        ),
    )
    parsed = parser.parse_args(args)
    if parsed.stage == "qualify":
        from .qualification import run_qualification

        result = run_qualification()
    elif parsed.stage == "lock-source":
        from .locks import write_source_lock

        result = write_source_lock()
    elif parsed.stage.startswith("train-"):
        from .training import train_cohort

        result = train_cohort(parsed.stage.removeprefix("train-"))
    elif parsed.stage.startswith("lock-"):
        from .locks import write_artifact_lock

        result = write_artifact_lock(parsed.stage.removeprefix("lock-"))
    elif parsed.stage.startswith("evaluate-"):
        from .evaluation import evaluate_cohort

        result = evaluate_cohort(parsed.stage.removeprefix("evaluate-"))
    elif parsed.stage.startswith("report-"):
        from .reporting import write_report

        result = write_report(parsed.stage.removeprefix("report-"))
    else:
        raise ValueError(f"unknown lifecycle stage: {parsed.stage}")
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
