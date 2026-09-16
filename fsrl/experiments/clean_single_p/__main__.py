"""Lifecycle commands for clean single-P development."""

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
            "lock-models",
            "evaluate",
            "report",
        ),
    )
    stage = parser.parse_args(args).stage
    if stage == "qualify":
        from .qualification import run_qualification

        result = run_qualification()
    elif stage == "lock-source":
        from .locks import write_source_lock

        result = write_source_lock()
    elif stage == "train":
        from .training import train_all

        result = train_all()
    elif stage == "lock-models":
        from .locks import write_model_lock

        result = write_model_lock()
    elif stage == "evaluate":
        from .evaluation import evaluate_all

        result = evaluate_all()
    elif stage == "report":
        from .reporting import write_report

        result = write_report()
    else:
        raise ValueError(stage)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
