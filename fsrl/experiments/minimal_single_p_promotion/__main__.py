"""Lifecycle commands for the M2 promotion solution-distribution study."""

from __future__ import annotations

import argparse
import json


def main(args=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(
            "qualify",
            "qualify-repair",
            "lock-source",
            "lock-repair",
            "train",
            "lock-models",
            "evaluate-generic",
            "report-generic",
            "evaluate-liu",
            "report",
        ),
    )
    stage = parser.parse_args(args).stage
    if stage == "qualify":
        from .qualification import run_qualification

        result = run_qualification()
    elif stage == "qualify-repair":
        from .qualification import run_repair_qualification

        result = run_repair_qualification()
    elif stage == "lock-source":
        from .locks import write_source_lock

        result = write_source_lock()
    elif stage == "lock-repair":
        from .locks import write_source_repair_lock

        result = write_source_repair_lock()
    elif stage == "train":
        from .training import train_all

        result = train_all()
    elif stage == "lock-models":
        from .locks import write_model_lock

        result = write_model_lock()
    elif stage == "evaluate-generic":
        from .evaluation import evaluate_generic_all

        result = evaluate_generic_all()
    elif stage == "report-generic":
        from .reporting import write_generic_report

        result = write_generic_report()
    elif stage == "evaluate-liu":
        from .evaluation import evaluate_liu_all

        result = evaluate_liu_all()
    elif stage == "report":
        from .reporting import write_final_report

        result = write_final_report()
    else:
        raise ValueError(stage)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
