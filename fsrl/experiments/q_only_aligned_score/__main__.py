"""Registered command-line stages for the aligned q-only comparator."""

from __future__ import annotations

import argparse
import json


def main(args=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=(
            "qualify",
            "lock-source-inputs",
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
        from .qualification import qualify

        result = qualify()
    elif stage == "lock-source-inputs":
        from .locks import write_source_input_lock

        result = write_source_input_lock()
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
        from .reporting import report_generic

        result = report_generic()
    elif stage == "evaluate-liu":
        from .evaluation import evaluate_liu_all

        result = evaluate_liu_all()
    else:
        from .reporting import report_final

        result = report_final()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
