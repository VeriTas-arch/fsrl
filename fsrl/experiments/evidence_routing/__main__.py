"""Explicit stages of the prospective evidence-routing experiment."""

import argparse
import json


def main(args=None):
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
    stage = parser.parse_args(args).stage
    if stage == "qualify":
        from .qualification import run_qualification

        result = run_qualification()
    elif stage == "lock-source":
        from .locks import write_source_lock

        lock = write_source_lock()
        result = {
            "source_commit": lock["source_commit"],
            "inputs": list(lock["inputs"]),
        }
    elif stage == "train":
        from .training import train_all

        result = train_all()
    elif stage == "lock-artifacts":
        from .locks import write_artifact_lock

        result = {"runs": list(write_artifact_lock()["runs"])}
    elif stage == "evaluate":
        from .evaluation import evaluate_all

        result = evaluate_all()
    else:
        from .reporting import write_report

        result = write_report()
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
