"""Explicit write-once stages for the experience-dependent plasticity pilot."""

import argparse
import json
from pathlib import Path

from .cohorts import evaluate_cohorts, lock_liu_inputs
from .evidence import lock_artifacts, lock_source
from .generic_selection import evaluate_generic, lock_selection
from .qualification import qualify
from .reporting import publish, verify_record
from .training import train_all


def main(args=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(
            "qualify",
            "lock-source",
            "train",
            "lock-artifacts",
            "evaluate-generic",
            "lock-selection",
            "lock-liu-inputs",
            "evaluate-liu",
            "publish",
            "verify-record",
        ),
    )
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--qualification-directory", type=Path)
    parser.add_argument("--selection-directory", type=Path)
    parsed = parser.parse_args(args)
    if parsed.stage == "qualify":
        result = qualify(parsed.attempt)
    elif parsed.stage == "lock-source":
        if parsed.qualification_directory is None:
            parser.error("lock-source requires --qualification-directory")
        result = lock_source(parsed.qualification_directory)
    elif parsed.stage == "lock-selection":
        if parsed.selection_directory is None:
            parser.error("lock-selection requires --selection-directory")
        result = lock_selection(parsed.selection_directory)
    else:
        result = {
            "train": train_all,
            "lock-artifacts": lock_artifacts,
            "evaluate-generic": evaluate_generic,
            "lock-liu-inputs": lock_liu_inputs,
            "evaluate-liu": evaluate_cohorts,
            "publish": publish,
            "verify-record": verify_record,
        }[parsed.stage]()
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
