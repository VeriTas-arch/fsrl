"""Write-once stages for the fresh global-decay replication."""

import argparse
import json
from pathlib import Path

from .cohorts import evaluate_cohorts, lock_liu_inputs
from .evidence import lock_artifacts, lock_source
from .generic import evaluate_generic, lock_generic
from .qualification import qualify
from .reporting import publish, repair_report, verify_record
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
            "lock-generic",
            "lock-liu-inputs",
            "evaluate-liu",
            "publish",
            "repair-report",
            "verify-record",
        ),
    )
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--qualification-directory", type=Path)
    parser.add_argument("--generic-directory", type=Path)
    parsed = parser.parse_args(args)
    if parsed.stage == "qualify":
        result = qualify(parsed.attempt)
    elif parsed.stage == "lock-source":
        if parsed.qualification_directory is None:
            parser.error("lock-source requires --qualification-directory")
        result = lock_source(parsed.qualification_directory)
    elif parsed.stage == "lock-generic":
        if parsed.generic_directory is None:
            parser.error("lock-generic requires --generic-directory")
        result = lock_generic(parsed.generic_directory)
    else:
        result = {
            "train": train_all,
            "lock-artifacts": lock_artifacts,
            "evaluate-generic": evaluate_generic,
            "lock-liu-inputs": lock_liu_inputs,
            "evaluate-liu": evaluate_cohorts,
            "publish": publish,
            "repair-report": repair_report,
            "verify-record": verify_record,
        }[parsed.stage]()
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
