"""Write-once stages for the common encoding state pilot."""

import argparse
import json
from pathlib import Path

from .cohorts import evaluate_cohorts, lock_liu_inputs
from .evidence import lock_artifacts, lock_source
from .generic import evaluate_generic, lock_generic
from .qualification import qualify
from .recovery_execution import evaluate_recovery, lock_recovery
from .reporting import publish, verify_record
from .training import train_all


def main(args=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(
            "qualify",
            "lock-source",
            "evaluate-recovery",
            "lock-recovery",
            "train",
            "lock-artifacts",
            "evaluate-generic",
            "lock-generic",
            "lock-liu-inputs",
            "evaluate-liu",
            "publish",
            "verify-record",
        ),
    )
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--qualification-directory", type=Path)
    parser.add_argument("--recovery-directory", type=Path)
    parser.add_argument("--generic-directory", type=Path)
    parsed = parser.parse_args(args)
    if parsed.stage == "qualify":
        result = qualify(parsed.attempt)
    elif parsed.stage == "lock-source":
        if parsed.qualification_directory is None:
            parser.error("lock-source requires --qualification-directory")
        result = lock_source(parsed.qualification_directory)
    elif parsed.stage == "lock-recovery":
        if parsed.recovery_directory is None:
            parser.error("lock-recovery requires --recovery-directory")
        result = lock_recovery(parsed.recovery_directory)
    elif parsed.stage == "lock-generic":
        if parsed.generic_directory is None:
            parser.error("lock-generic requires --generic-directory")
        result = lock_generic(parsed.generic_directory)
    else:
        result = {
            "evaluate-recovery": evaluate_recovery,
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
