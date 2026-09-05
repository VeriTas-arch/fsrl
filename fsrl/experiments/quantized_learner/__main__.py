"""Prospective finite-capacity learner: explicitly admitted execution stages."""

import argparse
import json
from pathlib import Path

from .evaluation import evaluate_all
from .evidence import lock_artifacts, lock_source
from .qualification import qualify
from .recovery_execution import execute_recovery, verify_recovery
from .reporting import publish, verify_record
from .training import train_all


def main(args=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(
            "qualify",
            "lock-source",
            "recover",
            "verify-recovery",
            "train",
            "lock-artifacts",
            "evaluate",
            "publish",
            "verify-record",
        ),
    )
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--qualification-directory", type=Path)
    parsed = parser.parse_args(args)
    if parsed.stage == "qualify":
        result = qualify(parsed.attempt)
    elif parsed.stage == "lock-source":
        if parsed.qualification_directory is None:
            parser.error("lock-source requires --qualification-directory")
        result = lock_source(parsed.qualification_directory)
    else:
        actions = {
            "recover": execute_recovery,
            "verify-recovery": verify_recovery,
            "train": train_all,
            "lock-artifacts": lock_artifacts,
            "evaluate": evaluate_all,
            "publish": publish,
            "verify-record": verify_record,
        }
        result = actions[parsed.stage]()
    print(json.dumps(result, indent=2))
    return 0 if result.get("passed", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
