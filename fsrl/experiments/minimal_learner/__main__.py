"""Prospective stages never implicitly advance across an artifact lock."""

import argparse
import json

from .evaluation import evaluate_all
from .locks import lock_artifacts, lock_source, validate_artifacts
from .protocol import RUN_ROOT
from .reporting import write_report
from .smoke import smoke
from .training import train_all


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(
            "smoke",
            "lock-source",
            "train",
            "lock-artifacts",
            "evaluate",
            "report",
            "verify",
        ),
    )
    parser.add_argument("--attempt", type=int, default=1)
    args = parser.parse_args(argv)
    actions = {
        "smoke": lambda: smoke(args.attempt),
        "lock-source": lambda: lock_source(
            RUN_ROOT / "smoke" / f"attempt-{args.attempt}" / "smoke.json"
        ),
        "train": train_all,
        "lock-artifacts": lock_artifacts,
        "evaluate": evaluate_all,
        "report": write_report,
        "verify": validate_artifacts,
    }
    result = actions[args.stage]()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("passed", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
