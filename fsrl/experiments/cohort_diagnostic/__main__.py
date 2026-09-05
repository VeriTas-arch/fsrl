"""Frozen-parameter cohort diagnostic: explicit write-once execution stages."""

import argparse
import json
from pathlib import Path

from .execution import evaluate
from .locks import lock_inputs
from .qualification import qualify
from .reporting import publish, verify_record


def main(args=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=("qualify", "lock-inputs", "evaluate", "publish", "verify-record"),
    )
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--qualification-directory", type=Path)
    parsed = parser.parse_args(args)
    if parsed.stage == "qualify":
        result = qualify(parsed.attempt)
    elif parsed.stage == "lock-inputs":
        if parsed.qualification_directory is None:
            parser.error("--qualification-directory is required")
        result = lock_inputs(parsed.qualification_directory)
    else:
        result = {
            "evaluate": evaluate,
            "publish": publish,
            "verify-record": verify_record,
        }[parsed.stage]()
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
