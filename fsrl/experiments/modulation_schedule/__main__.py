"""Prospectively ordered diagnostic stages; no work occurs on import."""

import argparse
import json

from fsrl.infra.formal_runtime import configure_formal_runtime
from fsrl.infra.validation_session import validation_session


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=(
            "qualify",
            "freeze",
            "audit",
            "lock-calibration",
            "evaluate",
            "report",
            "compact-diagnose",
            "finalize",
        ),
    )
    parser.add_argument("--mode", choices=("phase", "constant"), default="phase")
    args = parser.parse_args()
    configure_formal_runtime()
    from . import (
        audit,
        compact_diagnostic,
        evaluation,
        execution,
        qualification,
        reporting,
    )

    commands = {
        "qualify": qualification.qualify,
        "freeze": execution.freeze,
        "audit": audit.run,
        "lock-calibration": execution.lock_calibration,
        "evaluate": lambda: evaluation.evaluate(args.mode),
        "report": lambda: reporting.report(args.mode),
        "compact-diagnose": compact_diagnostic.run,
        "finalize": reporting.finalize,
    }
    with validation_session():
        result = commands[args.stage]()
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
