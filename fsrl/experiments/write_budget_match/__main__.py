"""Budget calibration never calls a behavioral objective."""

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
            "calibrate",
            "lock-calibration",
            "evaluate",
            "budgets",
            "report",
        ),
    )
    args = parser.parse_args()
    configure_formal_runtime()
    from . import execution, qualification, reporting

    commands = {
        "qualify": qualification.qualify,
        "freeze": execution.freeze,
        "calibrate": execution.calibrate,
        "lock-calibration": execution.lock_calibration,
        "evaluate": execution.evaluate,
        "budgets": execution.evaluation_budgets,
        "report": reporting.report,
    }
    with validation_session():
        result = commands[args.stage]()
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
