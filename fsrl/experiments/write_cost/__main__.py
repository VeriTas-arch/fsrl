"""Explicit prospective stages, with no outcome-dependent parameter rescue."""

import argparse
import json


def main(args=None):
    from . import evaluation, locks, preparation, qualification, reporting, training
    from .protocol import register

    stages = {
        "prepare": preparation.prepare,
        "qualify": qualification.run_qualification,
        "lock-source": locks.lock_source,
        "lock-scale": locks.lock_scale,
        "select": locks.select,
        "lock-models": locks.lock_models,
        "evaluate": evaluation.evaluate,
        "report": reporting.report,
        "register": register,
    }
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=[*stages, "develop-baseline", "develop-cost", "train-pairs"]
    )
    stage = parser.parse_args(args).stage
    result = stages[stage]() if stage in stages else training.train(stage)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
