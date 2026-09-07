"""Explicit bounded stages for the finite-state memory comparison."""

import argparse
import importlib
import json


def main(args=None):
    stages = {
        "qualify": ("qualification", "run_qualification"),
        "lock-source": ("locks", "lock_source"),
        "lock-development": ("locks", "lock_development"),
        "select": ("evaluation", "select"),
        "lock-models": ("locks", "lock_models"),
        "evaluate": ("reporting", "evaluate"),
        "report": ("reporting", "report"),
        "register": ("protocol", "register"),
    }
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=[*stages, "develop", "train-pairs"])
    stage = parser.parse_args(args).stage
    if stage in ("develop", "train-pairs"):
        from .training import train

        result = train(stage)
    else:
        module, function = stages[stage]
        result = getattr(importlib.import_module(f".{module}", __package__), function)()
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
