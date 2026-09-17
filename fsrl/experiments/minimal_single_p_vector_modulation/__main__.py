"""Registered command-line stages for the M2 vector-modulation study."""

from __future__ import annotations

import argparse
import importlib
import json

STAGES = {
    "qualify": ("qualification", "qualify"),
    "lock-source-inputs": ("locks", "write_source_lock"),
    "train": ("training", "train_all"),
    "lock-models": ("locks", "write_model_lock"),
    "evaluate-generic": ("evaluation", "evaluate_generic_all"),
    "report-generic": ("reporting", "report_generic"),
    "evaluate-liu": ("evaluation", "evaluate_liu_all"),
    "report": ("reporting", "report_final"),
}


def main(args=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=tuple(STAGES))
    stage = parser.parse_args(args).stage
    module_name, function_name = STAGES[stage]
    module = importlib.import_module(f"{__package__}.{module_name}")
    result = getattr(module, function_name)()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
