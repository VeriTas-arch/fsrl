"""Registered command-line stages for the aligned q-only comparator."""

from __future__ import annotations

import argparse
import importlib
import json

STAGES = {
    "qualify": ("qualification", "qualify"),
    "qualify-repair1": ("qualification", "qualify_repair"),
    "qualify-repair2": ("qualification", "qualify_repair2"),
    "qualify-repair3": ("qualification", "qualify_repair3"),
    "lock-source-inputs": ("locks", "write_source_input_lock"),
    "lock-source-repair1": ("locks", "write_source_repair_lock"),
    "lock-source-repair2": ("locks", "write_source_repair2_lock"),
    "lock-source-repair3": ("locks", "write_source_repair3_lock"),
    "train": ("training", "train_all"),
    "lock-models": ("locks", "write_model_lock"),
    "evaluate-generic": ("evaluation", "evaluate_generic_all"),
    "report-generic": ("reporting", "report_generic"),
    "evaluate-liu": ("evaluation", "evaluate_liu_all"),
    "report": ("reporting", "report_final"),
}


def main(args=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=tuple(STAGES),
    )
    stage = parser.parse_args(args).stage
    module_name, function_name = STAGES[stage]
    module = importlib.import_module(f"{__package__}.{module_name}")
    result = getattr(module, function_name)()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
