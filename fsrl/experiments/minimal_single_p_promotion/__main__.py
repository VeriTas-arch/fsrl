"""Lifecycle commands for the M2 promotion solution-distribution study."""

from __future__ import annotations

import argparse
import json
from importlib import import_module

STAGES = {
    "qualify": ("qualification", "run_qualification"),
    "qualify-repair": ("qualification", "run_repair_qualification"),
    "lock-source": ("locks", "write_source_lock"),
    "lock-repair": ("locks", "write_source_repair_lock"),
    "train": ("training", "train_all"),
    "lock-models": ("locks", "write_model_lock"),
    "evaluate-generic": ("evaluation", "evaluate_generic_all"),
    "report-generic": ("reporting", "write_generic_report"),
    "evaluate-liu": ("evaluation", "evaluate_liu_all"),
    "report": ("reporting", "write_final_report"),
}


def main(args=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=tuple(STAGES),
    )
    stage = parser.parse_args(args).stage
    module_name, function_name = STAGES[stage]
    function = getattr(
        import_module(f"fsrl.experiments.minimal_single_p_promotion.{module_name}"),
        function_name,
    )
    result = function()
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
