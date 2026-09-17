"""Lifecycle commands for acute observation uncertainty in frozen M2."""

from __future__ import annotations

import argparse
import json
from importlib import import_module

STAGES = {
    "qualify": ("qualification", "run_qualification"),
    "lock-source-inputs": ("locks", "write_source_input_lock"),
    "evaluate": ("evaluation", "evaluate_all"),
    "report": ("reporting", "write_report"),
}


def main(args=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=tuple(STAGES))
    stage = parser.parse_args(args).stage
    module_name, function_name = STAGES[stage]
    function = getattr(
        import_module(
            f"fsrl.experiments.minimal_single_p_observation_uncertainty.{module_name}"
        ),
        function_name,
    )
    print(json.dumps(function(), indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
