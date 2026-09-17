"""Lifecycle commands for the historical single-P morphology re-audit."""

from __future__ import annotations

import argparse
import json
from importlib import import_module

STAGES = {
    "qualify": ("qualification", "run_qualification"),
    "lock-inputs": ("locks", "write_source_input_lock"),
    "analyze": ("analysis", "analyze_all"),
    "report": ("reporting", "write_report"),
}


def main(args=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=tuple(STAGES))
    stage = parser.parse_args(args).stage
    module_name, function_name = STAGES[stage]
    function = getattr(
        import_module(
            f"fsrl.experiments.historical_single_p_morphology_reaudit.{module_name}"
        ),
        function_name,
    )
    print(json.dumps(function(), indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
