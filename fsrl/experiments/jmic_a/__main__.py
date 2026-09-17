"""Registered stages for the analytic JMIC-A study."""

from __future__ import annotations

import argparse
import importlib
import json

STAGES = {
    "qualify": ("qualification", "qualify"),
    "lock-source-inputs": ("locks", "write_source_input_lock"),
    "run": ("execution", "run"),
    "report": ("reporting", "report"),
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
