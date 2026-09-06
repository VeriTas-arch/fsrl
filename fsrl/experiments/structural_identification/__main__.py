"""Run the fixed, source-locked structural-identification study."""

import argparse
import importlib
import json

from fsrl.infra.formal_runtime import (
    configure_formal_cuda_runtime,
    configure_formal_runtime,
)

STAGES = {
    "qualify": ("execution", "qualify"),
    "lock-source": ("execution", "lock_source"),
    "measurement": ("measurement", "run_measurement"),
    "prepare-inputs": ("inputs", "prepare_inputs"),
    "train": ("execution", "train_all"),
    "recover": ("recovery", "run_recovery"),
    "generic": ("evaluation", "evaluate_generic"),
    "liu": ("evaluation", "evaluate_liu"),
    "manipulations": ("evaluation", "evaluate_manipulations"),
    "summarize": ("reporting", "summarize"),
    "publish": ("reporting", "publish"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=STAGES)
    args = parser.parse_args()
    if args.stage in {"qualify", "train", "recover", "generic", "liu", "manipulations"}:
        configure_formal_cuda_runtime()
    else:
        configure_formal_runtime()
    module, function = STAGES[args.stage]
    result = getattr(importlib.import_module(f"{__package__}.{module}"), function)()
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
