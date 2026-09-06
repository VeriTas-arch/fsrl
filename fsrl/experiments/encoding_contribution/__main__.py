"""Run the frozen encoding-contribution diagnostic, one resumable stage at a time."""

import argparse
import importlib
import json

from fsrl.infra.formal_runtime import (
    configure_formal_cuda_runtime,
    configure_formal_runtime,
)

STAGES = {
    "qualify": ("execution", "qualify"),
    "lock-source": ("protocol", "lock_source"),
    "generic": ("execution", "generic"),
    "cross": ("execution", "cross"),
    "finite-cohort": ("statistics", "finite_cohort"),
    "manipulations": ("statistics", "manipulations"),
    "summarize": ("reporting", "summarize"),
    "publish": ("reporting", "publish"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=STAGES)
    args = parser.parse_args()
    if args.stage in ("qualify", "generic", "cross"):
        configure_formal_cuda_runtime()
    else:
        configure_formal_runtime()
    module, name = STAGES[args.stage]
    result = getattr(importlib.import_module(f"{__package__}.{module}"), name)()
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
