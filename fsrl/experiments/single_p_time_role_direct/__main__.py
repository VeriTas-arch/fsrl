"""Lifecycle commands for the two-phase direct-authority successor."""

from __future__ import annotations

import argparse
import json

from .runtime import configure_authority_runtime


def main(args=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(
            "qualify",
            "lock-baseline-source",
            "baseline",
            "lock-baseline",
            "mechanism",
            "report",
        ),
    )
    stage = parser.parse_args(args).stage
    runtime = configure_authority_runtime()
    if stage == "qualify":
        from .qualification import run_qualification

        result = run_qualification()
    elif stage == "lock-baseline-source":
        from .locks import write_source_lock

        result = write_source_lock(runtime)
    elif stage == "baseline":
        from .baseline import run

        result = run(runtime)
    elif stage == "lock-baseline":
        from .locks import write_baseline_artifact_lock

        result = write_baseline_artifact_lock()
    elif stage == "mechanism":
        from .mechanism import run

        result = run(runtime)
    else:
        from .reporting import write_report

        result = write_report()
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
