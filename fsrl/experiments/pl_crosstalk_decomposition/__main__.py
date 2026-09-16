"""Lifecycle commands for exact P/L cross-talk decomposition."""

from __future__ import annotations

import argparse
import json


def main(args=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("qualify", "lock-source", "analyze", "freeze")
    )
    parsed = parser.parse_args(args)
    if parsed.stage == "qualify":
        from .qualification import run_qualification

        result = run_qualification()
    elif parsed.stage == "lock-source":
        from .locks import write_source_lock

        result = write_source_lock()
    elif parsed.stage == "analyze":
        from .analysis import run_analysis

        result = run_analysis()
    else:
        from .reporting import freeze_outputs

        result = freeze_outputs()
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
