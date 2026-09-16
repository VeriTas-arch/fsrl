"""Lifecycle commands for the frozen single-P time-role diagnostic."""

from __future__ import annotations

import argparse
import json


def main(args=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("qualify", "lock-source", "run", "report"))
    stage = parser.parse_args(args).stage
    if stage == "qualify":
        from .qualification import run_qualification

        result = run_qualification()
    elif stage == "lock-source":
        from .locks import write_source_lock

        result = write_source_lock()
    elif stage == "run":
        from .execution import run

        result = run()
    else:
        from .reporting import write_report

        result = write_report()
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
