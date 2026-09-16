"""Lifecycle commands for exact P/L reparameterization."""

from __future__ import annotations

import argparse
import json


def main(args=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("lock-source", "qualify"))
    parsed = parser.parse_args(args)
    if parsed.stage == "lock-source":
        from .locks import write_source_lock

        result = write_source_lock()
    else:
        from .qualification import run_qualification

        result = run_qualification()
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
