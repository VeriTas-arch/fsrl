"""Bounded runtime entry point for Liu item-count transport."""

from __future__ import annotations

import sys

from .formal_runtime import configure_formal_runtime


def main(args=None) -> int:
    configure_formal_runtime()
    from .item_count_transport import main as workflow_main

    return workflow_main(list(sys.argv[1:] if args is None else args))


if __name__ == "__main__":
    raise SystemExit(main())
