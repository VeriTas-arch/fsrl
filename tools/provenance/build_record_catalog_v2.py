"""Build or audit the non-destructive v2 historical record catalog."""

from __future__ import annotations

import argparse
import json
import sys

from fsrl.infra.record_catalog import (
    CATALOG_PATH,
    check_record_catalog,
    render_record_catalog,
)


def run(*, apply: bool) -> dict:
    if apply:
        CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        CATALOG_PATH.write_text(render_record_catalog(), encoding="utf-8")
    return check_record_catalog()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    result = run(apply=arguments.apply)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
