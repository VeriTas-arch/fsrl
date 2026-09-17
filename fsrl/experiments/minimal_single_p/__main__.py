"""Lifecycle commands for the minimal single-P ladder."""

from __future__ import annotations

import argparse
import json


def main(args=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(
            "qualify",
            "lock-source",
            "train",
            "lock-models",
            "evaluate",
            "report",
        ),
    )
    parser.add_argument("level", nargs="?")
    parsed = parser.parse_args(args)
    if parsed.stage in {"train", "lock-models", "evaluate", "report"}:
        if parsed.level is None:
            parser.error(f"{parsed.stage} requires a registered level")
    elif parsed.level is not None:
        parser.error(f"{parsed.stage} does not accept a level")
    level = parsed.level

    if parsed.stage == "qualify":
        from .qualification import run_qualification

        result = run_qualification()
    elif parsed.stage == "lock-source":
        from .locks import write_source_lock

        result = write_source_lock()
    elif parsed.stage == "train":
        from .training import train_all

        assert level is not None
        result = train_all(level)
    elif parsed.stage == "lock-models":
        from .locks import write_model_lock

        assert level is not None
        result = write_model_lock(level)
    elif parsed.stage == "evaluate":
        from .evaluation import evaluate_all

        assert level is not None
        result = evaluate_all(level)
    elif parsed.stage == "report":
        from .reporting import write_report

        assert level is not None
        result = write_report(level)
    else:
        raise ValueError(parsed.stage)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
