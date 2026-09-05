"""Prospective finite-capacity learner: explicitly admitted execution stages."""

import argparse
import json

from .qualification import qualify


def main(args=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("qualify",))
    parser.add_argument("--attempt", type=int, default=1)
    parsed = parser.parse_args(args)
    result = qualify(parsed.attempt)
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
