"""Explicit stages for the frozen weak-evidence contribution study."""

import argparse
import json


def main(args=None):
    from . import execution, locks, protocol, qualification, reporting

    stages = {
        "qualify": qualification.run,
        "lock": locks.lock,
        "evaluate": execution.evaluate,
        "report": reporting.report,
        "register": protocol.register,
    }
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=stages)
    result = stages[parser.parse_args(args).stage]()
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
