"""Explicit execution stages; importing the package never runs an experiment."""

import argparse
import json

from fsrl.infra.formal_runtime import configure_formal_cuda_runtime
from fsrl.infra.validation_session import validation_session


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("freeze", "evaluate", "report"))
    args = parser.parse_args()
    configure_formal_cuda_runtime()
    from . import execution

    with validation_session():
        result = getattr(execution, args.stage)()
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
