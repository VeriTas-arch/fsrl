"""Direct bounded entry point; no changes to frozen parent dispatch."""

import argparse
import json

from fsrl.infra.formal_runtime import configure_formal_runtime
from fsrl.infra.validation_session import validation_session


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=("qualify", "freeze", "train", "lock-models", "evaluate", "report"),
    )
    args = parser.parse_args()
    configure_formal_runtime()
    from .evaluation import evaluate, qualify
    from .execution import freeze, lock_models, train
    from .reporting import report

    with validation_session():
        result = {
            "qualify": qualify,
            "freeze": freeze,
            "train": train,
            "lock-models": lock_models,
            "evaluate": evaluate,
            "report": report,
        }[args.stage]()
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
