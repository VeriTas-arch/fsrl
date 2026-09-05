"""Lock, execute, independently verify and publish the fixed circuit study."""

import argparse
import json

from fsrl.infra.provenance import load_json
from fsrl.infra.runtime import ExecutionProfile, configure_runtime
from fsrl.paths import REPO_ROOT

from .evidence import specification, validate_lock, write_lock
from .execution import execute
from .reporting import RESULT, publish
from .verification import verify_result, verify_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("lock", "evaluate", "verify", "publish", "verify-record")
    )
    stage = parser.parse_args().stage
    if stage == "lock":
        result = write_lock()
    elif stage == "evaluate":
        result = execute()
    else:
        configure_runtime(
            ExecutionProfile(device="cpu", compile=False, require_cuda=False)
        )
        validate_lock(pushed=False)
        directory = REPO_ROOT / specification()["numerics"]["run_directory"]
        if stage == "verify-record":
            result = verify_result(load_json(RESULT))
        elif stage == "publish":
            result = publish(directory)
        else:
            result = verify_run(directory)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
