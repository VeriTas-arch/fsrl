"""Single-thread frozen-array analysis with prospective write-once artifacts."""

from __future__ import annotations

import argparse
import json

from fsrl.experiments.training_strategy.evaluation import (
    flatten_arrays,
    json_ready,
    write_arrays,
)
from fsrl.experiments.training_strategy.locks import reference, require_pushed_clean
from fsrl.infra.provenance import write_json_exclusive
from fsrl.infra.run_manifest import ProspectiveRun
from fsrl.infra.runtime import ExecutionProfile, configure_runtime
from fsrl.paths import REPO_ROOT
from fsrl.tasks.protocol_catalog import load_registered_protocol

from .analysis import analyze_pair
from .estimands import direction
from .evidence import (
    PROTOCOL_COMMIT,
    PROTOCOL_HASH,
    specification,
    validate_lock,
    validate_parent,
    write_lock,
)
from .reporting import publish
from .verification import verify_run


def directions(fits: dict) -> dict:
    result = {}
    for domain, key in (
        ("global", "contrasts"),
        ("local", "effects"),
        ("between_recipe", "between_recipe"),
    ):
        source = "global" if domain == "global" else "local"
        first = next(iter(fits.values()))[source][key]
        result[domain] = {
            contrast: {
                endpoint: direction(
                    [row[source][key][contrast][endpoint] for row in fits.values()]
                )
                for endpoint in endpoints
            }
            for contrast, endpoints in first.items()
        }
    return result


def execute() -> dict:
    spec = specification()
    lock = validate_lock(pushed=True)
    commit = require_pushed_clean()
    original = validate_parent()
    runtime = configure_runtime(
        ExecutionProfile(device="cpu", compile=False, require_cuda=False)
    )
    if any(pool["num_threads"] != 1 for pool in runtime["blas_threadpools"]):
        raise RuntimeError("diagnostic BLAS must be single-threaded")
    directory = REPO_ROOT / spec["runtime"]["run_directory"]
    with ProspectiveRun.start(
        directory,
        workflow_id=spec["experiment_id"],
        execution_id="analysis-v1",
        producer={
            "module": __name__,
            "source_commit": lock["source_commit"],
            "execution_commit": commit,
        },
        resolved_config={"protocol_sha256": PROTOCOL_HASH, "runtime": runtime},
    ) as run:
        raw, fits = {}, {}
        protocol = load_registered_protocol("liu_v2")
        for seed in spec["seeds"]:
            raw[str(seed)], fits[str(seed)] = analyze_pair(
                seed, original, protocol, spec
            )
            print(json.dumps({"seed": seed, "analysis_complete": True}), flush=True)
        arrays_path = run.output_dir / "arrays.npz"
        write_arrays(arrays_path, flatten_arrays(raw))
        result = {
            "experiment_id": spec["experiment_id"],
            "protocol_sha256": PROTOCOL_HASH,
            "protocol_commit": PROTOCOL_COMMIT,
            "source_commit": lock["source_commit"],
            "execution_commit": commit,
            "arrays": reference(arrays_path),
            "runtime": runtime,
            "fits": fits,
            "directions": directions(fits),
            "outcome": "diagnostic_localization",
            "claim_boundary": spec["claim_boundary"],
            "stop_rule": spec["stop_rule"],
        }
        validate_lock(pushed=True)
        write_json_exclusive(run.output_dir / "result.json", json_ready(result))
    return {"run": str(directory), "complete": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("lock", "analyze", "verify", "report"))
    stage = parser.parse_args().stage
    spec = specification()
    directory = REPO_ROOT / spec["runtime"]["run_directory"]
    if stage == "lock":
        result = write_lock()
    elif stage == "analyze":
        result = execute()
    else:
        configure_runtime(
            ExecutionProfile(device="cpu", compile=False, require_cuda=False)
        )
        validate_lock(pushed=False)
        result = (
            verify_run(directory, spec) if stage == "verify" else publish(directory)
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
