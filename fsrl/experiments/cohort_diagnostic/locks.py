"""Joint implementation, three-fit and all-input pre-evaluation barrier."""

from fsrl.experiments.minimal_learner.locks import validate_complete
from fsrl.experiments.training_strategy.locks import (
    reference,
    require_pushed_clean,
    verify_reference,
)
from fsrl.infra.provenance import load_json, write_json_exclusive

from .inputs import load_cohorts, save_inputs
from .protocol import (
    RECORDS,
    implementation_sources,
    load_parameters,
    parameter_records,
    scientific_inputs,
    specification,
)

LOCK = RECORDS / "benchmarks/execution_lock.json"
QUALIFICATION = RECORDS / "benchmarks/qualification.json"


def validate_qualification(record: dict) -> None:
    if record["sources"] != implementation_sources() or record["passed"] is not True:
        raise RuntimeError("qualification source identity differs")
    if record["new_liu_evaluated"] or record["new_parameters_trained"]:
        raise RuntimeError("qualification crossed the diagnostic boundary")
    if set(record["numerical"]) != {str(seed) for seed in specification()["fits"]}:
        raise RuntimeError("qualification omits a frozen fit")
    if not all(
        row["passed"]
        and row["reference_parity"]
        and row["z_off"]
        and row["query_no_write"]
        for row in record["numerical"].values()
    ):
        raise RuntimeError("qualification misses numerical identities")
    if (
        len(record["parent_point_errors"]) != 9
        or max(record["parent_point_errors"].values()) > 1e-12
    ):
        raise RuntimeError("original behavioral point qualification differs")
    runtime = record["runtime"]
    expected = {
        "cuda_available": True,
        "torch_intraop_threads": 1,
        "torch_interop_threads": 1,
        "blas_thread_limit": 1,
        "compiler_threads": 1,
        "matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
    }
    if any(runtime[key] != value for key, value in expected.items()):
        raise RuntimeError("qualification runtime differs")
    profile = runtime["profile"]
    if not (
        profile["device"] == "cuda"
        and profile["compile"]
        and profile["compile_fullgraph"]
        and profile["compile_mode"] == "default"
        and profile["compile_backend"] == "inductor"
    ):
        raise RuntimeError("qualification compiler profile differs")


def lock_inputs(directory) -> dict:
    commit = require_pushed_clean()
    validate_complete(directory)
    record = load_json(directory / "qualification.json")
    validate_qualification(record)
    if record["source_commit"] != commit:
        raise RuntimeError("qualify the final committed implementation")
    for row in implementation_sources() + scientific_inputs():
        verify_reference(row, commit=commit)
    load_parameters()
    shards = save_inputs()
    write_json_exclusive(QUALIFICATION, record)
    result = {
        "source_commit": commit,
        "sources": implementation_sources(),
        "inputs": scientific_inputs(),
        "parameters": parameter_records(),
        "qualification": reference(QUALIFICATION),
        "cohort_shards": shards,
        "cohort_count": specification()["cohorts"]["count"],
        "models_evaluated": False,
    }
    write_json_exclusive(LOCK, result)
    return result


def validate_lock() -> dict:
    commit = require_pushed_clean()
    lock = load_json(verify_reference(reference(LOCK), commit=commit))
    if (
        lock["sources"] != implementation_sources()
        or lock["inputs"] != scientific_inputs()
        or lock["parameters"] != parameter_records()
    ):
        raise RuntimeError("diagnostic lock identity differs")
    for row in lock["sources"] + lock["inputs"]:
        verify_reference(row, commit=lock["source_commit"])
    validate_qualification(
        load_json(verify_reference(lock["qualification"], commit=commit))
    )
    if (
        lock["models_evaluated"]
        or lock["cohort_count"] != specification()["cohorts"]["count"]
    ):
        raise RuntimeError("input lock admission differs")
    size = specification()["cohorts"]["shard_size"]
    if len(lock["cohort_shards"]) * size != lock["cohort_count"]:
        raise RuntimeError("input lock omits mandatory cohorts")
    for offset, row in enumerate(lock["cohort_shards"]):
        verify_reference(row, commit=commit)
        load_cohorts(row, offset * size)
    return lock
