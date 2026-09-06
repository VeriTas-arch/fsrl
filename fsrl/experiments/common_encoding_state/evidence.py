"""Prospective source, recovery, fit and input barriers for the pilot."""

from __future__ import annotations

import hashlib
import json

import torch

from fsrl.experiments.adaptive_plasticity.model import make_model
from fsrl.experiments.minimal_learner.locks import validate_complete
from fsrl.experiments.training_strategy.locks import (
    reference,
    require_pushed_clean,
    verify_reference,
)
from fsrl.infra.provenance import load_json, tensor_hashes, write_json_exclusive

from .inputs import GENERIC_MANIFEST, load_group, save_generic_inputs
from .protocol import (
    CONDITIONS,
    DESIGN_HASH,
    QUALIFICATION_SEED,
    RECORDS,
    SEEDS,
    resolved_specification,
    run_directory,
)
from .provenance import implementation_sources, scientific_inputs
from .recovery_execution import LOCK as RHO_LOCK
from .recovery_execution import summarize as summarize_recovery

SOURCE_LOCK = RECORDS / "benchmarks/source_lock.json"
QUALIFICATION = RECORDS / "benchmarks/qualification.json"
ARTIFACT_LOCK = RECORDS / "benchmarks/artifact_lock.json"


def validate_qualification(record: dict) -> None:
    expected = {
        "passed": True,
        "protocol_sha256": DESIGN_HASH,
        "seed": QUALIFICATION_SEED,
        "sources": implementation_sources(),
        "liu_evaluated": False,
        "parameters_trained": False,
    }
    if any(record.get(key) != value for key, value in expected.items()):
        raise RuntimeError("common-encoding qualification identity differs")
    if not record["checks"] or not all(
        row["passed"] is True for row in record["checks"].values()
    ):
        raise RuntimeError("common-encoding qualification is incomplete")
    runtime = record["runtime"]
    required = {
        "cuda_available": True,
        "torch_intraop_threads": 1,
        "torch_interop_threads": 1,
        "blas_thread_limit": 1,
        "compiler_threads": 1,
        "matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
    }
    profile = {
        "device": "cuda",
        "compile": True,
        "compile_fullgraph": True,
        "compile_backend": "inductor",
        "compile_mode": "default",
    }
    if any(runtime.get(key) != value for key, value in required.items()) or any(
        runtime["profile"].get(key) != value for key, value in profile.items()
    ):
        raise RuntimeError("qualification runtime differs from the frozen profile")


def lock_source(qualification_directory) -> dict:
    commit = require_pushed_clean()
    validate_complete(qualification_directory)
    record = load_json(qualification_directory / "qualification.json")
    validate_qualification(record)
    if record["source_commit"] != commit:
        raise RuntimeError("qualify the final committed common-encoding implementation")
    for row in implementation_sources() + scientific_inputs():
        verify_reference(row, commit=commit)
    generic = (
        save_generic_inputs()
        if not GENERIC_MANIFEST.exists()
        else load_json(GENERIC_MANIFEST)
    )
    write_json_exclusive(
        QUALIFICATION,
        {
            **record,
            "cpu_test_transcript": (qualification_directory / "tests.txt").read_text(),
        },
    )
    result = {
        "source_commit": commit,
        "protocol_sha256": DESIGN_HASH,
        "sources": implementation_sources(),
        "scientific_inputs": scientific_inputs(),
        "qualification": reference(QUALIFICATION),
        "generic_manifest": reference(GENERIC_MANIFEST),
        "generic_groups": generic["groups"],
        "recovery_performed": False,
        "training_performed": False,
        "liu_evaluated": False,
    }
    write_json_exclusive(SOURCE_LOCK, result)
    return result


def validate_source() -> dict:
    commit = require_pushed_clean()
    lock = load_json(verify_reference(reference(SOURCE_LOCK), commit=commit))
    if (
        lock["protocol_sha256"] != DESIGN_HASH
        or lock["sources"] != implementation_sources()
        or lock["scientific_inputs"] != scientific_inputs()
        or lock["recovery_performed"]
        or lock["training_performed"]
        or lock["liu_evaluated"]
    ):
        raise RuntimeError("common-encoding source lock differs")
    for row in lock["sources"] + lock["scientific_inputs"]:
        verify_reference(row, commit=lock["source_commit"])
    validate_qualification(
        load_json(verify_reference(lock["qualification"], commit=commit))
    )
    manifest = load_json(verify_reference(lock["generic_manifest"], commit=commit))
    if (
        manifest["groups"] != lock["generic_groups"]
        or manifest["model_rollout_performed"]
    ):
        raise RuntimeError("common-encoding generic input manifest differs")
    for row in lock["generic_groups"].values():
        verify_reference(row["arrays"], commit=commit)
        load_group(row)
    return lock


def validate_recovery() -> dict:
    source = validate_source()
    commit = require_pushed_clean()
    lock_reference = reference(RHO_LOCK)
    lock = load_json(verify_reference(lock_reference, commit=commit))
    result = load_json(verify_reference(lock["result"], commit=commit))
    summary = summarize_recovery(result["rows"])
    if (
        result["experiment_id"] != "common_encoding_state_v1"
        or result["source_commit"] != source["source_commit"]
        or result["qualification"] != source["qualification"]
        or result["summary"] != summary
        or not summary["passed"]
        or lock["selected_rho"] != summary["selected_rho"]
        or not lock["passed"]
        or lock["source_commit"] != source["source_commit"]
        or lock["liu_evaluated"]
        or lock["parameters_trained"]
    ):
        raise RuntimeError("common-encoding rho lock differs")
    return {**lock, "lock_reference": lock_reference, "result_record": result}


def validate_training(seed: int, condition: str) -> dict:
    recovery = validate_recovery()
    directory = run_directory(seed, condition)
    validate_complete(directory)
    config = load_json(directory / "config.json")
    settings = resolved_specification()["optimization"]
    expected = {
        "seed": seed,
        "condition": condition,
        "model_condition": "adaptive_eta_resampled",
        "scheduler": "global",
        "rho": recovery["selected_rho"],
        "rho_lock": recovery["lock_reference"],
        "protocol_sha256": DESIGN_HASH,
        "optimization": settings,
        "episodes": settings["total_episode_exposures"],
    }
    if any(config.get(key) != value for key, value in expected.items()):
        raise RuntimeError("common-encoding training identity differs")
    logs = [
        json.loads(line)
        for line in (directory / "train_log.jsonl").read_text().splitlines()
    ]
    if [row["step"] for row in logs] != list(range(settings["total_steps"])):
        raise RuntimeError("common-encoding training log omits a registered step")
    for channel in ("base", "latent", "encoded"):
        digest = hashlib.sha256()
        for row in logs:
            digest.update(bytes.fromhex(row[f"{channel}_batch_sha256"]))
            if row[f"{channel}_stream_sha256"] != digest.hexdigest():
                raise RuntimeError("common-encoding training hash chain differs")
        if config[f"{channel}_stream_sha256"] != digest.hexdigest():
            raise RuntimeError("final common-encoding stream hash differs")
    if set(config["optimizer_steps"]) != {"raw_eta", "raw_global_gain"} or set(
        config["optimizer_steps"].values()
    ) != {settings["total_steps"]}:
        raise RuntimeError("both slow scalars require every optimizer step")
    model = make_model(
        "adaptive_eta_resampled", resolved_specification(), scheduler="global"
    )
    model.load_state_dict(
        {
            key: torch.tensor(value, dtype=torch.float32)
            for key, value in config["raw_parameters"].items()
        }
    )
    if tensor_hashes(model) != config["final_parameters"]:
        raise RuntimeError("archived common-encoding parameters differ")
    return {"config": config, "logs": logs}


def _paired(runs: dict) -> None:
    expected = {f"{seed}/{condition}" for seed in SEEDS for condition in CONDITIONS}
    if set(runs) != expected:
        raise RuntimeError("all nine common-encoding fits are mandatory")
    for seed in SEEDS:
        rows = [runs[f"{seed}/{condition}"]["config"] for condition in CONDITIONS]
        for key in ("base_stream_sha256", "latent_stream_sha256", "initial_parameters"):
            if len({json.dumps(row[key], sort_keys=True) for row in rows}) != 1:
                raise RuntimeError(
                    "paired task, latent stream or initialization differs"
                )
        if len({row["encoded_stream_sha256"] for row in rows}) != len(CONDITIONS):
            raise RuntimeError("common encoder structures produced identical streams")


def lock_artifacts() -> dict:
    source = validate_source()
    recovery = validate_recovery()
    runs = {
        f"{seed}/{condition}": validate_training(seed, condition)
        for seed in SEEDS
        for condition in CONDITIONS
    }
    _paired(runs)
    destination = RECORDS / "results"
    destination.mkdir(parents=True, exist_ok=True)
    records = {}
    for identity, row in runs.items():
        path = destination / f"training-{identity.replace('/', '-')}.json"
        write_json_exclusive(path, row)
        records[identity] = reference(path)
    result = {
        "source_commit": source["source_commit"],
        "source_lock": reference(SOURCE_LOCK),
        "rho_lock": recovery["lock_reference"],
        "selected_rho": recovery["selected_rho"],
        "runs": records,
        "all_fits_locked_before_evaluation": True,
    }
    write_json_exclusive(ARTIFACT_LOCK, result)
    return result


def validate_artifacts() -> dict:
    source = validate_source()
    recovery = validate_recovery()
    commit = require_pushed_clean()
    lock = load_json(verify_reference(reference(ARTIFACT_LOCK), commit=commit))
    if (
        lock["source_lock"] != reference(SOURCE_LOCK)
        or lock["source_commit"] != source["source_commit"]
        or lock["rho_lock"] != recovery["lock_reference"]
        or lock["selected_rho"] != recovery["selected_rho"]
    ):
        raise RuntimeError("common-encoding artifact/source lock differs")
    runs = {
        identity: load_json(verify_reference(row, commit=commit))
        for identity, row in lock["runs"].items()
    }
    _paired(runs)
    for identity, row in runs.items():
        seed, condition = identity.split("/")
        if row != validate_training(int(seed), condition):
            raise RuntimeError("archived common-encoding training record differs")
    return {**lock, "archives": runs}
