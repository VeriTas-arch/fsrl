"""Prospective source and all-fit barriers."""

from __future__ import annotations

import hashlib
import json

import torch

from fsrl.experiments.minimal_learner.locks import validate_complete
from fsrl.experiments.training_strategy.locks import (
    reference,
    require_pushed_clean,
    verify_reference,
)
from fsrl.infra.provenance import load_json, tensor_hashes, write_json_exclusive

from .inputs import GENERIC_MANIFEST, load_group, save_generic_inputs
from .model import make_model
from .protocol import (
    CONDITIONS,
    DESIGN_HASH,
    RECORDS,
    SEEDS,
    resolved_specification,
    run_directory,
)
from .provenance import implementation_sources, scientific_inputs

ORIGINAL_SOURCE_LOCK = RECORDS / "benchmarks/source_lock.json"
SOURCE_LOCK = RECORDS / "benchmarks/source_lock_repair_1.json"
QUALIFICATION = RECORDS / "benchmarks/qualification_repair_1.json"
ARTIFACT_LOCK = RECORDS / "benchmarks/artifact_lock.json"


def validate_qualification(record: dict) -> None:
    expected = {
        "passed": True,
        "protocol_sha256": DESIGN_HASH,
        "seed": 1_090_001,
        "sources": implementation_sources(),
        "liu_evaluated": False,
        "parameters_trained": False,
    }
    if any(record.get(key) != value for key, value in expected.items()):
        raise RuntimeError("adaptive-plasticity qualification identity differs")
    if not record["checks"] or not all(
        row["passed"] is True for row in record["checks"].values()
    ):
        raise RuntimeError("adaptive-plasticity qualification is incomplete")
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
        raise RuntimeError("qualify the final committed implementation")
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
        "repair_of": reference(ORIGINAL_SOURCE_LOCK),
        "retained_artifact_lock": reference(ARTIFACT_LOCK),
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
        or lock["training_performed"]
        or lock["liu_evaluated"]
    ):
        raise RuntimeError("adaptive-plasticity source lock differs")
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
        raise RuntimeError("generic input manifest differs")
    for groups in lock["generic_groups"].values():
        for row in groups.values():
            verify_reference(row["arrays"], commit=commit)
            load_group(row)
    return lock


def validate_training(seed: int, condition: str) -> dict:
    directory = run_directory(seed, condition)
    validate_complete(directory)
    config = load_json(directory / "config.json")
    settings = resolved_specification()["optimization"]
    expected = {
        "seed": seed,
        "condition": condition,
        "protocol_sha256": DESIGN_HASH,
        "optimization": settings,
        "episodes": settings["total_episode_exposures"],
    }
    if any(config.get(key) != value for key, value in expected.items()):
        raise RuntimeError("adaptive-plasticity training identity differs")
    logs = [
        json.loads(line)
        for line in (directory / "train_log.jsonl").read_text().splitlines()
    ]
    if [row["step"] for row in logs] != list(range(settings["total_steps"])):
        raise RuntimeError("training log omits a registered step")
    for channel in ("base", "uniform", "encoded"):
        digest = hashlib.sha256()
        for row in logs:
            digest.update(bytes.fromhex(row[f"{channel}_batch_sha256"]))
            if row[f"{channel}_stream_sha256"] != digest.hexdigest():
                raise RuntimeError("training stream hash chain differs")
        if config[f"{channel}_stream_sha256"] != digest.hexdigest():
            raise RuntimeError("final training stream hash differs")
    if set(config["optimizer_steps"]) != {"raw_eta", "raw_global_gain"} or set(
        config["optimizer_steps"].values()
    ) != {settings["total_steps"]}:
        raise RuntimeError("both slow scalars require every optimizer step")
    model = make_model(condition, resolved_specification())
    model.load_state_dict(
        {
            key: torch.tensor(value, dtype=torch.float32)
            for key, value in config["raw_parameters"].items()
        }
    )
    if tensor_hashes(model) != config["final_parameters"]:
        raise RuntimeError("archived adaptive-plasticity parameters differ")
    return {"config": config, "logs": logs}


def _paired(runs: dict) -> None:
    expected = {f"{seed}/{condition}" for seed in SEEDS for condition in CONDITIONS}
    if set(runs) != expected:
        raise RuntimeError("all six adaptive-plasticity fits are mandatory")
    for seed in SEEDS:
        rows = [runs[f"{seed}/{condition}"]["config"] for condition in CONDITIONS]
        for key in (
            "base_stream_sha256",
            "uniform_stream_sha256",
            "encoded_stream_sha256",
            "initial_parameters",
        ):
            if rows[0][key] != rows[1][key]:
                raise RuntimeError("paired training streams or initialization differ")


def lock_artifacts() -> dict:
    source = validate_source()
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
        "runs": records,
        "all_fits_locked_before_evaluation": True,
    }
    write_json_exclusive(ARTIFACT_LOCK, result)
    return result


def validate_artifacts() -> dict:
    source = validate_source()
    commit = require_pushed_clean()
    lock = load_json(verify_reference(reference(ARTIFACT_LOCK), commit=commit))
    original_source = load_json(
        verify_reference(reference(ORIGINAL_SOURCE_LOCK), commit=commit)
    )
    if (
        lock["source_lock"] != reference(ORIGINAL_SOURCE_LOCK)
        or lock["source_commit"] != original_source["source_commit"]
        or source["repair_of"] != reference(ORIGINAL_SOURCE_LOCK)
        or source["retained_artifact_lock"] != reference(ARTIFACT_LOCK)
    ):
        raise RuntimeError("adaptive-plasticity artifact/source lock differs")
    runs = {
        identity: load_json(verify_reference(row, commit=commit))
        for identity, row in lock["runs"].items()
    }
    _paired(runs)
    for identity, row in runs.items():
        seed, condition = identity.split("/")
        expected = validate_training(int(seed), condition)
        if row != expected:
            raise RuntimeError("archived training record differs")
    return {**lock, "archives": runs}
