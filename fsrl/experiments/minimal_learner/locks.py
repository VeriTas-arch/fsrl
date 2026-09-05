"""Write-once prospective source and all-model artifact admission."""

from __future__ import annotations

import hashlib
import json

from fsrl.experiments.training_strategy.locks import (
    git_text,
    reference,
    require_pushed_clean,
    verify_reference,
)
from fsrl.infra.file_contracts import validate_run_manifest
from fsrl.infra.provenance import load_json, write_json_exclusive
from fsrl.infra.study_registry import resolve_record
from fsrl.paths import REPO_ROOT
from fsrl.tasks.protocol_catalog import protocol_path

from .protocol import (
    PROTOCOL_COMMIT,
    PROTOCOL_PATH,
    PROTOCOL_SHA256,
    RECORD_ROOT,
    run_directory,
    specification,
)

SOURCE_LOCK = (
    RECORD_ROOT / "benchmarks" / "minimal_relational_learner_v1.source_lock.json"
)
ARTIFACT_LOCK = (
    RECORD_ROOT / "benchmarks" / "minimal_relational_learner_v1.artifact_lock.json"
)


def sources() -> list[dict]:
    paths = list((REPO_ROOT / "fsrl").rglob("*.py"))
    paths.extend((REPO_ROOT / "tests" / "experiments" / "minimal_learner").glob("*.py"))
    paths.extend([REPO_ROOT / "pyproject.toml", REPO_ROOT / ".envrc"])
    return [reference(path) for path in sorted(paths)]


def inputs() -> list[dict]:
    behavior = specification()["decision_contract"]["behavior"]
    contract = REPO_ROOT / behavior["reference_contract"]
    human = load_json(contract)["registered_sources"]["human_benchmark"]["path"]
    historical = (
        REPO_ROOT
        / "studies/joint_training_strategy/records/results/joint_training_strategy_v1.json"
    )
    paths = {
        PROTOCOL_PATH,
        contract,
        REPO_ROOT / behavior["reference_result"],
        resolve_record(human),
        historical,
        protocol_path("liu_v1"),
        protocol_path("liu_v2"),
    }
    return [reference(path) for path in sorted(paths)]


def validate_complete(directory) -> dict:
    manifest = load_json(directory / "run.json")
    if (
        manifest["lifecycle_state"] != "complete"
        or not validate_run_manifest(directory / "run.json")["passed"]
    ):
        raise RuntimeError(f"run is incomplete or modified: {directory}")
    return manifest


def lock_source(smoke_path) -> dict:
    commit = require_pushed_clean()
    smoke = load_json(smoke_path)
    validate_complete(smoke_path.parent)
    validate_smoke(smoke)
    for record in sources() + inputs():
        verify_reference(record, commit=commit)
    verify_reference(reference(PROTOCOL_PATH), commit=PROTOCOL_COMMIT)
    result = {
        "source_commit": commit,
        "protocol_commit": PROTOCOL_COMMIT,
        "protocol_sha256": PROTOCOL_SHA256,
        "sources": sources(),
        "inputs": inputs(),
        "smoke": reference(smoke_path),
    }
    write_json_exclusive(SOURCE_LOCK, result)
    return result


def validate_smoke(smoke: dict) -> None:
    if smoke["passed"] is not True or smoke["seed"] != 910101 or smoke["liu_evaluated"]:
        raise RuntimeError("source lock requires the registered non-Liu CUDA smoke")
    required = set()
    for condition in ("score_only", "score_trace"):
        names = ["raw_eta", "raw_global_gain"]
        if condition == "score_trace":
            names.append("local.raw_gain")
        required.update(
            f"{condition}/{kind}_{name}"
            for name in names
            for kind in ("gradient", "nonzero_gradient", "updated")
        )
        required.update(f"{condition}/output_{index}" for index in range(4))
        required.update(f"{condition}/query_{index}" for index in range(5))
        required.add(f"{condition}/loss")
    if (
        smoke["sources"] != sources()
        or not required.issubset(smoke["checks"])
        or not all(row["passed"] is True for row in smoke["checks"].values())
    ):
        raise RuntimeError(
            "smoke does not verify every required output/gradient/update"
        )
    runtime = smoke["runtime"]
    expected = {
        "compiler_threads": 1,
        "torch_intraop_threads": 1,
        "torch_interop_threads": 1,
        "blas_thread_limit": 1,
        "cuda_available": True,
        "matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "recompile_limit": 64,
    }
    profile = {
        "device": "cuda",
        "compile": True,
        "compile_fullgraph": True,
        "compile_backend": "inductor",
        "compile_mode": "default",
    }
    if any(runtime.get(key) != value for key, value in expected.items()) or any(
        runtime["profile"].get(key) != value for key, value in profile.items()
    ):
        raise RuntimeError("smoke used a different numerical execution profile")


def validate_source(*, pushed: bool = True) -> dict:
    commit = require_pushed_clean() if pushed else git_text("rev-parse", "HEAD")
    lock = load_json(SOURCE_LOCK)
    verify_reference(reference(SOURCE_LOCK), commit=commit)
    if lock["sources"] != sources() or lock["inputs"] != inputs():
        raise RuntimeError("locked source or scientific inputs changed")
    if (
        lock["protocol_sha256"] != PROTOCOL_SHA256
        or lock["protocol_commit"] != PROTOCOL_COMMIT
    ):
        raise RuntimeError("locked protocol identity differs")
    for record in lock["sources"] + lock["inputs"]:
        verify_reference(record, commit=lock["source_commit"])
    verify_reference(reference(PROTOCOL_PATH), commit=PROTOCOL_COMMIT)
    smoke_path = verify_reference(lock["smoke"])
    validate_complete(smoke_path.parent)
    validate_smoke(load_json(smoke_path))
    return lock


def validate_training(seed: int, condition: str) -> dict:
    directory = run_directory(seed, condition)
    validate_complete(directory)
    config = load_json(directory / "config.json")
    settings = specification()["optimization"]
    expected = {
        "seed": seed,
        "condition": condition,
        "protocol_sha256": PROTOCOL_SHA256,
        "optimization": settings,
        "episodes": settings["total_episode_exposures"],
    }
    if any(config.get(key) != value for key, value in expected.items()):
        raise RuntimeError("training identity, settings or exposure differs")
    logs = [
        json.loads(line)
        for line in (directory / "train_log.jsonl").read_text().splitlines()
    ]
    if [row["step"] for row in logs] != list(range(settings["total_steps"])):
        raise RuntimeError("training log omits or duplicates a registered step")
    digest = hashlib.sha256()
    for row in logs:
        digest.update(bytes.fromhex(row["batch_sha256"]))
        if row["stream_sha256"] != digest.hexdigest():
            raise RuntimeError("training stream hash chain changed")
    if digest.hexdigest() != config["stream_sha256"]:
        raise RuntimeError("training stream does not match final metadata")
    count = 3 if condition == "score_trace" else 2
    if len(config["optimizer_steps"]) != count or set(
        config["optimizer_steps"].values()
    ) != {settings["total_steps"]}:
        raise RuntimeError("actual scalar Adam counters differ from protocol")
    verify_reference(config["checkpoint"])
    return config


def lock_artifacts() -> dict:
    source = validate_source()
    runs = {}
    for seed in specification()["seeds"]["mandatory"]:
        for condition in specification()["seeds"]["conditions"]:
            directory = run_directory(seed, condition)
            runs[f"{seed}/{condition}"] = {
                "config": validate_training(seed, condition),
                "files": [
                    reference(p) for p in sorted(directory.iterdir()) if p.is_file()
                ],
            }
        first, second = (
            runs[f"{seed}/{name}"]["config"] for name in ("score_only", "score_trace")
        )
        if first["stream_sha256"] != second["stream_sha256"]:
            raise RuntimeError("paired training streams differ")
        for key in ("raw_eta", "raw_global_gain"):
            if first["initial_parameters"][key] != second["initial_parameters"][key]:
                raise RuntimeError("paired initial scalar tensors differ")
    result = {
        "source_commit": source["source_commit"],
        "source_lock": reference(SOURCE_LOCK),
        "creation_commit": require_pushed_clean(),
        "protocol_sha256": PROTOCOL_SHA256,
        "runs": runs,
    }
    write_json_exclusive(ARTIFACT_LOCK, result)
    return result


def validate_artifacts() -> dict:
    source = validate_source()
    lock = load_json(ARTIFACT_LOCK)
    verify_reference(reference(ARTIFACT_LOCK), commit=require_pushed_clean())
    verify_reference(lock["source_lock"])
    if lock["source_commit"] != source["source_commit"]:
        raise RuntimeError("artifact lock names a different implementation")
    expected = {
        f"{seed}/{condition}"
        for seed in specification()["seeds"]["mandatory"]
        for condition in specification()["seeds"]["conditions"]
    }
    if set(lock["runs"]) != expected or lock["protocol_sha256"] != PROTOCOL_SHA256:
        raise RuntimeError("artifact lock lacks the exact six final models")
    for identity, run in lock["runs"].items():
        seed, condition = identity.split("/")
        if run["config"] != validate_training(int(seed), condition):
            raise RuntimeError("locked training metadata changed")
        directory = run_directory(int(seed), condition)
        if run["files"] != [
            reference(p) for p in sorted(directory.iterdir()) if p.is_file()
        ]:
            raise RuntimeError("locked training file inventory changed")
    return lock
