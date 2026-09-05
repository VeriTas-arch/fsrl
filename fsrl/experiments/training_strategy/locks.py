"""Write-once source/artifact locks and pre-evaluation admission checks."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

from fsrl.infra.file_contracts import safe_relative_path, validate_run_manifest
from fsrl.infra.git_provenance import git_blob_sha256
from fsrl.infra.provenance import file_sha256, load_json, write_json_exclusive
from fsrl.infra.study_registry import resolve_record
from fsrl.paths import REPO_ROOT, RUNS_ROOT, STUDIES_ROOT
from fsrl.tasks.protocol_catalog import protocol_path

from .protocol import (
    PROTOCOL_COMMIT,
    PROTOCOL_PATH,
    PROTOCOL_SHA256,
    load_specification,
    phase_for_step,
    training_config,
)

RUN_ROOT = RUNS_ROOT / "joint_training_strategy_v1"
RECORD_ROOT = STUDIES_ROOT / "joint_training_strategy" / "records"
SOURCE_LOCK_PATH = (
    RECORD_ROOT / "benchmarks" / "joint_training_strategy_v1.execution_lock.json"
)
ARTIFACT_LOCK_PATH = (
    RECORD_ROOT / "benchmarks" / "joint_training_strategy_v1.artifact_lock.json"
)


def git_text(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def require_pushed_clean() -> str:
    if git_text("branch", "--show-current") != "dev":
        raise RuntimeError("the prospective workflow requires shared dev")
    if git_text("status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("scientific execution requires a clean committed worktree")
    commit = git_text("rev-parse", "HEAD")
    remote = git_text("ls-remote", "--exit-code", "origin", "refs/heads/dev").split()[0]
    if commit != remote:
        raise RuntimeError(
            "HEAD must be pushed to origin/dev before scientific execution"
        )
    return commit


def reference(path: Path) -> dict:
    return {
        "path": path.relative_to(REPO_ROOT).as_posix(),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
    }


def verify_reference(record: dict, *, commit: str | None = None) -> Path:
    path = REPO_ROOT / safe_relative_path(record["path"])
    if reference(path) != record:
        raise RuntimeError(f"locked file identity changed: {record['path']}")
    if (
        commit is not None
        and git_blob_sha256(REPO_ROOT, commit, record["path"]) != record["sha256"]
    ):
        raise RuntimeError(f"Git witness differs: {record['path']}")
    return path


def implementation_sources() -> list[dict]:
    paths = list((REPO_ROOT / "fsrl").rglob("*.py"))
    paths.extend(
        (REPO_ROOT / "tests" / "experiments" / "training_strategy").glob("*.py")
    )
    paths.extend([REPO_ROOT / "pyproject.toml", REPO_ROOT / ".envrc"])
    return [reference(path) for path in sorted(paths)]


def scientific_inputs() -> list[dict]:
    specification = load_specification()
    behavior = specification["decision_contract"]["behavior"]
    behavior_contract = REPO_ROOT / behavior["reference_contract"]
    human = load_json(behavior_contract)["registered_sources"]["human_benchmark"][
        "path"
    ]
    paths = {
        PROTOCOL_PATH,
        behavior_contract,
        REPO_ROOT / behavior["reference_result"],
        resolve_record(human),
        resolve_record("benchmarks/qualification_v2.json"),
        protocol_path("liu_v1"),
        protocol_path("liu_v2"),
    }
    return [reference(path) for path in sorted(paths)]


def _validate_smoke(smoke: dict) -> None:
    expected = {"passed": True, "seed": 910001, "liu_evaluated": False}
    if any(smoke.get(key) != value for key, value in expected.items()):
        raise RuntimeError("successful non-Liu seed-910001 CUDA smoke is required")
    runtime = smoke["runtime"]
    profile = {
        "device": "cuda",
        "compile": True,
        "compile_fullgraph": True,
        "compile_mode": "default",
        "compile_backend": "inductor",
        "cpu_threads": 1,
        "blas_threads": 1,
        "require_cuda": True,
    }
    effective = {
        "cuda_available": True,
        "compiler_threads": 1,
        "torch_intraop_threads": 1,
        "torch_interop_threads": 1,
        "blas_thread_limit": 1,
        "matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "float32_matmul_precision": "highest",
    }
    if runtime["profile"] != profile or any(
        runtime.get(key) != value for key, value in effective.items()
    ):
        raise RuntimeError("CUDA smoke used a different numerical execution profile")
    required = (
        {
            "P_T",
            "logits",
            "loss",
            "updated_raw_gain",
            "updated_backbone_h2DA.weight",
        }
        | {f"gradient_{i}" for i in range(4)}
        | {
            f"evaluation_{condition}_logits"
            for condition in ("intact", "local_off", "P_off", "query_shuffle")
        }
    )
    if not required.issubset(smoke["checks"]) or not all(
        row["passed"] is True for row in smoke["checks"].values()
    ):
        raise RuntimeError(
            "CUDA smoke lacks the required output/gradient/update checks"
        )
    if smoke["sources"] != implementation_sources():
        raise RuntimeError(
            "CUDA smoke did not exercise the complete current source set"
        )


def write_source_lock(smoke_path: Path) -> dict:
    specification = load_specification()
    commit = require_pushed_clean()
    smoke = load_json(smoke_path)
    _validate_smoke(smoke)
    if not validate_run_manifest(smoke_path.parent / "run.json")["passed"]:
        raise RuntimeError("CUDA smoke run manifest failed verification")
    for record in smoke["sources"]:
        verify_reference(record, commit=commit)
    sources = implementation_sources()
    for record in sources:
        verify_reference(record, commit=commit)
    protocol = reference(PROTOCOL_PATH)
    verify_reference(protocol, commit=PROTOCOL_COMMIT)
    inputs = scientific_inputs()
    for record in inputs:
        verify_reference(record, commit=commit)
    result = {
        "schema_version": 1,
        "experiment_id": specification["experiment_id"],
        "source_commit": commit,
        "protocol_commit": PROTOCOL_COMMIT,
        "protocol": protocol,
        "sources": sources,
        "scientific_inputs": inputs,
        "smoke": reference(smoke_path),
        "smoke_summary": {
            key: smoke[key] for key in ("passed", "seed", "runtime", "checks")
        },
    }
    write_json_exclusive(SOURCE_LOCK_PATH, result)
    return result


def validate_source_lock(*, require_pushed: bool = True) -> dict:
    load_specification()
    commit = require_pushed_clean() if require_pushed else git_text("rev-parse", "HEAD")
    lock = load_json(SOURCE_LOCK_PATH)
    verify_reference(reference(SOURCE_LOCK_PATH), commit=commit)
    if (
        lock["protocol"]["sha256"] != PROTOCOL_SHA256
        or lock["protocol_commit"] != PROTOCOL_COMMIT
    ):
        raise RuntimeError("source lock uses a different scientific contract")
    verify_reference(lock["protocol"], commit=PROTOCOL_COMMIT)
    if lock["sources"] != implementation_sources():
        raise RuntimeError("the complete implementation source set has changed")
    for record in lock["sources"]:
        verify_reference(record, commit=lock["source_commit"])
    if lock["scientific_inputs"] != scientific_inputs():
        raise RuntimeError("frozen scientific inputs changed after implementation lock")
    for record in lock["scientific_inputs"]:
        verify_reference(record, commit=lock["source_commit"])
    smoke = load_json(verify_reference(lock["smoke"]))
    _validate_smoke(smoke)
    if lock["smoke_summary"] != {
        key: smoke[key] for key in ("passed", "seed", "runtime", "checks")
    }:
        raise RuntimeError("source lock misstates the CUDA smoke results")
    return lock


def run_directory(seed: int, condition: str) -> Path:
    specification = load_specification()
    if (
        seed not in specification["seeds"]["mandatory"]
        or condition not in specification["seeds"]["conditions"]
    ):
        raise ValueError("unregistered seed or condition")
    return RUN_ROOT / f"seed-{seed}" / condition


def validate_training_run(directory: Path, specification: dict) -> dict:
    required = {"run.json", "config.json", "net.pth", "local.pth", "train_log.jsonl"}
    if not required.issubset(
        path.name for path in directory.iterdir() if path.is_file()
    ):
        raise RuntimeError("training run lacks required final artifacts")
    validation = validate_run_manifest(directory / "run.json")
    manifest = load_json(directory / "run.json")
    if not validation["passed"] or manifest["lifecycle_state"] != "complete":
        raise RuntimeError(f"training run is not complete and verified: {directory}")
    metadata = load_json(directory / "config.json")
    if metadata["protocol_sha256"] != PROTOCOL_SHA256:
        raise RuntimeError("training run uses a different protocol")
    _validate_training_identity(directory, metadata, specification)
    _validate_log(directory, metadata, specification)
    _validate_optimizer_counts(metadata, specification)
    if (
        metadata["condition"] == "matched_staged"
        and metadata["stage_boundary_backbone"] != metadata["final_backbone"]
    ):
        raise RuntimeError("staged backbone changed during local adaptation")
    if verify_reference(metadata["checkpoint"]) != directory / "net.pth":
        raise RuntimeError(
            "training checkpoint reference identifies a different artifact"
        )
    return metadata


def _validate_training_identity(
    directory: Path, metadata: dict, specification: dict
) -> None:
    seed, condition = metadata["seed"], metadata["condition"]
    if (
        seed not in specification["seeds"]["mandatory"]
        or condition not in specification["seeds"]["conditions"]
    ):
        raise RuntimeError("unregistered training identity")
    if directory.name != condition or directory.parent.name != f"seed-{seed}":
        raise RuntimeError("training metadata identifies a different artifact slot")
    if metadata["training"] != asdict(training_config(specification, seed)):
        raise RuntimeError("training configuration differs from the registered recipe")
    if (
        metadata["episode_exposures"]
        != specification["optimization"]["total_episode_exposures"]
    ):
        raise RuntimeError("training episode exposure budget differs")


def _validate_log(directory: Path, metadata: dict, specification: dict) -> None:
    with (directory / "train_log.jsonl").open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    expected = specification["optimization"]["total_steps"]
    if [row["step"] for row in rows] != list(range(expected)):
        raise RuntimeError("training run does not contain every registered update")
    digest = hashlib.sha256()
    for row in rows:
        if row["phase"] != phase_for_step(
            specification, metadata["condition"], row["step"]
        ):
            raise RuntimeError("training log has an unregistered update phase")
        digest.update(bytes.fromhex(row["batch_fingerprint"]))
        if row["stream_fingerprint"] != digest.hexdigest():
            raise RuntimeError("training stream fingerprint chain differs")
    if metadata["stream_fingerprint"] != digest.hexdigest():
        raise RuntimeError("final training stream fingerprint differs")


def _validate_optimizer_counts(metadata: dict, specification: dict) -> None:
    expected_counts = specification["optimization"][metadata["condition"]]
    for name in ("backbone_updates", "local_updates"):
        if metadata[name] != expected_counts[name]:
            raise RuntimeError(f"incorrect update budget: {name}")
    observed = metadata["optimizer_parameter_steps"]
    if observed["local.raw_gain"] != expected_counts["local_updates"]:
        raise RuntimeError("local Adam counter does not match the update budget")
    backbone_counts = {
        key: value for key, value in observed.items() if key.startswith("backbone.")
    }
    if backbone_counts["backbone.h2DA.weight"] != expected_counts["backbone_updates"]:
        raise RuntimeError("support plasticity did not receive every registered update")
    if any(
        value not in (0, expected_counts["backbone_updates"])
        for value in backbone_counts.values()
    ):
        raise RuntimeError("backbone Adam counters contain partial optimization")


def write_artifact_lock() -> dict:
    source = validate_source_lock()
    specification = load_specification()
    runs = {}
    for seed in specification["seeds"]["mandatory"]:
        paired = {}
        for condition in specification["seeds"]["conditions"]:
            directory = run_directory(seed, condition)
            metadata = validate_training_run(directory, specification)
            paired[condition] = metadata
            runs[f"{seed}/{condition}"] = {
                "metadata": metadata,
                "files": [
                    reference(path)
                    for path in sorted(directory.iterdir())
                    if path.is_file()
                ],
            }
        for key in ("initial_backbone", "initial_local", "stream_fingerprint"):
            if paired["matched_staged"][key] != paired["joint"][key]:
                raise RuntimeError(f"paired seed {seed} does not match on {key}")
    result = {
        "schema_version": 1,
        "protocol_sha256": PROTOCOL_SHA256,
        "source_lock": reference(SOURCE_LOCK_PATH),
        "source_commit": source["source_commit"],
        "creation_commit": git_text("rev-parse", "HEAD"),
        "runs": runs,
    }
    write_json_exclusive(ARTIFACT_LOCK_PATH, result)
    return result


def validate_artifact_lock() -> dict:
    validate_source_lock()
    lock = load_json(ARTIFACT_LOCK_PATH)
    verify_reference(
        reference(ARTIFACT_LOCK_PATH), commit=git_text("rev-parse", "HEAD")
    )
    verify_reference(lock["source_lock"])
    specification = load_specification()
    expected = {
        f"{seed}/{condition}"
        for seed in specification["seeds"]["mandatory"]
        for condition in specification["seeds"]["conditions"]
    }
    if set(lock["runs"]) != expected or lock["protocol_sha256"] != PROTOCOL_SHA256:
        raise RuntimeError("artifact lock must contain all six registered final runs")
    for name, run in lock["runs"].items():
        seed, condition = name.split("/")
        directory = run_directory(int(seed), condition)
        if run["metadata"] != validate_training_run(directory, specification):
            raise RuntimeError("locked training metadata differs from the verified run")
        files = [
            reference(path) for path in sorted(directory.iterdir()) if path.is_file()
        ]
        if run["files"] != files:
            raise RuntimeError("artifact lock does not cover the complete run file set")
        for record in run["files"]:
            verify_reference(record)
    return lock
