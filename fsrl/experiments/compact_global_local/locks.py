"""Local-Git source and runtime-artifact locks for the compact candidate."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from fsrl.infra.file_contracts import safe_relative_path, validate_run_manifest
from fsrl.infra.git_provenance import git_blob_sha256
from fsrl.infra.provenance import file_sha256, load_json, write_json_exclusive
from fsrl.infra.study_registry import resolve_record
from fsrl.paths import REPO_ROOT, RUNS_ROOT, STUDIES_ROOT

from .protocol import (
    PROTOCOL_COMMIT,
    PROTOCOL_PATH,
    PROTOCOL_SHA256,
    REPAIR_COMMIT,
    REPAIR_PATH,
    REPAIR_SHA256,
    load_specification,
    phase_for_step,
    registered_seeds,
)

RUN_ROOT = RUNS_ROOT / "compact_global_local_model_v1"
RECORD_ROOT = STUDIES_ROOT / "compact_global_local_model" / "records"
SOURCE_LOCK_PATH = (
    RECORD_ROOT / "benchmarks" / "compact_global_local_model_v1.execution_lock.json"
)


def git_text(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def require_clean_commit() -> str:
    if git_text("branch", "--show-current") != "dev":
        raise RuntimeError("the compact-model workflow requires dev")
    if git_text("status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("scientific execution requires a clean committed worktree")
    return git_text("rev-parse", "HEAD")


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
    if commit is not None:
        observed = git_blob_sha256(REPO_ROOT, commit, record["path"])
        if observed != record["sha256"]:
            raise RuntimeError(f"Git witness differs: {record['path']}")
    return path


def implementation_sources() -> list[dict]:
    tests = REPO_ROOT / "tests" / "experiments" / "compact_global_local"
    paths = list((REPO_ROOT / "fsrl").rglob("*.py")) + list(tests.glob("*.py"))
    paths.extend([REPO_ROOT / "pyproject.toml", REPO_ROOT / ".envrc"])
    return [reference(path) for path in sorted(set(paths))]


def scientific_inputs() -> list[dict]:
    specification = load_specification()
    behavior_contract = (
        REPO_ROOT / specification["parent_evidence"]["behavior_contract"]
    )
    human_record = load_json(behavior_contract)["registered_sources"][
        "human_benchmark"
    ]["path"]
    paths = {
        PROTOCOL_PATH,
        REPAIR_PATH,
        behavior_contract,
        REPO_ROOT / specification["parent_evidence"]["behavior_result"],
        resolve_record(human_record),
        REPO_ROOT
        / "studies"
        / "dual_evidence_access_confirmation"
        / "records"
        / "benchmarks"
        / "dual_evidence_access_confirmation_v2_4.json",
        REPO_ROOT
        / "studies"
        / "task_fidelity"
        / "records"
        / "benchmarks"
        / "liu_v2.json",
    }
    existing = [path for path in paths if path.is_file()]
    if len(existing) != len(paths):
        missing = sorted(path.as_posix() for path in paths if not path.is_file())
        raise RuntimeError(f"scientific inputs are missing: {missing}")
    return [reference(path) for path in sorted(existing)]


def qualification_path() -> Path:
    return RUN_ROOT / "qualification" / "qualification.json"


def _validate_qualification(record: dict) -> None:
    expected = {
        "passed": True,
        "seed": 920001,
        "liu_evaluated": False,
        "protocol_sha256": PROTOCOL_SHA256,
        "repair_sha256": REPAIR_SHA256,
    }
    if any(record.get(key) != value for key, value in expected.items()):
        raise RuntimeError("successful compact-model qualification is required")
    required = {
        "packed_local_equivalence",
        "margin_loss_equivalence",
        "write_timing",
        "query_write_discard",
        "gradient_path",
        "backbone_freeze",
        "cuda_eager_compile_output",
        "cuda_eager_compile_gradients",
        "cuda_eager_compile_update",
    }
    if set(record["checks"]) != required or not all(
        row["passed"] is True for row in record["checks"].values()
    ):
        raise RuntimeError("compact-model qualification checks are incomplete")
    if record["sources"] != implementation_sources():
        raise RuntimeError("qualification did not exercise the current sources")


def write_source_lock() -> dict:
    specification = load_specification()
    commit = require_clean_commit()
    qualification = load_json(qualification_path())
    _validate_qualification(qualification)
    sources = implementation_sources()
    for record in sources:
        verify_reference(record, commit=commit)
    inputs = scientific_inputs()
    for record in inputs:
        witness = (
            PROTOCOL_COMMIT
            if record["path"] == reference(PROTOCOL_PATH)["path"]
            else REPAIR_COMMIT
            if record["path"] == reference(REPAIR_PATH)["path"]
            else commit
        )
        verify_reference(record, commit=witness)
    result = {
        "schema_version": 1,
        "experiment_id": specification["experiment_id"],
        "source_commit": commit,
        "protocol": {**reference(PROTOCOL_PATH), "commit": PROTOCOL_COMMIT},
        "repair": {**reference(REPAIR_PATH), "commit": REPAIR_COMMIT},
        "sources": sources,
        "scientific_inputs": inputs,
        "qualification": reference(qualification_path()),
        "qualification_summary": {
            key: qualification[key] for key in ("passed", "seed", "runtime", "checks")
        },
        "remote_state": "not_required; local committed Git objects are the execution witness",
    }
    SOURCE_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_json_exclusive(SOURCE_LOCK_PATH, result)
    return result


def validate_source_lock(*, require_clean: bool = True) -> dict:
    commit = require_clean_commit() if require_clean else git_text("rev-parse", "HEAD")
    lock = load_json(SOURCE_LOCK_PATH)
    if lock["protocol"]["sha256"] != PROTOCOL_SHA256:
        raise RuntimeError("source lock uses a different parent contract")
    if lock["repair"]["sha256"] != REPAIR_SHA256:
        raise RuntimeError("source lock uses a different repair contract")
    for record, witness in (
        (lock["protocol"], PROTOCOL_COMMIT),
        (lock["repair"], REPAIR_COMMIT),
    ):
        verify_reference(
            {key: record[key] for key in ("path", "sha256", "bytes")},
            commit=witness,
        )
    if lock["sources"] != implementation_sources():
        raise RuntimeError("implementation changed after the source lock")
    for record in lock["sources"]:
        verify_reference(record, commit=lock["source_commit"])
    if lock["scientific_inputs"] != scientific_inputs():
        raise RuntimeError("scientific inputs changed after the source lock")
    qualification = load_json(verify_reference(lock["qualification"]))
    _validate_qualification(qualification)
    if lock["qualification_summary"] != {
        key: qualification[key] for key in ("passed", "seed", "runtime", "checks")
    }:
        raise RuntimeError("source lock misstates qualification")
    if require_clean and commit != git_text("rev-parse", "HEAD"):
        raise RuntimeError("worktree moved during source validation")
    return lock


def run_directory(seed: int, cohort: str) -> Path:
    specification = load_specification()
    if seed not in registered_seeds(specification, cohort):
        raise ValueError("unregistered seed or cohort")
    return RUN_ROOT / cohort / f"seed-{seed}"


def _validate_training_identity(
    directory: Path, metadata: dict, specification: dict
) -> None:
    if metadata["protocol_sha256"] != PROTOCOL_SHA256:
        raise RuntimeError("training uses a different parent contract")
    if metadata["repair_sha256"] != REPAIR_SHA256:
        raise RuntimeError("training uses a different repair contract")
    seed = int(directory.name.removeprefix("seed-"))
    cohort = directory.parent.name
    if metadata["seed"] != seed or metadata["cohort"] != cohort:
        raise RuntimeError("training metadata identifies a different run slot")
    if seed not in registered_seeds(specification, cohort):
        raise RuntimeError("training metadata identifies an unregistered seed")
    architecture = metadata["architecture"]
    hidden_size = specification["architecture"]["hidden_size"]
    input_size = specification["architecture"]["input_size"]
    expected_parameters = (
        2 * hidden_size**2 + hidden_size * input_size + 3 * hidden_size + 5
    )
    if architecture != {
        "input_size": input_size,
        "support_steps": specification["architecture"]["support_trial_steps"],
        "query_steps": specification["architecture"]["query_steps"],
        "P_scalars": specification["architecture"]["persistent_state_sizes"]["P"],
        "L_scalars": specification["architecture"]["persistent_state_sizes"]["L"],
        "parameters": expected_parameters,
    }:
        raise RuntimeError(
            "training architecture differs from the registered candidate"
        )
    if (
        metadata["episode_exposures"]
        != specification["optimization"]["total_episode_exposures"]
    ):
        raise RuntimeError("training exposure budget differs")


def _validate_training_log(
    directory: Path, metadata: dict, specification: dict
) -> None:
    steps = specification["optimization"]["total_steps"]
    with (directory / "train_log.jsonl").open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    if [row["step"] for row in rows] != list(range(steps)):
        raise RuntimeError("training log does not contain every registered update")
    digest = hashlib.sha256()
    for row in rows:
        if row["phase"] != phase_for_step(specification, row["step"]):
            raise RuntimeError("training phase differs from the contract")
        digest.update(bytes.fromhex(row["batch_fingerprint"]))
        if row["stream_fingerprint"] != digest.hexdigest():
            raise RuntimeError("training stream hash chain differs")
    if metadata["stream_fingerprint"] != digest.hexdigest():
        raise RuntimeError("final training stream hash differs")


def _validate_optimizer_counts(metadata: dict, specification: dict) -> None:
    if metadata["stage_boundary_backbone"] != metadata["final_backbone"]:
        raise RuntimeError("backbone changed during local adaptation")
    expected_backbone = specification["optimization"]["global_only_steps"]
    expected_local = specification["optimization"]["total_steps"] - expected_backbone
    counters = metadata["optimizer_parameter_steps"]
    backbone_counters = {
        name: value for name, value in counters.items() if name.startswith("backbone.")
    }
    if not backbone_counters or set(backbone_counters.values()) != {expected_backbone}:
        raise RuntimeError("backbone optimizer counters differ from the budget")
    if counters.get("local.raw_gain") != expected_local:
        raise RuntimeError("local optimizer counter differs from the budget")


def validate_training_run(directory: Path, specification: dict) -> dict:
    required = {"run.json", "config.json", "model.pth", "train_log.jsonl"}
    if not required.issubset(
        path.name for path in directory.iterdir() if path.is_file()
    ):
        raise RuntimeError("training run lacks required final artifacts")
    manifest = load_json(directory / "run.json")
    if (
        manifest["lifecycle_state"] != "complete"
        or not validate_run_manifest(directory / "run.json")["passed"]
    ):
        raise RuntimeError("training run is incomplete or modified")
    metadata = load_json(directory / "config.json")
    _validate_training_identity(directory, metadata, specification)
    _validate_training_log(directory, metadata, specification)
    _validate_optimizer_counts(metadata, specification)
    if verify_reference(metadata["checkpoint"]) != directory / "model.pth":
        raise RuntimeError("checkpoint identity differs")
    return metadata


def artifact_lock_path(cohort: str) -> Path:
    if cohort not in {"development", "confirmation"}:
        raise ValueError(f"unknown cohort: {cohort}")
    return (
        RECORD_ROOT
        / "benchmarks"
        / f"compact_global_local_model_v1.{cohort}_artifact_lock.json"
    )


def write_artifact_lock(cohort: str) -> dict:
    source = validate_source_lock()
    specification = load_specification()
    runs = {}
    for seed in registered_seeds(specification, cohort):
        directory = run_directory(seed, cohort)
        metadata = validate_training_run(directory, specification)
        runs[str(seed)] = {
            "metadata": metadata,
            "files": [
                reference(path)
                for path in sorted(directory.iterdir())
                if path.is_file()
            ],
        }
    result = {
        "schema_version": 1,
        "experiment_id": specification["experiment_id"],
        "cohort": cohort,
        "source_lock": reference(SOURCE_LOCK_PATH),
        "source_commit": source["source_commit"],
        "runs": runs,
        "evaluation_exposed": False,
    }
    write_json_exclusive(artifact_lock_path(cohort), result)
    return result


def validate_artifact_lock(cohort: str) -> dict:
    source = validate_source_lock()
    path = artifact_lock_path(cohort)
    lock = load_json(path)
    if lock["source_lock"] != reference(SOURCE_LOCK_PATH):
        raise RuntimeError("artifact lock cites a different source lock")
    if lock["source_commit"] != source["source_commit"]:
        raise RuntimeError("artifact lock cites a different source commit")
    specification = load_specification()
    expected = {str(seed) for seed in registered_seeds(specification, cohort)}
    if set(lock["runs"]) != expected:
        raise RuntimeError("artifact lock does not contain the complete cohort")
    for seed, record in lock["runs"].items():
        directory = run_directory(int(seed), cohort)
        if record["metadata"] != validate_training_run(directory, specification):
            raise RuntimeError("locked training metadata changed")
        if record["files"] != [
            reference(file) for file in sorted(directory.iterdir()) if file.is_file()
        ]:
            raise RuntimeError("locked training files changed")
    return lock
