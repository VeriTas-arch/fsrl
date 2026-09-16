"""Source and artifact identities for direct P/L training."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from fsrl.infra.file_contracts import validate_run_manifest
from fsrl.infra.git_provenance import git_blob_sha256
from fsrl.infra.provenance import file_sha256, load_json, write_json_exclusive
from fsrl.infra.study_registry import resolve_record
from fsrl.paths import REPO_ROOT, RUNS_ROOT, STUDIES_ROOT

RUN_ROOT = RUNS_ROOT / "pl_direct_training_v1"
RECORD_ROOT = STUDIES_ROOT / "pl_direct_training" / "records"
SOURCE_LOCK_PATH = (
    RECORD_ROOT / "benchmarks" / "pl_direct_training_v1.execution_lock.json"
)
QUALIFICATION_ATTEMPT = 2


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
        raise RuntimeError("direct P/L training requires dev")
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
    from fsrl.infra.file_contracts import safe_relative_path

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
    tests = REPO_ROOT / "tests" / "experiments" / "pl_direct_training"
    paths = list((REPO_ROOT / "fsrl").rglob("*.py")) + list(tests.glob("*.py"))
    paths.extend([REPO_ROOT / "pyproject.toml", REPO_ROOT / ".envrc"])
    return [reference(path) for path in sorted(set(paths))]


def qualification_path() -> Path:
    return (
        RUN_ROOT
        / f"qualification-attempt{QUALIFICATION_ATTEMPT}"
        / "qualification.json"
    )


def scientific_inputs() -> list[dict]:
    from .protocol import PROTOCOL_PATH, REPAIR_PATH, load_specification

    specification = load_specification()
    behavior_contract = (
        REPO_ROOT
        / "studies"
        / "behavior_reproduction_map"
        / "records"
        / "benchmarks"
        / "model_behavior_reproduction_map_v1.json"
    )
    human_record = load_json(behavior_contract)["registered_sources"][
        "human_benchmark"
    ]["path"]
    paths = {
        PROTOCOL_PATH,
        REPAIR_PATH,
        REPO_ROOT / specification["parent_evidence"]["exact_structure"],
        REPO_ROOT / specification["parent_evidence"]["training_strategy"],
        behavior_contract,
        REPO_ROOT
        / "studies"
        / "behavior_reproduction_map"
        / "records"
        / "results"
        / "model_behavior_reproduction_map_v1.json",
        resolve_record(human_record),
        REPO_ROOT
        / "studies"
        / "task_fidelity"
        / "records"
        / "benchmarks"
        / "liu_v2.json",
    }
    missing = sorted(path.as_posix() for path in paths if not path.is_file())
    if missing:
        raise RuntimeError(f"direct-training scientific inputs are missing: {missing}")
    return [reference(path) for path in sorted(paths)]


def _validate_qualification(record: dict) -> None:
    from .protocol import PROTOCOL_SHA256, REPAIR_SHA256

    expected = {
        "passed": True,
        "seed": 930001,
        "attempt": QUALIFICATION_ATTEMPT,
        "liu_evaluated": False,
        "protocol_sha256": PROTOCOL_SHA256,
        "repair_sha256": REPAIR_SHA256,
    }
    if any(record.get(key) != value for key, value in expected.items()):
        raise RuntimeError("successful direct-training qualification is required")
    required = {
        "initialization_and_fresh_boundary",
        "no_time_isolation",
        "optimizer_forward_loss",
        "optimizer_effective_parameters",
        "optimizer_mapped_moments",
        "packed_local",
        "timestep_identity",
        "gradient_paths",
        "cuda_time_retained_control_output",
        "cuda_time_retained_control_gradients",
        "cuda_time_retained_control_update",
        "cuda_no_time_candidate_output",
        "cuda_no_time_candidate_gradients",
        "cuda_no_time_candidate_update",
    }
    if set(record["checks"]) != required or not all(
        row["passed"] is True for row in record["checks"].values()
    ):
        raise RuntimeError("direct-training qualification checks are incomplete")
    if record["sources"] != implementation_sources():
        raise RuntimeError("qualification did not exercise the current sources")


def write_source_lock() -> dict:
    from .protocol import (
        PROTOCOL_COMMIT,
        PROTOCOL_PATH,
        REPAIR_COMMIT,
        REPAIR_PATH,
        load_specification,
    )

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
            key: qualification[key]
            for key in ("passed", "seed", "attempt", "runtime", "checks")
        },
        "remote_state": "not_required; local committed Git objects are the execution witness",
    }
    SOURCE_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_json_exclusive(SOURCE_LOCK_PATH, result)
    return result


def validate_source_lock(*, require_clean: bool = True) -> dict:
    from .protocol import PROTOCOL_SHA256, REPAIR_SHA256

    commit = require_clean_commit() if require_clean else git_text("rev-parse", "HEAD")
    lock = load_json(SOURCE_LOCK_PATH)
    if lock["protocol"]["sha256"] != PROTOCOL_SHA256:
        raise RuntimeError("source lock uses a different direct-training contract")
    if lock["repair"]["sha256"] != REPAIR_SHA256:
        raise RuntimeError("source lock uses a different direct-training repair")
    if lock["sources"] != implementation_sources():
        raise RuntimeError("direct-training implementation changed after source lock")
    for record in lock["sources"]:
        verify_reference(record, commit=lock["source_commit"])
    if lock["scientific_inputs"] != scientific_inputs():
        raise RuntimeError("direct-training scientific inputs changed")
    qualification = load_json(verify_reference(lock["qualification"]))
    _validate_qualification(qualification)
    if lock["qualification_summary"] != {
        key: qualification[key]
        for key in ("passed", "seed", "attempt", "runtime", "checks")
    }:
        raise RuntimeError("source lock misstates direct-training qualification")
    if require_clean and commit != git_text("rev-parse", "HEAD"):
        raise RuntimeError("worktree moved during direct-training source validation")
    return lock


def run_directory(seed: int, condition: str, cohort: str) -> Path:
    from .protocol import (
        load_specification,
        registered_conditions,
        registered_seeds,
    )

    specification = load_specification()
    if seed not in registered_seeds(
        specification, cohort
    ) or condition not in registered_conditions(specification):
        raise ValueError("unregistered direct-training run identity")
    return RUN_ROOT / cohort / f"seed-{seed}" / condition


def _validate_training_log(
    directory: Path, metadata: dict, specification: dict
) -> None:
    steps = specification["optimization"]["total_steps"]
    with (directory / "train_log.jsonl").open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    if [row["step"] for row in rows] != list(range(steps)):
        raise RuntimeError("direct-training log lacks a registered update")
    digest = hashlib.sha256()
    for row in rows:
        digest.update(bytes.fromhex(row["batch_fingerprint"]))
        if row["stream_fingerprint"] != digest.hexdigest():
            raise RuntimeError("direct-training stream hash chain differs")
    if metadata["stream_fingerprint"] != digest.hexdigest():
        raise RuntimeError("final direct-training stream hash differs")


def validate_training_run(directory: Path, specification: dict, cohort: str) -> dict:
    from .protocol import PROTOCOL_SHA256, REPAIR_SHA256, registered_conditions

    required = {"run.json", "config.json", "model.pth", "train_log.jsonl"}
    if not required.issubset(
        path.name for path in directory.iterdir() if path.is_file()
    ):
        raise RuntimeError("direct-training run lacks required artifacts")
    manifest = load_json(directory / "run.json")
    if (
        manifest["lifecycle_state"] != "complete"
        or not validate_run_manifest(directory / "run.json")["passed"]
    ):
        raise RuntimeError("direct-training run is incomplete or modified")
    metadata = load_json(directory / "config.json")
    seed = int(directory.parent.name.removeprefix("seed-"))
    condition = directory.name
    expected = {
        "protocol_sha256": PROTOCOL_SHA256,
        "repair_sha256": REPAIR_SHA256,
        "seed": seed,
        "condition": condition,
        "cohort": cohort,
        "optimization": specification["optimization"],
        "episode_exposures": specification["optimization"]["total_episode_exposures"],
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise RuntimeError("direct-training run identity differs from the contract")
    if condition not in registered_conditions(specification):
        raise RuntimeError("direct-training run has an unknown condition")
    architecture = metadata["architecture"]
    expected_parameters = specification["architecture"][condition][
        "backbone_parameters"
    ]
    if (
        architecture["backbone_parameters"] != expected_parameters
        or architecture["compatibility_buffers"] != 0
        or architecture["L_scalars"] != 105
        or architecture["P_scalars"] != 40000
    ):
        raise RuntimeError("direct-training architecture identity differs")
    _validate_training_log(directory, metadata, specification)
    expected_steps = specification["optimization"]["total_steps"]
    counters = metadata["optimizer_parameter_steps"]
    if not counters or set(counters.values()) != {expected_steps}:
        raise RuntimeError("direct-training optimizer counters differ from the budget")
    if verify_reference(metadata["checkpoint"]) != directory / "model.pth":
        raise RuntimeError("direct-training checkpoint identity differs")
    return metadata


def artifact_lock_path(cohort: str) -> Path:
    if cohort != "development" and cohort != "confirmation":
        raise ValueError(f"unknown cohort: {cohort}")
    return (
        RECORD_ROOT
        / "benchmarks"
        / f"pl_direct_training_v1.{cohort}_artifact_lock.json"
    )


def write_artifact_lock(cohort: str) -> dict:
    from .protocol import (
        load_specification,
        registered_conditions,
        registered_seeds,
    )

    source = validate_source_lock()
    specification = load_specification()
    runs = {}
    for seed in registered_seeds(specification, cohort):
        for condition in registered_conditions(specification):
            directory = run_directory(seed, condition, cohort)
            metadata = validate_training_run(directory, specification, cohort)
            runs[f"{seed}/{condition}"] = {
                "metadata": metadata,
                "files": [
                    reference(path)
                    for path in sorted(directory.iterdir())
                    if path.is_file()
                ],
            }
    for seed in registered_seeds(specification, cohort):
        control = runs[f"{seed}/time_retained_control"]["metadata"]
        candidate = runs[f"{seed}/no_time_candidate"]["metadata"]
        for key in (
            "initial_shadow",
            "initial_shared",
            "initial_local",
            "stream_fingerprint",
        ):
            if control[key] != candidate[key]:
                raise RuntimeError(f"paired direct-training {key} differs")
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
    from .protocol import (
        load_specification,
        registered_conditions,
        registered_seeds,
    )

    source = validate_source_lock()
    lock = load_json(artifact_lock_path(cohort))
    if lock["source_lock"] != reference(SOURCE_LOCK_PATH):
        raise RuntimeError("artifact lock cites a different source lock")
    if lock["source_commit"] != source["source_commit"]:
        raise RuntimeError("artifact lock cites a different source commit")
    specification = load_specification()
    expected = {
        f"{seed}/{condition}"
        for seed in registered_seeds(specification, cohort)
        for condition in registered_conditions(specification)
    }
    if set(lock["runs"]) != expected:
        raise RuntimeError("artifact lock does not contain the paired cohort")
    for identity, record in lock["runs"].items():
        seed, condition = identity.split("/", 1)
        directory = run_directory(int(seed), condition, cohort)
        if record["metadata"] != validate_training_run(
            directory, specification, cohort
        ):
            raise RuntimeError("locked direct-training metadata changed")
        if record["files"] != [
            reference(path) for path in sorted(directory.iterdir()) if path.is_file()
        ]:
            raise RuntimeError("locked direct-training files changed")
    return lock
