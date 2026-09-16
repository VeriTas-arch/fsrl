"""Source and artifact locks for functional replication."""

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

from .protocol import CONDITION

RUN_ROOT = RUNS_ROOT / "pl_functional_replication_v1"
RECORD_ROOT = STUDIES_ROOT / "pl_functional_replication" / "records"
SOURCE_LOCK_PATH = (
    RECORD_ROOT / "benchmarks" / "pl_functional_replication_v1.execution_lock.json"
)
SOURCE_REPAIR_LOCK_PATH = (
    RECORD_ROOT / "benchmarks" / "pl_functional_replication_v1.source_repair1.json"
)
ARTIFACT_LOCK_PATH = (
    RECORD_ROOT / "benchmarks" / "pl_functional_replication_v1.artifact_lock.json"
)
ORIGINAL_QUALIFICATION_ATTEMPT = 2
QUALIFICATION_ATTEMPT = 3


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
        raise RuntimeError("functional replication requires dev")
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
    paths = list((REPO_ROOT / "fsrl").rglob("*.py")) + list(tests.rglob("*.py"))
    paths.extend([REPO_ROOT / "pyproject.toml", REPO_ROOT / ".envrc"])
    return [reference(path) for path in sorted(set(paths))]


def qualification_path(attempt: int | None = None) -> Path:
    selected = QUALIFICATION_ATTEMPT if attempt is None else attempt
    return RUN_ROOT / f"qualification-attempt{selected}" / "qualification.json"


def scientific_inputs() -> list[dict]:
    from .protocol import PROTOCOL_PATH, REPAIR_PATH, load_specification

    specification = load_specification()
    paths = {PROTOCOL_PATH, REPAIR_PATH}
    for section, name in (
        ("historical_claim_provenance", "source_contract"),
        ("historical_claim_provenance", "source_result"),
        ("historical_claim_provenance", "old_direct_training_result"),
        ("candidate_identity", "parent_contract"),
        ("candidate_identity", "parent_repair"),
        ("candidate_identity", "exact_structure_result"),
        ("competence_and_global_path_prerequisites", "threshold_provenance"),
        ("behavior_reproduction_contract", "reference_contract"),
        ("behavior_reproduction_contract", "reference_result"),
        ("triggered_item_count_transport", "reference_contract"),
    ):
        paths.add(REPO_ROOT / specification[section][name]["path"])
    behavior_contract = load_json(
        REPO_ROOT
        / specification["behavior_reproduction_contract"]["reference_contract"]["path"]
    )
    paths.add(
        resolve_record(
            behavior_contract["registered_sources"]["human_benchmark"]["path"]
        )
    )
    paths.add(
        REPO_ROOT
        / "studies"
        / "task_fidelity"
        / "records"
        / "benchmarks"
        / "liu_v2.json"
    )
    missing = sorted(path.as_posix() for path in paths if not path.is_file())
    if missing:
        raise RuntimeError(f"functional-replication inputs are missing: {missing}")
    return [reference(path) for path in sorted(paths)]


def _validate_qualification(
    record: dict,
    *,
    expected_attempt: int,
    expected_sources: list[dict],
) -> None:
    from .protocol import PROTOCOL_SHA256, REPAIR_SHA256

    expected = {
        "passed": True,
        "seed": 930001,
        "attempt": expected_attempt,
        "liu_evaluated": False,
        "protocol_sha256": PROTOCOL_SHA256,
        "repair_sha256": REPAIR_SHA256,
    }
    if any(record.get(key) != value for key, value in expected.items()):
        raise RuntimeError(
            "successful functional-replication qualification is required"
        )
    required = {
        "initialization_and_fresh_boundary",
        "no_time_isolation",
        "optimizer_forward_loss",
        "optimizer_effective_parameters",
        "optimizer_mapped_moments",
        "packed_local",
        "timestep_identity",
        "gradient_paths",
        "access_algebra",
        "presentation_invariance",
        "historical_estimand_parity",
        "routing_integrity",
        "cuda_time_retained_control_output",
        "cuda_time_retained_control_gradients",
        "cuda_time_retained_control_update",
        "cuda_no_time_candidate_output",
        "cuda_no_time_candidate_gradients",
        "cuda_no_time_candidate_update",
    }
    if expected_attempt >= 3:
        required.add("nullable_endpoint_parity")
    if set(record["checks"]) != required or not all(
        row["passed"] is True for row in record["checks"].values()
    ):
        raise RuntimeError("functional-replication qualification is incomplete")
    if record["sources"] != expected_sources:
        raise RuntimeError("qualification did not exercise the expected sources")


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
    sources = implementation_sources()
    _validate_qualification(
        qualification,
        expected_attempt=QUALIFICATION_ATTEMPT,
        expected_sources=sources,
    )
    for record in sources:
        verify_reference(record, commit=commit)
    inputs = scientific_inputs()
    protocol_path = reference(PROTOCOL_PATH)["path"]
    repair_path = reference(REPAIR_PATH)["path"]
    for record in inputs:
        witness = (
            PROTOCOL_COMMIT
            if record["path"] == protocol_path
            else REPAIR_COMMIT
            if record["path"] == repair_path
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


def _validate_original_source_lock(lock: dict) -> None:
    from .protocol import PROTOCOL_SHA256, REPAIR_SHA256

    if lock["protocol"]["sha256"] != PROTOCOL_SHA256:
        raise RuntimeError("source lock uses a different functional contract")
    if lock["repair"]["sha256"] != REPAIR_SHA256:
        raise RuntimeError("source lock uses a different functional repair")
    for record in lock["sources"]:
        if (
            git_blob_sha256(REPO_ROOT, lock["source_commit"], record["path"])
            != record["sha256"]
        ):
            raise RuntimeError(f"original Git witness differs: {record['path']}")
    if lock["scientific_inputs"] != scientific_inputs():
        raise RuntimeError("functional-replication scientific inputs changed")
    qualification = load_json(verify_reference(lock["qualification"]))
    _validate_qualification(
        qualification,
        expected_attempt=ORIGINAL_QUALIFICATION_ATTEMPT,
        expected_sources=lock["sources"],
    )
    expected_summary = {
        key: qualification[key]
        for key in ("passed", "seed", "attempt", "runtime", "checks")
    }
    if lock["qualification_summary"] != expected_summary:
        raise RuntimeError("source lock misstates functional qualification")


def _source_replacements(original: list[dict], current: list[dict]) -> list[dict]:
    original_by_path = {row["path"]: row for row in original}
    current_by_path = {row["path"]: row for row in current}
    if set(original_by_path) != set(current_by_path):
        raise RuntimeError("source repair may not add or remove implementation paths")
    return [
        {
            "original": original_by_path[path],
            "replacement": current_by_path[path],
        }
        for path in sorted(original_by_path)
        if original_by_path[path] != current_by_path[path]
    ]


def write_source_repair_lock() -> dict:
    from .protocol import (
        EXECUTION_REPAIR_PATH,
        load_execution_repair,
        load_specification,
    )

    commit = require_clean_commit()
    specification = load_specification()
    original = load_json(SOURCE_LOCK_PATH)
    _validate_original_source_lock(original)
    artifact_lock = load_json(ARTIFACT_LOCK_PATH)
    if (
        artifact_lock["source_lock"] != reference(SOURCE_LOCK_PATH)
        or artifact_lock["source_commit"] != original["source_commit"]
    ):
        raise RuntimeError("artifact lock does not belong to the original source lock")
    execution_repair = load_execution_repair()
    if execution_repair["authorized_repair"]["scientific_change"] is not False:
        raise RuntimeError("source repair must preserve all scientific content")
    failed_attempt = execution_repair["failure"]["failed_run_manifest"]
    verify_reference(failed_attempt)
    sources = implementation_sources()
    for record in sources:
        verify_reference(record, commit=commit)
    qualification = load_json(qualification_path())
    _validate_qualification(
        qualification,
        expected_attempt=QUALIFICATION_ATTEMPT,
        expected_sources=sources,
    )
    result = {
        "schema_version": 1,
        "experiment_id": specification["experiment_id"],
        "repair_id": execution_repair["repair_id"],
        "original_source_lock": reference(SOURCE_LOCK_PATH),
        "artifact_lock": reference(ARTIFACT_LOCK_PATH),
        "execution_repair": reference(EXECUTION_REPAIR_PATH),
        "training_source_commit": original["source_commit"],
        "source_commit": commit,
        "sources": sources,
        "replacements": _source_replacements(original["sources"], sources),
        "scientific_inputs": original["scientific_inputs"],
        "qualification": reference(qualification_path()),
        "qualification_summary": {
            key: qualification[key]
            for key in ("passed", "seed", "attempt", "runtime", "checks")
        },
        "failed_attempt": failed_attempt,
        "unchanged": execution_repair["unchanged"],
        "status": "locked_integrity_only_repair_before_complete_evaluation_replay",
    }
    write_json_exclusive(SOURCE_REPAIR_LOCK_PATH, result)
    return result


def validate_source_lock(*, require_clean: bool = True) -> dict:
    from .protocol import EXECUTION_REPAIR_PATH, load_execution_repair

    commit = require_clean_commit() if require_clean else git_text("rev-parse", "HEAD")
    original = load_json(SOURCE_LOCK_PATH)
    _validate_original_source_lock(original)
    current_sources = implementation_sources()
    if not SOURCE_REPAIR_LOCK_PATH.exists():
        if original["sources"] != current_sources:
            raise RuntimeError("functional-replication source changed after lock")
        for record in original["sources"]:
            verify_reference(record, commit=original["source_commit"])
        return {
            **original,
            "training_source_commit": original["source_commit"],
            "active_source_commit": original["source_commit"],
            "source_repair": None,
        }
    verify_reference(reference(SOURCE_REPAIR_LOCK_PATH), commit=commit)
    repair = load_json(SOURCE_REPAIR_LOCK_PATH)
    execution_repair = load_execution_repair()
    expected = {
        "repair_id": execution_repair["repair_id"],
        "original_source_lock": reference(SOURCE_LOCK_PATH),
        "artifact_lock": reference(ARTIFACT_LOCK_PATH),
        "execution_repair": reference(EXECUTION_REPAIR_PATH),
        "training_source_commit": original["source_commit"],
        "scientific_inputs": original["scientific_inputs"],
        "failed_attempt": execution_repair["failure"]["failed_run_manifest"],
        "unchanged": execution_repair["unchanged"],
    }
    if any(repair.get(key) != value for key, value in expected.items()):
        raise RuntimeError("functional source-repair provenance differs")
    if repair["sources"] != current_sources:
        raise RuntimeError("functional source changed after the repair lock")
    for record in current_sources:
        verify_reference(record, commit=repair["source_commit"])
    if repair["replacements"] != _source_replacements(
        original["sources"], current_sources
    ):
        raise RuntimeError("functional source-repair replacements differ")
    qualification = load_json(verify_reference(repair["qualification"]))
    _validate_qualification(
        qualification,
        expected_attempt=QUALIFICATION_ATTEMPT,
        expected_sources=current_sources,
    )
    expected_summary = {
        key: qualification[key]
        for key in ("passed", "seed", "attempt", "runtime", "checks")
    }
    if repair["qualification_summary"] != expected_summary:
        raise RuntimeError("source repair misstates functional qualification")
    verify_reference(repair["failed_attempt"])
    if require_clean and commit != git_text("rev-parse", "HEAD"):
        raise RuntimeError("worktree moved during source validation")
    return {
        **repair,
        "active_source_commit": repair["source_commit"],
        "source_repair": reference(SOURCE_REPAIR_LOCK_PATH),
    }


def run_directory(seed: int) -> Path:
    from .protocol import registered_seeds

    if seed not in registered_seeds():
        raise ValueError("unregistered functional-replication seed")
    return RUN_ROOT / "training" / f"seed-{seed}" / CONDITION


def _validate_training_log(
    directory: Path, metadata: dict, specification: dict
) -> None:
    steps = int(specification["optimization"]["total_steps"])
    with (directory / "train_log.jsonl").open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    if [row["step"] for row in rows] != list(range(steps)):
        raise RuntimeError("functional training log lacks a registered update")
    digest = hashlib.sha256()
    for row in rows:
        digest.update(bytes.fromhex(row["batch_fingerprint"]))
        if row["stream_fingerprint"] != digest.hexdigest():
            raise RuntimeError("functional training stream hash chain differs")
    if metadata["stream_fingerprint"] != digest.hexdigest():
        raise RuntimeError("functional final stream hash differs")


def validate_training_run(directory: Path) -> dict:
    from .protocol import (
        PARENT_PROTOCOL_SHA256,
        PARENT_REPAIR_SHA256,
        PROTOCOL_SHA256,
        REPAIR_SHA256,
        candidate_specification,
    )

    required = {"run.json", "config.json", "model.pth", "train_log.jsonl"}
    if not required.issubset(
        path.name for path in directory.iterdir() if path.is_file()
    ):
        raise RuntimeError("functional training run lacks required artifacts")
    manifest = load_json(directory / "run.json")
    if (
        manifest["lifecycle_state"] != "complete"
        or not validate_run_manifest(directory / "run.json")["passed"]
    ):
        raise RuntimeError("functional training run is incomplete or modified")
    metadata = load_json(directory / "config.json")
    seed = int(directory.parent.name.removeprefix("seed-"))
    specification = candidate_specification()
    expected = {
        "protocol_sha256": PROTOCOL_SHA256,
        "repair_sha256": REPAIR_SHA256,
        "parent_protocol_sha256": PARENT_PROTOCOL_SHA256,
        "parent_repair_sha256": PARENT_REPAIR_SHA256,
        "seed": seed,
        "condition": CONDITION,
        "optimization": specification["optimization"],
        "episode_exposures": specification["optimization"]["total_episode_exposures"],
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise RuntimeError("functional training identity differs from the contract")
    architecture = metadata["architecture"]
    if architecture != {
        **architecture,
        "task_input_size": 32,
        "hidden_size": 200,
        "support_steps": 4,
        "query_steps": 2,
        "P_scalars": 40000,
        "L_scalars": 105,
        "backbone_parameters": 87205,
        "time_parameters": 0,
        "compatibility_buffers": 0,
    }:
        raise RuntimeError("functional architecture identity differs")
    _validate_training_log(directory, metadata, specification)
    expected_steps = int(specification["optimization"]["total_steps"])
    counters = metadata["optimizer_parameter_steps"]
    if not counters or set(counters.values()) != {expected_steps}:
        raise RuntimeError("functional optimizer counters differ from the budget")
    if verify_reference(metadata["checkpoint"]) != directory / "model.pth":
        raise RuntimeError("functional checkpoint identity differs")
    return metadata


def write_artifact_lock() -> dict:
    from .protocol import load_specification, registered_seeds

    source = validate_source_lock()
    specification = load_specification()
    runs = {}
    for seed in registered_seeds(specification):
        directory = run_directory(seed)
        metadata = validate_training_run(directory)
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
        "source_lock": reference(SOURCE_LOCK_PATH),
        "source_commit": source["training_source_commit"],
        "runs": runs,
        "generic_or_liu_evaluation_exposed": False,
    }
    write_json_exclusive(ARTIFACT_LOCK_PATH, result)
    return result


def validate_artifact_lock() -> dict:
    from .protocol import load_specification, registered_seeds

    source = validate_source_lock()
    lock = load_json(ARTIFACT_LOCK_PATH)
    if lock["source_lock"] != reference(SOURCE_LOCK_PATH):
        raise RuntimeError("artifact lock cites a different source lock")
    if lock["source_commit"] != source["training_source_commit"]:
        raise RuntimeError("artifact lock cites a different source commit")
    specification = load_specification()
    expected = {str(seed) for seed in registered_seeds(specification)}
    if set(lock["runs"]) != expected:
        raise RuntimeError("artifact lock does not contain all fresh seeds")
    for seed, record in lock["runs"].items():
        directory = run_directory(int(seed))
        if record["metadata"] != validate_training_run(directory):
            raise RuntimeError("locked functional metadata changed")
        files = [
            reference(path) for path in sorted(directory.iterdir()) if path.is_file()
        ]
        if record["files"] != files:
            raise RuntimeError("locked functional files changed")
    return {
        **lock,
        "active_evaluation_source_commit": source["active_source_commit"],
        "source_repair": source["source_repair"],
    }
