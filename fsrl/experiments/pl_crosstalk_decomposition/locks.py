"""Source, input, and qualification lock for cross-talk decomposition."""

from __future__ import annotations

import subprocess
from pathlib import Path

from fsrl.infra.file_contracts import safe_relative_path
from fsrl.infra.git_provenance import git_blob_sha256
from fsrl.infra.provenance import file_sha256, load_json, write_json_exclusive
from fsrl.paths import REPO_ROOT, RUNS_ROOT, STUDIES_ROOT

from .protocol import (
    PROTOCOL_COMMIT,
    PROTOCOL_PATH,
    PROTOCOL_SHA256,
    load_specification,
    registered_seeds,
)

RUN_ROOT = RUNS_ROOT / "pl_crosstalk_decomposition_v1"
QUALIFICATION_PATH = RUN_ROOT / "qualification" / "qualification.json"
ANALYSIS_ROOT = RUN_ROOT / "analysis"
RUNTIME_RESULT_PATH = ANALYSIS_ROOT / "result.json"
RUNTIME_ARRAY_PATH = ANALYSIS_ROOT / "pl_crosstalk_decomposition_v1.npz"
RECORD_ROOT = STUDIES_ROOT / "pl_crosstalk_decomposition" / "records"
SOURCE_LOCK_PATH = (
    RECORD_ROOT / "benchmarks" / "pl_crosstalk_decomposition_v1.execution_lock.json"
)
RESULT_PATH = RECORD_ROOT / "results" / "pl_crosstalk_decomposition_v1.json"
ARRAY_PATH = RECORD_ROOT / "artifacts" / "pl_crosstalk_decomposition_v1.npz"
REPORT_PATH = RECORD_ROOT / "reports" / "pl_crosstalk_decomposition_v1.md"


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
        raise RuntimeError("cross-talk decomposition requires dev")
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
    expected = {key: record[key] for key in ("path", "sha256", "bytes")}
    if reference(path) != expected:
        raise RuntimeError(f"locked file identity changed: {record['path']}")
    if commit is not None:
        observed = git_blob_sha256(REPO_ROOT, commit, record["path"])
        if observed != record["sha256"]:
            raise RuntimeError(f"Git witness differs: {record['path']}")
    return path


def implementation_sources() -> list[dict]:
    package = REPO_ROOT / "fsrl" / "experiments" / "pl_crosstalk_decomposition"
    tests = REPO_ROOT / "tests" / "experiments" / "pl_crosstalk_decomposition"
    paths = {
        *package.glob("*.py"),
        *tests.glob("*.py"),
        REPO_ROOT / "fsrl" / "analysis" / "statistics.py",
        REPO_ROOT / "fsrl" / "evaluation" / "sampling.py",
        REPO_ROOT / "fsrl" / "infra" / "formal_runtime.py",
        REPO_ROOT / "fsrl" / "infra" / "provenance.py",
        REPO_ROOT / "fsrl" / "infra" / "study_registry.py",
        REPO_ROOT / "fsrl" / "tasks" / "protocol.py",
        REPO_ROOT / "fsrl" / "tasks" / "protocol_catalog.py",
        REPO_ROOT / "pyproject.toml",
        REPO_ROOT / ".envrc",
    }
    missing = sorted(path.as_posix() for path in paths if not path.is_file())
    if missing:
        raise RuntimeError(f"cross-talk implementation sources are missing: {missing}")
    return [reference(path) for path in sorted(paths)]


def scientific_inputs() -> list[dict]:
    specification = load_specification()
    frozen = specification["design"]["frozen_inputs"]
    paths = {
        PROTOCOL_PATH,
        REPO_ROOT / frozen["parent_contract"]["path"],
        REPO_ROOT / frozen["parent_result"]["path"],
        *(
            REPO_ROOT / frozen["raw_arrays"][str(seed)]["path"]
            for seed in registered_seeds(specification)
        ),
    }
    missing = sorted(path.as_posix() for path in paths if not path.is_file())
    if missing:
        raise RuntimeError(f"cross-talk scientific inputs are missing: {missing}")
    return [reference(path) for path in sorted(paths)]


def _validate_declared_inputs(inputs: list[dict]) -> None:
    specification = load_specification()
    frozen = specification["design"]["frozen_inputs"]
    by_path = {row["path"]: row for row in inputs}
    for name in ("parent_contract", "parent_result"):
        declared = frozen[name]
        observed = by_path[declared["path"]]
        if observed["sha256"] != declared["sha256"]:
            raise RuntimeError(f"declared {name} changed")
    parent_result = load_json(REPO_ROOT / frozen["parent_result"]["path"])
    for seed in registered_seeds(specification):
        declared = frozen["raw_arrays"][str(seed)]
        observed = by_path[declared["path"]]
        if observed != declared:
            raise RuntimeError(f"declared raw arrays changed for seed {seed}")
        if parent_result["per_seed"][str(seed)]["raw_arrays"] != declared:
            raise RuntimeError(f"parent result cites different raw arrays for {seed}")


def _validate_qualification(record: dict, sources: list[dict]) -> None:
    expected = {
        "passed": True,
        "seed": 941001,
        "frozen_inputs_loaded": False,
        "protocol_sha256": PROTOCOL_SHA256,
    }
    if any(record.get(key) != value for key, value in expected.items()):
        raise RuntimeError(
            "successful synthetic decomposition qualification is required"
        )
    required = {
        "packed_key_norm_and_antisymmetry",
        "source_relation_sum",
        "gain_scaled_perturbation",
        "exact_probability_effect",
        "first_order_definition",
        "retained_subject_weighting",
        "source_concentration",
        "deterministic_npz_roundtrip",
    }
    if set(record.get("checks", {})) != required or not all(
        row["passed"] is True for row in record["checks"].values()
    ):
        raise RuntimeError("synthetic decomposition qualification is incomplete")
    if record.get("sources") != sources:
        raise RuntimeError("qualification did not exercise the locked sources")


def write_source_lock() -> dict:
    commit = require_clean_commit()
    sources = implementation_sources()
    for record in sources:
        verify_reference(record, commit=commit)
    inputs = scientific_inputs()
    _validate_declared_inputs(inputs)
    for record in inputs:
        witness = (
            PROTOCOL_COMMIT
            if record["path"] == reference(PROTOCOL_PATH)["path"]
            else None
        )
        verify_reference(record, commit=witness)
    qualification = load_json(QUALIFICATION_PATH)
    _validate_qualification(qualification, sources)
    result = {
        "schema_version": 1,
        "study_id": load_specification()["study_id"],
        "source_commit": commit,
        "protocol": {**reference(PROTOCOL_PATH), "commit": PROTOCOL_COMMIT},
        "sources": sources,
        "scientific_inputs": inputs,
        "qualification": reference(QUALIFICATION_PATH),
        "qualification_summary": {
            key: qualification[key]
            for key in ("passed", "seed", "frozen_inputs_loaded", "runtime", "checks")
        },
        "execution_order": "synthetic_qualification_then_source_and_input_lock_then_frozen_read_only_analysis",
        "remote_state": "not_required; local committed Git objects and exact file hashes are the witnesses",
    }
    SOURCE_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_json_exclusive(SOURCE_LOCK_PATH, result)
    return result


def validate_source_lock(*, require_clean: bool = True) -> dict:
    if require_clean:
        require_clean_commit()
    lock = load_json(SOURCE_LOCK_PATH)
    if lock["protocol"]["sha256"] != PROTOCOL_SHA256:
        raise RuntimeError("source lock uses a different decomposition contract")
    verify_reference(lock["protocol"], commit=PROTOCOL_COMMIT)
    if lock["sources"] != implementation_sources():
        raise RuntimeError("implementation changed after the source lock")
    for record in lock["sources"]:
        verify_reference(record, commit=lock["source_commit"])
    inputs = scientific_inputs()
    if lock["scientific_inputs"] != inputs:
        raise RuntimeError("scientific inputs changed after the source lock")
    _validate_declared_inputs(inputs)
    for record in inputs:
        verify_reference(record)
    qualification = load_json(verify_reference(lock["qualification"]))
    _validate_qualification(qualification, lock["sources"])
    return lock
