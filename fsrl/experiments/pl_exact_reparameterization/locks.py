"""Source and frozen-input locks for exact P/L qualification."""

from __future__ import annotations

import subprocess
from pathlib import Path

from fsrl.infra.file_contracts import safe_relative_path
from fsrl.infra.git_provenance import git_blob_sha256
from fsrl.infra.provenance import file_sha256, load_json, write_json_exclusive
from fsrl.paths import REPO_ROOT, STUDIES_ROOT
from fsrl.tasks.protocol_catalog import protocol_path

from .protocol import (
    NUMERICAL_REPAIR_COMMIT,
    NUMERICAL_REPAIR_PATH,
    NUMERICAL_REPAIR_SHA256,
    PROTOCOL_COMMIT,
    PROTOCOL_PATH,
    PROTOCOL_SHA256,
    REPAIR_COMMIT,
    REPAIR_PATH,
    REPAIR_SHA256,
    load_specification,
)

RECORD_ROOT = STUDIES_ROOT / "pl_exact_reparameterization" / "records"
SOURCE_LOCK_PATH = (
    RECORD_ROOT
    / "benchmarks"
    / ("pl_exact_reparameterization_v1.repair2.execution_lock.json")
)
ATTEMPT1_PATH = RECORD_ROOT / "results" / "pl_exact_reparameterization_v1.attempt1.json"


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
        raise RuntimeError("the exact-reparameterization workflow requires dev")
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
    if reference(path) != {key: record[key] for key in ("path", "sha256", "bytes")}:
        raise RuntimeError(f"locked file identity changed: {record['path']}")
    if commit is not None:
        observed = git_blob_sha256(REPO_ROOT, commit, record["path"])
        if observed != record["sha256"]:
            raise RuntimeError(f"Git witness differs: {record['path']}")
    return path


def implementation_sources() -> list[dict]:
    package = REPO_ROOT / "fsrl" / "experiments" / "pl_exact_reparameterization"
    tests = REPO_ROOT / "tests" / "experiments" / "pl_exact_reparameterization"
    paths = {
        REPO_ROOT / "fsrl" / "core" / "__init__.py",
        REPO_ROOT / "fsrl" / "core" / "factorized_plastic_rnn.py",
        REPO_ROOT / "fsrl" / "core" / "local_trace.py",
        REPO_ROOT / "fsrl" / "infra" / "formal_runtime.py",
        REPO_ROOT / "tests" / "core" / "test_factorized_plastic_rnn.py",
        REPO_ROOT / "tests" / "core" / "test_local_trace.py",
        REPO_ROOT / "tests" / "infra" / "test_formal_runtime.py",
        REPO_ROOT / "pyproject.toml",
        REPO_ROOT / ".envrc",
        *package.glob("*.py"),
        *tests.glob("*.py"),
    }
    return [reference(path) for path in sorted(paths)]


def scientific_inputs() -> list[dict]:
    specification = load_specification()
    paths = {
        PROTOCOL_PATH,
        REPAIR_PATH,
        NUMERICAL_REPAIR_PATH,
        ATTEMPT1_PATH,
        protocol_path("liu_v2"),
    }
    for source in specification["frozen_sources"].values():
        paths.add(REPO_ROOT / source["checkpoint"])
        paths.add(REPO_ROOT / source["gain"])
    return [reference(path) for path in sorted(paths)]


def _validate_declared_frozen_sources(inputs: list[dict]) -> None:
    specification = load_specification()
    by_path = {record["path"]: record for record in inputs}
    for source in specification["frozen_sources"].values():
        checkpoint = by_path[source["checkpoint"]]
        if (
            checkpoint["sha256"] != source["checkpoint_sha256"]
            or checkpoint["bytes"] != source["checkpoint_bytes"]
        ):
            raise RuntimeError("a frozen checkpoint differs from the contract")
        gain = by_path[source["gain"]]
        if gain["sha256"] != source["gain_sha256"]:
            raise RuntimeError("a frozen local gain differs from the contract")
        if (
            load_json(REPO_ROOT / source["gain"])["raw_lambda_L"]
            != source["raw_local_gain"]
        ):
            raise RuntimeError("a frozen local gain value differs from the contract")


def write_source_lock() -> dict:
    commit = require_clean_commit()
    sources = implementation_sources()
    for record in sources:
        verify_reference(record, commit=commit)
    inputs = scientific_inputs()
    _validate_declared_frozen_sources(inputs)
    for record in inputs:
        witness = (
            PROTOCOL_COMMIT
            if record["path"] == reference(PROTOCOL_PATH)["path"]
            else REPAIR_COMMIT
            if record["path"] == reference(REPAIR_PATH)["path"]
            else NUMERICAL_REPAIR_COMMIT
            if record["path"] == reference(NUMERICAL_REPAIR_PATH)["path"]
            else None
        )
        verify_reference(record, commit=witness)
    result = {
        "schema_version": 1,
        "study_id": load_specification()["study_id"],
        "source_commit": commit,
        "protocol": {**reference(PROTOCOL_PATH), "commit": PROTOCOL_COMMIT},
        "repair": {**reference(REPAIR_PATH), "commit": REPAIR_COMMIT},
        "numerical_repair": {
            **reference(NUMERICAL_REPAIR_PATH),
            "commit": NUMERICAL_REPAIR_COMMIT,
        },
        "sources": sources,
        "scientific_inputs": inputs,
        "execution_order": "source_locked_before_frozen_checkpoint_qualification",
        "remote_state": "not_required; local committed Git objects are the execution witness",
    }
    SOURCE_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_json_exclusive(SOURCE_LOCK_PATH, result)
    return result


def validate_source_lock(*, require_clean: bool = True) -> dict:
    current = require_clean_commit() if require_clean else git_text("rev-parse", "HEAD")
    lock = load_json(SOURCE_LOCK_PATH)
    if lock["protocol"]["sha256"] != PROTOCOL_SHA256:
        raise RuntimeError("source lock uses a different contract")
    if lock["repair"]["sha256"] != REPAIR_SHA256:
        raise RuntimeError("source lock uses a different timestep repair")
    if lock["numerical_repair"]["sha256"] != NUMERICAL_REPAIR_SHA256:
        raise RuntimeError("source lock uses a different numerical repair")
    verify_reference(lock["protocol"], commit=PROTOCOL_COMMIT)
    verify_reference(lock["repair"], commit=REPAIR_COMMIT)
    verify_reference(lock["numerical_repair"], commit=NUMERICAL_REPAIR_COMMIT)
    if lock["sources"] != implementation_sources():
        raise RuntimeError("implementation changed after the source lock")
    for record in lock["sources"]:
        verify_reference(record, commit=lock["source_commit"])
    if lock["scientific_inputs"] != scientific_inputs():
        raise RuntimeError("scientific inputs changed after the source lock")
    _validate_declared_frozen_sources(lock["scientific_inputs"])
    for record in lock["scientific_inputs"]:
        verify_reference(record)
    if require_clean and current != git_text("rev-parse", "HEAD"):
        raise RuntimeError("worktree moved during source validation")
    return lock
