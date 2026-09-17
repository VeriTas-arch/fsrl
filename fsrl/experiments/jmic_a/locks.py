"""Committed source and allowed-input lock for JMIC-A."""

from __future__ import annotations

import subprocess

from fsrl.experiments.training_strategy.locks import reference, verify_reference
from fsrl.infra.git_provenance import git_blob_sha256
from fsrl.infra.provenance import file_sha256, load_json, write_json_exclusive
from fsrl.paths import REPO_ROOT

from .protocol import (
    PROTOCOL_SHA256,
    QUALIFICATION,
    REPAIR_SHA256,
    SOURCE_INPUT_LOCK,
    TASK_INPUT,
)
from .qualification import sources


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _clean_commit() -> str:
    if _git("branch", "--show-current") != "dev":
        raise RuntimeError("JMIC-A execution requires dev")
    if _git("status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("commit qualified JMIC-A source before locking")
    return _git("rev-parse", "HEAD")


def _require_committed(path) -> str:
    commit = _git("rev-parse", "HEAD")
    relative = path.relative_to(REPO_ROOT).as_posix()
    if git_blob_sha256(REPO_ROOT, commit, relative) != file_sha256(path):
        raise RuntimeError(f"JMIC-A freeze point is not committed: {relative}")
    return commit


def write_source_input_lock() -> dict:
    commit = _clean_commit()
    qualification = load_json(QUALIFICATION)
    current_sources = sources()
    if not qualification["passed"] or qualification["sources"] != current_sources:
        raise RuntimeError("qualification does not cover committed JMIC-A source")
    payload = {
        "schema_version": 1,
        "study_id": "jmic_a_v1",
        "protocol_sha256": PROTOCOL_SHA256,
        "quadrature_repair_sha256": REPAIR_SHA256,
        "source_commit": commit,
        "sources": current_sources,
        "qualification": reference(QUALIFICATION),
        "allowed_task_input": reference(TASK_INPUT),
        "participant_response_inputs": [],
        "neural_checkpoint_inputs": [],
        "scientific_outputs_exposed": False,
    }
    write_json_exclusive(SOURCE_INPUT_LOCK, payload)
    return {
        "source_commit": commit,
        "source_files": len(current_sources),
        "participant_response_inputs": 0,
    }


def validate_source_input_lock() -> dict:
    _require_committed(SOURCE_INPUT_LOCK)
    lock = load_json(SOURCE_INPUT_LOCK)
    if (
        lock["protocol_sha256"] != PROTOCOL_SHA256
        or lock["quadrature_repair_sha256"] != REPAIR_SHA256
        or lock["sources"] != sources()
        or lock["participant_response_inputs"]
        or lock["neural_checkpoint_inputs"]
    ):
        raise RuntimeError("JMIC-A source/input lock differs")
    for row in lock["sources"]:
        if (
            git_blob_sha256(REPO_ROOT, lock["source_commit"], row["path"])
            != row["sha256"]
        ):
            raise RuntimeError(f"JMIC-A Git source witness differs: {row['path']}")
    qualification = load_json(verify_reference(lock["qualification"]))
    if not qualification["passed"] or qualification["sources"] != lock["sources"]:
        raise RuntimeError("JMIC-A qualification lock differs")
    verify_reference(lock["allowed_task_input"])
    return lock


__all__ = ["validate_source_input_lock", "write_source_input_lock"]
