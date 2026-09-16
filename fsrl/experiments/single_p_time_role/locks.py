"""Source, parent-artifact, and input locks for the time-role diagnostic."""

from __future__ import annotations

import subprocess

from fsrl.experiments.training_strategy.locks import reference, verify_reference
from fsrl.infra.provenance import file_sha256, load_json, write_json_exclusive
from fsrl.paths import REPO_ROOT

from .protocol import (
    ARITHMETIC_REPAIR,
    EXECUTION_REPAIR,
    IMPLEMENTATION_REPAIR,
    INTEGRITY_REPAIR,
    PROTOCOL,
    PROTOCOL_SHA256,
    QUALIFICATION,
    QUALIFICATION_FIX,
    QUALIFICATION_REPAIR,
    REPAIR,
    SOURCE_LOCK,
    specification,
)


def sources() -> list[dict]:
    paths = list((REPO_ROOT / "fsrl/experiments/single_p_time_role").glob("*.py"))
    paths += list((REPO_ROOT / "tests/experiments/single_p_time_role").glob("*.py"))
    paths += [
        REPO_ROOT / "fsrl/experiments/clean_single_p/adapter.py",
        REPO_ROOT / "fsrl/experiments/clean_single_p/evaluation.py",
        REPO_ROOT / "fsrl/experiments/clean_single_p/model.py",
        REPO_ROOT / "fsrl/experiments/finite_state/model.py",
        REPO_ROOT / "fsrl/experiments/local_memory_removal/model.py",
        REPO_ROOT / "fsrl/infra/formal_runtime.py",
        PROTOCOL,
        REPAIR,
        ARITHMETIC_REPAIR,
        IMPLEMENTATION_REPAIR,
        EXECUTION_REPAIR,
        INTEGRITY_REPAIR,
        QUALIFICATION_FIX,
        QUALIFICATION_REPAIR,
        REPO_ROOT / "pyproject.toml",
        REPO_ROOT / ".envrc",
    ]
    return [reference(path) for path in sorted(set(paths))]


def _clean_commit() -> str:
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT):
        raise RuntimeError("commit qualified time-role source before locking")
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def parent_references() -> dict:
    spec = specification()
    parents = {}
    for name, value in spec["parents"].items():
        path = REPO_ROOT / value["path"]
        if file_sha256(path) != value["sha256"]:
            raise RuntimeError(f"frozen parent differs: {name}")
        parents[name] = reference(path)
    parent_result = load_json(verify_reference(parents["clean_single_p_result"]))
    parents["parent_raw"] = parent_result["raw"]
    model_lock = load_json(verify_reference(parents["clean_single_p_model_lock"]))
    parents["checkpoints"] = {
        identity: row["files"]["model.pth"]
        for identity, row in model_lock["runs"].items()
    }
    source = load_json(verify_reference(parents["clean_single_p_source_repair"]))
    parents["inputs"] = {
        f"{panel}/{name}": value
        for panel, row in source["panels"].items()
        for name, value in row["inputs"].items()
    }
    evaluation_root = REPO_ROOT / "artifacts/runs/clean_single_p_v1/evaluation-attempt2"
    parents["parent_evaluation_raw"] = {
        "/".join(path.relative_to(evaluation_root).parts[:-1]): reference(path)
        for path in sorted(evaluation_root.glob("*/*/*/*/raw.npz"))
    }
    if len(parents["parent_evaluation_raw"]) != 72:
        raise RuntimeError("expected 72 frozen parent evaluation arrays")
    return parents


def write_source_lock() -> dict:
    commit = _clean_commit()
    qualification = load_json(QUALIFICATION)
    if not qualification["passed"] or qualification["sources"] != sources():
        raise RuntimeError("qualification does not cover committed diagnostic source")
    parents = parent_references()
    for group in ("checkpoints", "inputs", "parent_evaluation_raw"):
        for row in parents[group].values():
            verify_reference(row)
    payload = {
        "schema_version": 1,
        "protocol_sha256": PROTOCOL_SHA256,
        "source_commit": commit,
        "sources": sources(),
        "qualification": reference(QUALIFICATION),
        "parents": parents,
        "new_diagnostic_outcomes_exposed": False,
    }
    write_json_exclusive(SOURCE_LOCK, payload)
    return {
        "source_commit": commit,
        "checkpoints": len(parents["checkpoints"]),
        "inputs": len(parents["inputs"]),
        "parent_evaluation_raw": len(parents["parent_evaluation_raw"]),
    }


def validate_source_lock() -> dict:
    lock = load_json(SOURCE_LOCK)
    if lock["protocol_sha256"] != PROTOCOL_SHA256:
        raise RuntimeError("time-role source lock protocol differs")
    for row in lock["sources"]:
        verify_reference(row, commit=lock["source_commit"])
    verify_reference(lock["qualification"])
    for group in ("checkpoints", "inputs", "parent_evaluation_raw"):
        for row in lock["parents"][group].values():
            verify_reference(row)
    for name in (
        "clean_single_p_protocol",
        "clean_single_p_model_lock",
        "clean_single_p_source_repair",
        "clean_single_p_result",
        "parent_raw",
    ):
        verify_reference(lock["parents"][name])
    return lock


__all__ = [
    "parent_references",
    "reference",
    "sources",
    "validate_source_lock",
    "write_source_lock",
]
