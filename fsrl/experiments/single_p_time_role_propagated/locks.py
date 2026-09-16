"""Prospective source and content locks for the propagated successor."""

from __future__ import annotations

import subprocess
from pathlib import Path

from fsrl.infra.provenance import file_sha256, load_json, write_json_exclusive
from fsrl.paths import REPO_ROOT

from .protocol import (
    BASELINE_ARTIFACT_LOCK,
    BASELINE_RESULT,
    BASELINE_RUNS,
    PROTOCOL,
    PROTOCOL_SHA256,
    QUALIFICATION,
    REPAIRS,
    SOURCE_LOCK,
    specification,
)


def reference(path: Path) -> dict:
    target = path if path.is_absolute() else REPO_ROOT / path
    return {
        "path": target.relative_to(REPO_ROOT).as_posix(),
        "sha256": file_sha256(target),
        "bytes": target.stat().st_size,
    }


def verify_reference(row: dict, *, commit: str | None = None) -> Path:
    path = REPO_ROOT / row["path"]
    if commit is None:
        if (
            not path.is_file()
            or path.stat().st_size != row["bytes"]
            or file_sha256(path) != row["sha256"]
        ):
            raise RuntimeError(f"locked reference differs: {row['path']}")
    else:
        payload = subprocess.check_output(
            ["git", "show", f"{commit}:{row['path']}"], cwd=REPO_ROOT
        )
        import hashlib

        if (
            len(payload) != row["bytes"]
            or hashlib.sha256(payload).hexdigest() != row["sha256"]
        ):
            raise RuntimeError(f"committed reference differs: {row['path']}")
    return path


def sources() -> list[dict]:
    paths = list(
        (REPO_ROOT / "fsrl/experiments/single_p_time_role_propagated").glob("*.py")
    )
    paths += list(
        (REPO_ROOT / "tests/experiments/single_p_time_role_propagated").glob("*.py")
    )
    paths += [
        REPO_ROOT / "fsrl/experiments/clean_single_p/model.py",
        REPO_ROOT / "fsrl/experiments/clean_single_p/protocol.py",
        REPO_ROOT / "fsrl/experiments/single_p_time_role_direct/batches.py",
        REPO_ROOT / "fsrl/experiments/single_p_time_role_direct/direct.py",
        REPO_ROOT / "fsrl/experiments/single_p_time_role_direct/runtime.py",
        REPO_ROOT / "fsrl/experiments/single_p_time_role_direct/storage.py",
        REPO_ROOT / "fsrl/experiments/finite_state/liu.py",
        REPO_ROOT / "fsrl/experiments/local_memory_removal/statistics.py",
        REPO_ROOT / "fsrl/experiments/memory_structure/interventions.py",
        REPO_ROOT / "fsrl/experiments/observation_replication/reporting.py",
        REPO_ROOT / "fsrl/experiments/observation_replication/statistics.py",
        REPO_ROOT / "fsrl/experiments/single_p_time_role/analysis.py",
        REPO_ROOT / "fsrl/experiments/single_p_time_role/estimands.py",
        REPO_ROOT / "fsrl/experiments/write_cost/evaluation.py",
        REPO_ROOT / "fsrl/infra/formal_runtime.py",
        REPO_ROOT / "fsrl/infra/runtime.py",
        PROTOCOL,
        *REPAIRS,
        REPO_ROOT / "pyproject.toml",
        REPO_ROOT / ".envrc",
    ]
    return [reference(path) for path in sorted(set(paths))]


def _clean_commit() -> str:
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT):
        raise RuntimeError(
            "commit qualified propagated-authority source before locking"
        )
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def parent_references() -> dict:
    spec = specification()
    parents = {}
    for name, value in spec["authorities"].items():
        path = REPO_ROOT / value["path"]
        if file_sha256(path) != value["sha256"]:
            raise RuntimeError(f"frozen authority differs: {name}")
        parents[name] = reference(path)
    model_lock = load_json(verify_reference(parents["parent_model_lock"]))
    parents["checkpoints"] = {}
    for identity, row in model_lock["runs"].items():
        checkpoint = dict(row["files"]["model.pth"])
        checkpoint["tensor_hashes"] = row["metadata"]["final_model"]
        parents["checkpoints"][identity] = checkpoint
    source = load_json(verify_reference(parents["parent_source_repair"]))
    parents["inputs"] = {
        f"{panel}/{name}": value
        for panel, row in source["panels"].items()
        for name, value in row["inputs"].items()
    }
    root = REPO_ROOT / "artifacts/runs/clean_single_p_v1/evaluation-attempt2"
    parents["parent_evaluation_raw"] = {
        "/".join(path.relative_to(root).parts[:-1]): reference(path)
        for path in sorted(root.glob("*/*/*/*/raw.npz"))
    }
    if len(parents["checkpoints"]) != 12:
        raise RuntimeError("expected 12 frozen parent checkpoints")
    if len(parents["inputs"]) != 15:
        raise RuntimeError("expected 15 frozen parent inputs")
    if len(parents["parent_evaluation_raw"]) != 72:
        raise RuntimeError("expected 72 frozen parent evaluation arrays")
    return parents


def write_source_lock(runtime: dict) -> dict:
    commit = _clean_commit()
    qualification = load_json(QUALIFICATION)
    if not qualification["passed"] or qualification["sources"] != sources():
        raise RuntimeError("qualification does not cover committed propagated source")
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
        "runtime": runtime,
        "parents": parents,
        "baseline_or_mechanism_outcomes_exposed": False,
    }
    write_json_exclusive(SOURCE_LOCK, payload)
    return {
        "source_commit": commit,
        "sources": len(payload["sources"]),
        "checkpoints": len(parents["checkpoints"]),
        "inputs": len(parents["inputs"]),
        "parent_evaluation_raw": len(parents["parent_evaluation_raw"]),
    }


def validate_source_lock(runtime: dict) -> dict:
    lock = load_json(SOURCE_LOCK)
    if lock["protocol_sha256"] != PROTOCOL_SHA256 or lock["runtime"] != runtime:
        raise RuntimeError("propagated-authority source lock differs")
    for row in lock["sources"]:
        verify_reference(row, commit=lock["source_commit"])
    verify_reference(lock["qualification"])
    for name, row in lock["parents"].items():
        if name in {"checkpoints", "inputs", "parent_evaluation_raw"}:
            for member in row.values():
                verify_reference(member)
        else:
            verify_reference(row)
    return lock


def _runtime_artifacts() -> list[Path]:
    return sorted(path for path in BASELINE_RUNS.rglob("*") if path.is_file())


def require_exact_inventory(observed: set[str], expected: set[str]) -> None:
    if observed != expected:
        raise RuntimeError("baseline runtime artifact inventory differs")


def write_baseline_artifact_lock() -> dict:
    result = load_json(BASELINE_RESULT)
    if result["outcome"] != "baseline_compatible":
        raise RuntimeError("only a compatible baseline may be artifact-locked")
    paths = _runtime_artifacts()
    if not paths:
        raise RuntimeError("propagated baseline has no runtime artifacts")
    payload = {
        "schema_version": 1,
        "protocol_sha256": PROTOCOL_SHA256,
        "source_lock": reference(SOURCE_LOCK),
        "baseline_result": reference(BASELINE_RESULT),
        "artifacts": [reference(path) for path in paths],
        "artifact_count": len(paths),
        "total_bytes": sum(path.stat().st_size for path in paths),
        "mechanism_outcomes_exposed": False,
    }
    write_json_exclusive(BASELINE_ARTIFACT_LOCK, payload)
    return {"artifact_count": len(paths), "total_bytes": payload["total_bytes"]}


def validate_baseline_artifact_lock() -> dict:
    lock = load_json(BASELINE_ARTIFACT_LOCK)
    if lock["protocol_sha256"] != PROTOCOL_SHA256:
        raise RuntimeError("baseline artifact lock protocol differs")
    verify_reference(lock["source_lock"])
    result = load_json(verify_reference(lock["baseline_result"]))
    if result["outcome"] != "baseline_compatible":
        raise RuntimeError("frozen propagated baseline is not compatible")
    expected = {row["path"] for row in lock["artifacts"]}
    observed = {path.relative_to(REPO_ROOT).as_posix() for path in _runtime_artifacts()}
    require_exact_inventory(observed, expected)
    for row in lock["artifacts"]:
        verify_reference(row)
    return lock


__all__ = [
    "parent_references",
    "reference",
    "require_exact_inventory",
    "sources",
    "validate_baseline_artifact_lock",
    "validate_source_lock",
    "verify_reference",
    "write_baseline_artifact_lock",
    "write_source_lock",
]
