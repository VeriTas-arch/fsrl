"""Prospective source, parent-input, replay, and model locks."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from fsrl.infra.file_contracts import validate_run_manifest
from fsrl.infra.git_provenance import git_blob_sha256
from fsrl.infra.provenance import file_sha256, load_json, write_json_exclusive
from fsrl.paths import REPO_ROOT

from .protocol import (
    GENERIC_RESULT,
    MODEL_LOCK,
    PROTOCOL,
    PROTOCOL_SHA256,
    QUALIFICATION,
    REPAIR,
    REPAIR2,
    REPAIR2_QUALIFICATION,
    REPAIR2_SHA256,
    REPAIR_QUALIFICATION,
    REPAIR_SHA256,
    SOURCE_INPUT_LOCK,
    SOURCE_REPAIR2_LOCK,
    SOURCE_REPAIR_LOCK,
    specification,
    training_directory,
)

PARENT_PROMOTION_LOCK = (
    REPO_ROOT / "studies/minimal_single_p_promotion/records/benchmarks/source_lock.json"
)
PARENT_MODEL_LOCK = (
    REPO_ROOT / "studies/minimal_single_p_promotion/records/benchmarks/model_lock.json"
)
PARENT_OBSERVATION_LOCK = REPO_ROOT / (
    "studies/minimal_single_p_observation_uncertainty/records/benchmarks/"
    "source_input_lock.json"
)


def reference(path: Path) -> dict:
    return {
        "path": path.relative_to(REPO_ROOT).as_posix(),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
    }


def verify_reference(row: dict) -> Path:
    path = REPO_ROOT / row["path"]
    if (
        not path.is_file()
        or path.stat().st_size != row["bytes"]
        or file_sha256(path) != row["sha256"]
    ):
        raise RuntimeError(f"locked reference differs: {row['path']}")
    return path


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def clean_commit() -> str:
    if _git("branch", "--show-current") != "dev":
        raise RuntimeError("aligned comparator execution requires dev")
    if _git("status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("commit qualified aligned-comparator source before locking")
    return _git("rev-parse", "HEAD")


def require_committed(path: Path) -> str:
    commit = _git("rev-parse", "HEAD")
    relative = path.relative_to(REPO_ROOT).as_posix()
    if git_blob_sha256(REPO_ROOT, commit, relative) != file_sha256(path):
        raise RuntimeError(f"registered freeze point is not committed: {relative}")
    return commit


def sources() -> list[dict]:
    package = REPO_ROOT / "fsrl/experiments/q_only_aligned_score"
    paths = list(package.rglob("*.py"))
    paths += list((REPO_ROOT / "tests/experiments/q_only_aligned_score").rglob("*.py"))
    paths += [REPO_ROOT / "fsrl/infra/formal_runtime.py", PROTOCOL, REPAIR, REPAIR2]
    return [reference(path) for path in sorted(set(paths))]


def _parent_references() -> dict:
    promotion = load_json(PARENT_PROMOTION_LOCK)
    observation = load_json(PARENT_OBSERVATION_LOCK)
    generic = {
        panel: {name: row for name, row in values["generic"].items()}
        for panel, values in promotion["panels"].items()
    }
    liu = {
        panel: {name: row for name, row in values["conditions"].items()}
        for panel, values in observation["inputs"].items()
    }
    training_logs = {
        str(seed): reference(
            REPO_ROOT
            / f"artifacts/runs/minimal_single_p_promotion_v1/training/{seed}/train_log.jsonl"
        )
        for seed in specification()["design"]["training_streams"]
    }
    return {
        "promotion_source_lock": reference(PARENT_PROMOTION_LOCK),
        "promotion_model_lock": reference(PARENT_MODEL_LOCK),
        "observation_source_input_lock": reference(PARENT_OBSERVATION_LOCK),
        "pair_morphology_result": reference(
            REPO_ROOT
            / "studies/minimal_single_p_pair_morphology_attribution/records/results/minimal_single_p_pair_morphology_attribution_v1.json"
        ),
        "generic_inputs": generic,
        "liu_inputs": liu,
        "training_logs": training_logs,
        "task": promotion["task"],
    }


def write_source_input_lock() -> dict:
    commit = clean_commit()
    qualification = load_json(QUALIFICATION)
    current = sources()
    if not qualification["passed"] or qualification["sources"] != current:
        raise RuntimeError("qualification does not cover committed comparator source")
    payload = {
        "schema_version": 1,
        "protocol_sha256": PROTOCOL_SHA256,
        "source_commit": commit,
        "sources": current,
        "qualification": reference(QUALIFICATION),
        "parents": _parent_references(),
        "replay": qualification["stream_replay"],
        "scientific_outcomes_exposed": False,
    }
    write_json_exclusive(SOURCE_INPUT_LOCK, payload)
    return {"source_commit": commit, "replayed_streams": len(payload["replay"])}


def validate_source_input_lock() -> dict:
    lock = load_json(SOURCE_INPUT_LOCK)
    if lock["protocol_sha256"] != PROTOCOL_SHA256:
        raise RuntimeError("source/input lock protocol differs")
    current = sources()
    active_sources = lock["sources"]
    active_commit = lock["source_commit"]
    if lock["sources"] != current:
        if SOURCE_REPAIR2_LOCK.exists():
            repair = load_json(SOURCE_REPAIR2_LOCK)
            if (
                repair["parent_source_repair_lock"] != reference(SOURCE_REPAIR_LOCK)
                or repair["repair"] != reference(REPAIR2)
                or file_sha256(REPAIR2) != REPAIR2_SHA256
                or repair["sources"] != current
                or repair["model_lock"] != reference(MODEL_LOCK)
                or repair["generic_result"] != reference(GENERIC_RESULT)
            ):
                raise RuntimeError("aligned comparator source repair2 lock differs")
            repair_qualification = load_json(verify_reference(repair["qualification"]))
            if (
                not repair_qualification["passed"]
                or repair_qualification["sources"] != current
            ):
                raise RuntimeError("aligned comparator repair2 qualification differs")
        elif SOURCE_REPAIR_LOCK.exists():
            repair = load_json(SOURCE_REPAIR_LOCK)
            if (
                repair["parent_source_input_lock"] != reference(SOURCE_INPUT_LOCK)
                or repair["repair"] != reference(REPAIR)
                or file_sha256(REPAIR) != REPAIR_SHA256
                or repair["sources"] != current
                or repair["model_lock"] != reference(MODEL_LOCK)
            ):
                raise RuntimeError("aligned comparator source repair lock differs")
            repair_qualification = load_json(verify_reference(repair["qualification"]))
            if (
                not repair_qualification["passed"]
                or repair_qualification["sources"] != current
            ):
                raise RuntimeError("aligned comparator repair qualification differs")
        else:
            raise RuntimeError("aligned comparator source changed after locking")
        active_sources = repair["sources"]
        active_commit = repair["source_commit"]
    for row in active_sources:
        if git_blob_sha256(REPO_ROOT, active_commit, row["path"]) != row["sha256"]:
            raise RuntimeError(f"source Git witness differs: {row['path']}")
    qualification = load_json(verify_reference(lock["qualification"]))
    if not qualification["passed"] or qualification["sources"] != lock["sources"]:
        raise RuntimeError("locked qualification differs")
    parents = lock["parents"]
    for key in (
        "promotion_source_lock",
        "promotion_model_lock",
        "observation_source_input_lock",
        "pair_morphology_result",
    ):
        verify_reference(parents[key])
    for group in ("generic_inputs", "liu_inputs"):
        for panel in parents[group].values():
            for row in panel.values():
                verify_reference(row)
    for row in parents["training_logs"].values():
        verify_reference(row)
    return lock


def write_source_repair_lock() -> dict:
    if file_sha256(REPAIR) != REPAIR_SHA256:
        raise RuntimeError("aligned comparator repair contract changed")
    commit = clean_commit()
    current = sources()
    qualification = load_json(REPAIR_QUALIFICATION)
    if not qualification["passed"] or qualification["sources"] != current:
        raise RuntimeError("repair qualification does not cover committed source")
    payload = {
        "schema_version": 1,
        "protocol_sha256": PROTOCOL_SHA256,
        "parent_source_input_lock": reference(SOURCE_INPUT_LOCK),
        "repair": reference(REPAIR),
        "qualification": reference(REPAIR_QUALIFICATION),
        "model_lock": reference(MODEL_LOCK),
        "source_commit": commit,
        "sources": current,
        "scientific_outcomes_exposed": False,
        "model_lock_unchanged": True,
    }
    write_json_exclusive(SOURCE_REPAIR_LOCK, payload)
    return {"source_commit": commit, "repair": REPAIR.name}


def write_source_repair2_lock() -> dict:
    if file_sha256(REPAIR2) != REPAIR2_SHA256:
        raise RuntimeError("aligned comparator repair2 contract changed")
    require_committed(GENERIC_RESULT)
    commit = clean_commit()
    current = sources()
    qualification = load_json(REPAIR2_QUALIFICATION)
    if not qualification["passed"] or qualification["sources"] != current:
        raise RuntimeError("repair2 qualification does not cover committed source")
    payload = {
        "schema_version": 1,
        "protocol_sha256": PROTOCOL_SHA256,
        "parent_source_repair_lock": reference(SOURCE_REPAIR_LOCK),
        "repair": reference(REPAIR2),
        "qualification": reference(REPAIR2_QUALIFICATION),
        "model_lock": reference(MODEL_LOCK),
        "generic_result": reference(GENERIC_RESULT),
        "source_commit": commit,
        "sources": current,
        "generic_outcomes_exposed": True,
        "liu_outcomes_exposed": False,
        "model_lock_unchanged": True,
    }
    write_json_exclusive(SOURCE_REPAIR2_LOCK, payload)
    return {"source_commit": commit, "repair": REPAIR2.name}


def validate_training(seed: int) -> dict:
    directory = training_directory(seed)
    manifest = directory / "run.json"
    if not validate_run_manifest(manifest)["passed"]:
        raise RuntimeError(f"incomplete aligned-score run: {seed}")
    result = load_json(directory / "result.json")
    if (
        result["seed"] != seed
        or result["protocol_sha256"] != PROTOCOL_SHA256
        or result["optimizer_steps"] != {"raw_eta": 1500, "raw_gamma": 1500}
        or result["parameters"] != 2
    ):
        raise RuntimeError(f"aligned-score training identity differs: {seed}")
    return result


def write_model_lock() -> dict:
    source = validate_source_input_lock()
    runs = {}
    for seed in specification()["design"]["training_streams"]:
        result = validate_training(seed)
        directory = training_directory(seed)
        runs[str(seed)] = {
            "model": reference(directory / "model.pth"),
            "result": reference(directory / "result.json"),
            "run": reference(directory / "run.json"),
            "train_log": reference(directory / "train_log.jsonl"),
            "metadata": result,
        }
    payload = {
        "schema_version": 1,
        "protocol_sha256": PROTOCOL_SHA256,
        "source_input_lock": reference(SOURCE_INPUT_LOCK),
        "runs": runs,
        "scientific_outcomes_exposed": False,
    }
    write_json_exclusive(MODEL_LOCK, payload)
    return {"locked_runs": len(runs), "source_commit": source["source_commit"]}


def validate_model_lock() -> tuple[dict, dict]:
    source = validate_source_input_lock()
    lock = load_json(MODEL_LOCK)
    if lock["protocol_sha256"] != PROTOCOL_SHA256 or lock[
        "source_input_lock"
    ] != reference(SOURCE_INPUT_LOCK):
        raise RuntimeError("aligned-score model lock differs")
    expected = {str(seed) for seed in specification()["design"]["training_streams"]}
    if set(lock["runs"]) != expected:
        raise RuntimeError("aligned-score model cohort differs")
    for seed, row in lock["runs"].items():
        for name in ("model", "result", "run", "train_log"):
            verify_reference(row[name])
        if validate_training(int(seed)) != row["metadata"]:
            raise RuntimeError(f"aligned-score locked metadata differs: {seed}")
    return source, lock


def load_npz(row: dict) -> dict[str, np.ndarray]:
    with np.load(verify_reference(row), allow_pickle=False) as raw:
        return {key: raw[key] for key in raw.files}


__all__ = [
    "clean_commit",
    "load_npz",
    "reference",
    "require_committed",
    "sources",
    "validate_model_lock",
    "validate_source_input_lock",
    "validate_training",
    "verify_reference",
    "write_model_lock",
    "write_source_input_lock",
    "write_source_repair2_lock",
    "write_source_repair_lock",
]
