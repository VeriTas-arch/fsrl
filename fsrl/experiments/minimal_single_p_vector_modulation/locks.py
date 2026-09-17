"""Source, paired-input, and joint-model locks for vector modulation."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

from fsrl.experiments.clean_single_p.batches import SinglePEpisodeBatch
from fsrl.experiments.training_strategy.batches import EpisodeBatch
from fsrl.experiments.training_strategy.locks import reference, verify_reference
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
    SOURCE_LOCK,
    specification,
    training_directory,
)

PARENT_SOURCE = (
    REPO_ROOT / "studies/minimal_single_p_promotion/records/benchmarks/source_lock.json"
)
PARENT_MODELS = (
    REPO_ROOT / "studies/minimal_single_p_promotion/records/benchmarks/model_lock.json"
)
PARENT_GENERIC = (
    REPO_ROOT
    / "studies/minimal_single_p_promotion/records/results/minimal_single_p_promotion_v1.generic.json"
)
OBSERVATION_LOCK = (
    REPO_ROOT
    / "studies/minimal_single_p_observation_uncertainty/records/benchmarks/source_input_lock.json"
)
MORPHOLOGY_RESULT = (
    REPO_ROOT
    / "studies/minimal_single_p_pair_morphology_attribution/records/results/minimal_single_p_pair_morphology_attribution_v1.json"
)


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def clean_commit() -> str:
    if _git("branch", "--show-current") != "dev":
        raise RuntimeError("M2 vector-modulation execution requires dev")
    if _git("status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("commit qualified vector-modulation source before locking")
    return _git("rev-parse", "HEAD")


def require_committed(path: Path) -> str:
    commit = _git("rev-parse", "HEAD")
    relative = path.relative_to(REPO_ROOT).as_posix()
    if git_blob_sha256(REPO_ROOT, commit, relative) != file_sha256(path):
        raise RuntimeError(f"registered freeze point is not committed: {relative}")
    return commit


def sources() -> list[dict]:
    paths = list(
        (REPO_ROOT / "fsrl/experiments/minimal_single_p_vector_modulation").rglob(
            "*.py"
        )
    )
    paths += list(
        (REPO_ROOT / "tests/experiments/minimal_single_p_vector_modulation").rglob(
            "*.py"
        )
    )
    paths += [REPO_ROOT / "fsrl/infra/formal_runtime.py", PROTOCOL]
    return [reference(path) for path in sorted(set(paths))]


def _parent_liu_results(seeds: list[int], panels: list[int], conditions: list[str]):
    return {
        str(seed): {
            str(panel): {
                condition: reference(
                    REPO_ROOT
                    / "artifacts/runs/minimal_single_p_observation_uncertainty_v1"
                    / "evaluation"
                    / str(seed)
                    / str(panel)
                    / condition
                    / "result.json"
                )
                for condition in conditions
            }
            for panel in panels
        }
        for seed in seeds
    }


def write_source_lock() -> dict:
    commit = clean_commit()
    current = sources()
    qualification = load_json(QUALIFICATION)
    if not qualification["passed"] or qualification["sources"] != current:
        raise RuntimeError("qualification does not cover committed vector source")
    spec = specification()
    parent_source = load_json(PARENT_SOURCE)
    parent_models = load_json(PARENT_MODELS)
    observation = load_json(OBSERVATION_LOCK)
    seeds = spec["design"]["network_seeds"]
    payload = {
        "schema_version": 1,
        "protocol_sha256": PROTOCOL_SHA256,
        "source_commit": commit,
        "sources": current,
        "qualification": reference(QUALIFICATION),
        "parents": {
            "promotion_source_lock": reference(PARENT_SOURCE),
            "promotion_model_lock": reference(PARENT_MODELS),
            "promotion_generic_result": reference(PARENT_GENERIC),
            "observation_source_input_lock": reference(OBSERVATION_LOCK),
            "pair_morphology_result": reference(MORPHOLOGY_RESULT),
        },
        "task": parent_source["task"],
        "generic_inputs": {
            panel: values["generic"]
            for panel, values in parent_source["panels"].items()
        },
        "liu_inputs": {
            panel: values["conditions"]
            for panel, values in observation["inputs"].items()
        },
        "parent_runs": {
            str(seed): {
                "checkpoint": parent_models["runs"][str(seed)]["files"]["model.pth"],
                "result": parent_models["runs"][str(seed)]["files"]["result.json"],
                "train_log": parent_models["runs"][str(seed)]["files"][
                    "train_log.jsonl"
                ],
            }
            for seed in seeds
        },
        "parent_liu_results": _parent_liu_results(
            seeds,
            spec["design"]["liu_panels"],
            spec["design"]["liu_conditions"],
        ),
        "scientific_outcomes_exposed": False,
    }
    write_json_exclusive(SOURCE_LOCK, payload)
    return {"source_commit": commit, "paired_streams": len(payload["parent_runs"])}


def _verify_nested_references(value) -> None:
    if isinstance(value, dict) and {"path", "sha256", "bytes"} <= set(value):
        verify_reference(value)
    elif isinstance(value, dict):
        for child in value.values():
            _verify_nested_references(child)


def validate_source_lock() -> dict:
    require_committed(SOURCE_LOCK)
    lock = load_json(SOURCE_LOCK)
    if lock["protocol_sha256"] != PROTOCOL_SHA256 or lock["sources"] != sources():
        raise RuntimeError("M2 vector-modulation source lock differs")
    for row in lock["sources"]:
        if (
            git_blob_sha256(REPO_ROOT, lock["source_commit"], row["path"])
            != row["sha256"]
        ):
            raise RuntimeError(f"vector-modulation Git witness differs: {row['path']}")
    qualification = load_json(verify_reference(lock["qualification"]))
    if not qualification["passed"] or qualification["sources"] != lock["sources"]:
        raise RuntimeError("vector-modulation qualification differs")
    for group in (
        "parents",
        "generic_inputs",
        "liu_inputs",
        "parent_runs",
        "parent_liu_results",
    ):
        _verify_nested_references(lock[group])
    return lock


def load_generic(lock: dict, panel: int, name: str) -> SinglePEpisodeBatch:
    with np.load(
        verify_reference(lock["generic_inputs"][str(panel)][name]), allow_pickle=False
    ) as raw:
        return SinglePEpisodeBatch({key: raw[key] for key in raw.files})


def load_liu(lock: dict, panel: int, condition: str) -> EpisodeBatch:
    with np.load(
        verify_reference(lock["liu_inputs"][str(panel)][condition]), allow_pickle=False
    ) as raw:
        return EpisodeBatch({key: raw[key] for key in raw.files})


def validate_training(seed: int) -> dict:
    directory = training_directory(seed)
    if not validate_run_manifest(directory / "run.json")["passed"]:
        raise RuntimeError(f"incomplete vector-modulation run: {seed}")
    result = load_json(directory / "result.json")
    if result["seed"] != seed or result["protocol_sha256"] != PROTOCOL_SHA256:
        raise RuntimeError(f"vector-modulation training identity differs: {seed}")
    rows = [
        json.loads(line)
        for line in (directory / "train_log.jsonl").read_text().splitlines()
    ]
    if [row["step"] for row in rows] != list(range(1500)):
        raise RuntimeError(f"vector-modulation trajectory incomplete: {seed}")
    if result["parameters"] != 87002 or any(
        value != 1500 for value in result["optimizer_steps"].values()
    ):
        raise RuntimeError(f"vector-modulation optimizer integrity differs: {seed}")
    return result


def write_model_lock() -> dict:
    source = validate_source_lock()
    runs = {}
    for seed in specification()["design"]["network_seeds"]:
        metadata = validate_training(seed)
        directory = training_directory(seed)
        runs[str(seed)] = {
            "metadata": metadata,
            "files": {
                path.name: reference(path)
                for path in sorted(directory.iterdir())
                if path.is_file()
            },
        }
    payload = {
        "schema_version": 1,
        "protocol_sha256": PROTOCOL_SHA256,
        "source_lock": reference(SOURCE_LOCK),
        "runs": runs,
        "scientific_outcomes_exposed": False,
    }
    write_json_exclusive(MODEL_LOCK, payload)
    return {"locked_models": len(runs), "source_commit": source["source_commit"]}


def validate_model_lock() -> tuple[dict, dict]:
    source = validate_source_lock()
    require_committed(MODEL_LOCK)
    lock = load_json(MODEL_LOCK)
    expected = {str(seed) for seed in specification()["design"]["network_seeds"]}
    if lock["source_lock"] != reference(SOURCE_LOCK) or set(lock["runs"]) != expected:
        raise RuntimeError("M2 vector-modulation model lock differs")
    for seed, row in lock["runs"].items():
        for file in row["files"].values():
            verify_reference(file)
        if validate_training(int(seed)) != row["metadata"]:
            raise RuntimeError(f"locked vector metadata differs: {seed}")
    return source, lock


def require_generic_freeze() -> dict:
    validate_model_lock()
    require_committed(GENERIC_RESULT)
    result = load_json(GENERIC_RESULT)
    if result["protocol_sha256"] != PROTOCOL_SHA256 or len(result["networks"]) != 3:
        raise RuntimeError("M2 vector-modulation generic freeze differs")
    return result


__all__ = [
    "PARENT_GENERIC",
    "PARENT_MODELS",
    "PARENT_SOURCE",
    "load_generic",
    "load_liu",
    "reference",
    "require_generic_freeze",
    "sources",
    "validate_model_lock",
    "validate_source_lock",
    "validate_training",
    "verify_reference",
    "write_model_lock",
    "write_source_lock",
]
