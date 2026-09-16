"""Source, input, and final-model locks for clean single-P development."""

from __future__ import annotations

import json
import subprocess

import numpy as np

from fsrl.experiments.memory_structure.inputs import liu_inputs
from fsrl.experiments.observation_uncertainty.inputs import attach_noise
from fsrl.experiments.training_strategy.evaluation import write_arrays
from fsrl.experiments.training_strategy.generic_validation import (
    validation_episodes,
    validation_groups,
)
from fsrl.experiments.training_strategy.locks import reference, verify_reference
from fsrl.experiments.write_cost.inputs import with_learned
from fsrl.infra.file_contracts import validate_run_manifest
from fsrl.infra.provenance import load_json, write_json_exclusive
from fsrl.paths import REPO_ROOT

from .protocol import (
    MODEL_LOCK,
    PROTOCOL,
    PROTOCOL_SHA256,
    QUALIFICATION,
    RUNS,
    SOURCE_LOCK,
    inherited_recipe,
    specification,
    training_directory,
)


def clean_commit() -> str:
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT):
        raise RuntimeError("commit the qualified clean single-P source before locking")
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def sources() -> list[dict]:
    paths = list((REPO_ROOT / "fsrl").rglob("*.py"))
    paths += list((REPO_ROOT / "tests/experiments/clean_single_p").rglob("*.py"))
    paths += [PROTOCOL, REPO_ROOT / "pyproject.toml", REPO_ROOT / ".envrc"]
    return [reference(path) for path in sorted(set(paths))]


def _save_input(path, arrays: dict[str, np.ndarray]) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_arrays(path, arrays)
    return reference(path)


def freeze_inputs() -> dict:
    root = RUNS / "inputs"
    if root.exists():
        raise RuntimeError("clean single-P frozen input root already exists")
    panels = {}
    for panel in specification()["design"]["evaluation_panels"]:
        recipe = inherited_recipe(panel)
        episodes = validation_episodes(recipe)
        inputs = {}
        for length, indices in validation_groups(episodes).items():
            cpu = with_learned(tuple(episodes[index] for index in indices))
            cpu.arrays["episode_indices"] = np.asarray(indices)
            cpu = attach_noise(cpu, 7310 + panel, length)
            inputs[f"test-{length}"] = _save_input(
                root / str(panel) / f"test-{length}.npz", cpu.arrays
            )
        _, liu = liu_inputs(recipe, 8)
        liu = attach_noise(liu, 7310 + panel, replays=(0,))
        inputs["liu-8"] = _save_input(root / str(panel) / "liu-8.npz", liu.arrays)
        panels[str(panel)] = {"inputs": inputs}
    return panels


def write_source_lock() -> dict:
    commit = clean_commit()
    qualification = load_json(QUALIFICATION)
    if not qualification["passed"] or qualification["sources"] != sources():
        raise RuntimeError("qualification does not cover the committed source")
    panels = freeze_inputs()
    payload = {
        "schema_version": 1,
        "protocol_sha256": PROTOCOL_SHA256,
        "source_commit": commit,
        "sources": sources(),
        "qualification": reference(QUALIFICATION),
        "task": inherited_recipe()["task"],
        "panels": panels,
    }
    write_json_exclusive(SOURCE_LOCK, payload)
    return {"source_commit": commit, "panels": len(panels)}


def validate_source_lock() -> dict:
    lock = load_json(SOURCE_LOCK)
    if lock["protocol_sha256"] != PROTOCOL_SHA256:
        raise RuntimeError("clean single-P source protocol differs")
    for row in lock["sources"]:
        verify_reference(row, commit=lock["source_commit"])
    verify_reference(lock["qualification"])
    for panel in lock["panels"].values():
        for row in panel["inputs"].values():
            verify_reference(row)
    return lock


def validate_training_run(seed: int, condition: str, arm: str) -> dict:
    directory = training_directory(seed, condition, arm)
    manifest = load_json(directory / "run.json")
    if (
        manifest["lifecycle_state"] != "complete"
        or not validate_run_manifest(directory / "run.json")["passed"]
    ):
        raise RuntimeError("clean single-P training run is incomplete")
    result = load_json(directory / "result.json")
    expected = (seed, condition, arm, PROTOCOL_SHA256)
    observed = tuple(
        result[key] for key in ("seed", "condition", "arm", "protocol_sha256")
    )
    if observed != expected:
        raise RuntimeError("clean single-P training identity differs")
    spec = specification()
    rows = [
        json.loads(line)
        for line in (directory / "train_log.jsonl").read_text().splitlines()
    ]
    if [row["step"] for row in rows] != list(range(spec["training"]["steps"])):
        raise RuntimeError("clean single-P training trajectory is incomplete")
    if any(
        value != spec["training"]["steps"]
        for value in result["optimizer_steps"].values()
    ):
        raise RuntimeError("clean single-P optimizer did not update every parameter")
    checkpoint = verify_reference(result["checkpoint"])
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload["condition"] != condition or payload["arm"] != arm:
        raise RuntimeError("clean single-P checkpoint identity differs")
    return result


def write_model_lock() -> dict:
    source, spec = validate_source_lock(), specification()
    runs = {}
    for seed in spec["design"]["network_seeds"]:
        streams = {}
        initial = {}
        for condition in spec["design"]["architecture_conditions"]:
            for arm in spec["design"]["observation_training_arms"]:
                identity = f"{seed}/{condition}/{arm}"
                metadata = validate_training_run(seed, condition, arm)
                files = {
                    path.name: reference(path)
                    for path in sorted(
                        training_directory(seed, condition, arm).iterdir()
                    )
                    if path.is_file()
                }
                runs[identity] = {"metadata": metadata, "files": files}
                streams[identity] = metadata["stream_fingerprint"]
                initial[identity] = metadata["initial_shadow"]
        if len(set(streams.values())) != 1:
            raise RuntimeError("paired clean single-P training streams differ")
        if len({json.dumps(value, sort_keys=True) for value in initial.values()}) != 1:
            raise RuntimeError("paired clean single-P shadow initializations differ")
    payload = {
        "schema_version": 1,
        "protocol_sha256": PROTOCOL_SHA256,
        "source_commit": source["source_commit"],
        "source_lock": reference(SOURCE_LOCK),
        "runs": runs,
        "scientific_outcomes_exposed": False,
    }
    write_json_exclusive(MODEL_LOCK, payload)
    return {"locked_models": len(runs), "source_commit": source["source_commit"]}


def validate_model_lock() -> tuple[dict, dict]:
    source, lock = validate_source_lock(), load_json(MODEL_LOCK)
    if lock["source_lock"] != reference(SOURCE_LOCK):
        raise RuntimeError("clean single-P model lock source differs")
    for row in lock["runs"].values():
        for file in row["files"].values():
            verify_reference(file)
    return source, lock


__all__ = [
    "reference",
    "sources",
    "validate_model_lock",
    "validate_source_lock",
    "validate_training_run",
    "write_model_lock",
    "write_source_lock",
]
