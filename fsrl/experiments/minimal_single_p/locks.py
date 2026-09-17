"""Source, generic-input, model, and progression locks for the ladder."""

from __future__ import annotations

import json
import subprocess

import numpy as np
import torch

from fsrl.experiments.clean_single_p.batches import (
    SinglePEpisodeBatch,
    prepare_single_p,
)
from fsrl.experiments.clean_single_p.protocol import inherited_recipe
from fsrl.experiments.single_p_anytime.streams import learned_mask
from fsrl.experiments.training_strategy.evaluation import write_arrays
from fsrl.experiments.training_strategy.generic_validation import (
    validation_episodes,
    validation_groups,
)
from fsrl.experiments.training_strategy.locks import reference, verify_reference
from fsrl.infra.file_contracts import validate_run_manifest
from fsrl.infra.git_provenance import git_blob_sha256
from fsrl.infra.provenance import file_sha256, load_json, write_json_exclusive
from fsrl.paths import REPO_ROOT

from .decisions import successor
from .model import level_settings
from .protocol import (
    INPUTS,
    PROTOCOL,
    PROTOCOL_SHA256,
    QUALIFICATION,
    SOURCE_LOCK,
    model_lock_path,
    predecessor,
    result_path,
    specification,
    training_directory,
    validate_level,
)


def _git_text(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def clean_commit() -> str:
    if _git_text("branch", "--show-current") != "dev":
        raise RuntimeError("minimal single-P execution requires dev")
    if _git_text("status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError(
            "commit the qualified minimal single-P source before locking"
        )
    return _git_text("rev-parse", "HEAD")


def require_committed(path) -> str:
    relative = path.relative_to(REPO_ROOT).as_posix()
    commit = _git_text("rev-parse", "HEAD")
    if git_blob_sha256(REPO_ROOT, commit, relative) != file_sha256(path):
        raise RuntimeError(f"registered freeze point is not committed: {relative}")
    return commit


def sources() -> list[dict]:
    paths = list((REPO_ROOT / "fsrl").rglob("*.py"))
    paths += list((REPO_ROOT / "tests/experiments/minimal_single_p").rglob("*.py"))
    paths += [PROTOCOL, REPO_ROOT / "pyproject.toml", REPO_ROOT / ".envrc"]
    return [reference(path) for path in sorted(set(paths))]


def _save_input(path, arrays: dict[str, np.ndarray]) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_arrays(path, arrays)
    return reference(path)


def _freeze_generic_inputs() -> dict:
    panels = {}
    for panel in specification()["design"]["evaluation_panels"]:
        recipe = inherited_recipe(1)
        recipe["evaluation"]["generic"]["rng_seed"] = 791000 + panel * 100
        episodes = validation_episodes(recipe)
        inputs = {}
        for length, indices in sorted(validation_groups(episodes).items()):
            selected = tuple(episodes[index] for index in indices)
            cpu = prepare_single_p(
                selected,
                "clean",
                observation_seed=891000 + panel * 100 + length,
            )
            arrays = {name: value.copy() for name, value in cpu.arrays.items()}
            arrays["learned"] = learned_mask(cpu).T
            arrays["episode_indices"] = np.asarray(indices, dtype=np.int64)
            inputs[f"test-{length}"] = _save_input(
                INPUTS / str(panel) / f"test-{length}.npz", arrays
            )
        panels[str(panel)] = {
            "rng_seed": recipe["evaluation"]["generic"]["rng_seed"],
            "inputs": inputs,
        }
    return panels


def write_source_lock() -> dict:
    commit = clean_commit()
    qualification = load_json(QUALIFICATION)
    current = sources()
    if not qualification["passed"] or qualification["sources"] != current:
        raise RuntimeError("qualification does not cover committed minimal source")
    if INPUTS.exists():
        raise RuntimeError("minimal single-P frozen input root already exists")
    payload = {
        "schema_version": 1,
        "protocol_sha256": PROTOCOL_SHA256,
        "source_commit": commit,
        "sources": current,
        "qualification": reference(QUALIFICATION),
        "task": inherited_recipe(1)["task"],
        "panels": _freeze_generic_inputs(),
        "scientific_outcomes_exposed": False,
    }
    write_json_exclusive(SOURCE_LOCK, payload)
    return {"source_commit": commit, "panels": len(payload["panels"])}


def validate_source_lock() -> dict:
    require_committed(SOURCE_LOCK)
    lock = load_json(SOURCE_LOCK)
    if lock["protocol_sha256"] != PROTOCOL_SHA256:
        raise RuntimeError("minimal single-P source protocol differs")
    current = sources()
    if lock["sources"] != current:
        raise RuntimeError("minimal single-P implementation changed after source lock")
    for row in current:
        if (
            git_blob_sha256(REPO_ROOT, lock["source_commit"], row["path"])
            != row["sha256"]
        ):
            raise RuntimeError(f"minimal single-P Git witness differs: {row['path']}")
    qualification = load_json(verify_reference(lock["qualification"]))
    if not qualification["passed"] or qualification["sources"] != current:
        raise RuntimeError("minimal single-P qualification differs")
    for panel in lock["panels"].values():
        for row in panel["inputs"].values():
            verify_reference(row)
    return lock


def load_panel_input(source: dict, panel: int, name: str) -> SinglePEpisodeBatch:
    record = source["panels"][str(panel)]["inputs"][name]
    with np.load(verify_reference(record), allow_pickle=False) as raw:
        return SinglePEpisodeBatch({key: raw[key] for key in raw.files})


def validate_predecessor(level: str) -> None:
    previous = predecessor(validate_level(level))
    if previous is None:
        return
    path = result_path(previous)
    require_committed(path)
    result = load_json(path)
    if (
        result["outcome"] != "clear_continue"
        or successor(previous, result["outcome"]) != level
    ):
        raise RuntimeError(f"registered stop rule does not authorize {level}")


def validate_training_run(seed: int, level: str) -> dict:
    level = validate_level(level)
    directory = training_directory(seed, level)
    manifest = load_json(directory / "run.json")
    if (
        manifest["lifecycle_state"] != "complete"
        or not validate_run_manifest(directory / "run.json")["passed"]
    ):
        raise RuntimeError("minimal single-P training run is incomplete")
    result = load_json(directory / "result.json")
    expected = (seed, level, PROTOCOL_SHA256)
    observed = tuple(result[key] for key in ("seed", "level", "protocol_sha256"))
    if observed != expected:
        raise RuntimeError("minimal single-P training identity differs")
    spec = specification()
    rows = [
        json.loads(line)
        for line in (directory / "train_log.jsonl").read_text().splitlines()
    ]
    if [row["step"] for row in rows] != list(range(spec["training"]["updates"])):
        raise RuntimeError("minimal single-P training trajectory is incomplete")
    if any(
        value != spec["training"]["updates"]
        for value in result["optimizer_steps"].values()
    ):
        raise RuntimeError("minimal single-P optimizer did not update every parameter")
    settings = level_settings(level)
    if result["parameters"] != settings["parameter_count"]:
        raise RuntimeError("minimal single-P parameter count differs")
    checkpoint = verify_reference(result["checkpoint"])
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload["level"] != level or payload["seed"] != seed:
        raise RuntimeError("minimal single-P checkpoint identity differs")
    return result


def write_model_lock(level: str) -> dict:
    level = validate_level(level)
    source = validate_source_lock()
    validate_predecessor(level)
    runs = {}
    for seed in specification()["design"]["network_seeds"]:
        metadata = validate_training_run(seed, level)
        directory = training_directory(seed, level)
        runs[str(seed)] = {
            "metadata": metadata,
            "files": {
                path.name: reference(path)
                for path in sorted(directory.iterdir())
                if path.is_file()
            },
        }
        previous = predecessor(level)
        if previous is not None and training_directory(seed, previous).is_dir():
            old = validate_training_run(seed, previous)
            if old["stream_fingerprint"] != metadata["stream_fingerprint"]:
                raise RuntimeError("minimal single-P paired training streams differ")
            if level == "M1" and old["initial_model"] != metadata["initial_model"]:
                raise RuntimeError("C0 and M1 initial models differ")
            if (
                level in {"M2", "M3"}
                and old["initial_shared"] != metadata["initial_shared"]
            ):
                raise RuntimeError("shared H=200 initialization differs")
    payload = {
        "schema_version": 1,
        "protocol_sha256": PROTOCOL_SHA256,
        "source_commit": source["source_commit"],
        "source_lock": reference(SOURCE_LOCK),
        "level": level,
        "runs": runs,
        "scientific_outcomes_exposed": False,
    }
    path = model_lock_path(level)
    write_json_exclusive(path, payload)
    return {"level": level, "locked_models": len(runs)}


def validate_model_lock(level: str) -> tuple[dict, dict]:
    level = validate_level(level)
    source = validate_source_lock()
    path = model_lock_path(level)
    require_committed(path)
    lock = load_json(path)
    if lock["level"] != level or lock["source_lock"] != reference(SOURCE_LOCK):
        raise RuntimeError("minimal single-P model lock identity differs")
    if len(lock["runs"]) != len(specification()["design"]["network_seeds"]):
        raise RuntimeError("minimal single-P model lock is incomplete")
    for row in lock["runs"].values():
        for file in row["files"].values():
            verify_reference(file)
    return source, lock


__all__ = [
    "load_panel_input",
    "reference",
    "sources",
    "validate_model_lock",
    "validate_predecessor",
    "validate_source_lock",
    "validate_training_run",
    "write_model_lock",
    "write_source_lock",
]
