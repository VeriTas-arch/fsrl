"""Source, frozen-input, and joint-model locks for the promotion study."""

from __future__ import annotations

import json
import subprocess

import numpy as np
import torch

from fsrl.experiments.clean_single_p.batches import (
    SinglePEpisodeBatch,
    prepare_single_p,
)
from fsrl.experiments.memory_structure.inputs import liu_inputs
from fsrl.experiments.single_p_anytime.streams import learned_mask
from fsrl.experiments.training_strategy.batches import EpisodeBatch
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

from .protocol import (
    GENERIC_RESULT,
    INPUTS,
    MODEL_LOCK,
    PROTOCOL,
    PROTOCOL_SHA256,
    QUALIFICATION,
    REPAIR,
    REPAIR_QUALIFICATION,
    REPAIR_SHA256,
    SOURCE_LOCK,
    SOURCE_REPAIR_LOCK,
    inherited_recipe,
    specification,
    training_directory,
)


def _git_text(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def clean_commit() -> str:
    if _git_text("branch", "--show-current") != "dev":
        raise RuntimeError("promotion execution requires dev")
    if _git_text("status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("commit the qualified promotion source before locking")
    return _git_text("rev-parse", "HEAD")


def require_committed(path) -> str:
    relative = path.relative_to(REPO_ROOT).as_posix()
    commit = _git_text("rev-parse", "HEAD")
    if git_blob_sha256(REPO_ROOT, commit, relative) != file_sha256(path):
        raise RuntimeError(f"registered freeze point is not committed: {relative}")
    return commit


def sources() -> list[dict]:
    paths = list((REPO_ROOT / "fsrl").rglob("*.py"))
    paths += list(
        (REPO_ROOT / "tests/experiments/minimal_single_p_promotion").rglob("*.py")
    )
    paths += [PROTOCOL, REPAIR, REPO_ROOT / "pyproject.toml", REPO_ROOT / ".envrc"]
    return [reference(path) for path in sorted(set(paths))]


def _save(path, arrays: dict[str, np.ndarray]) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_arrays(path, arrays)
    return reference(path)


def _freeze_inputs() -> dict:
    panels = {}
    for panel in specification()["design"]["evaluation_panels"]:
        recipe = inherited_recipe(panel)
        episodes = validation_episodes(recipe)
        generic = {}
        for length, indices in sorted(validation_groups(episodes).items()):
            selected = tuple(episodes[index] for index in indices)
            cpu = prepare_single_p(
                selected,
                "clean",
                observation_seed=962000 + panel * 100 + length,
            )
            arrays = {name: value.copy() for name, value in cpu.arrays.items()}
            arrays["learned"] = learned_mask(cpu).T
            arrays["episode_indices"] = np.asarray(indices, dtype=np.int64)
            generic[f"test-{length}"] = _save(
                INPUTS / str(panel) / f"generic-{length}.npz", arrays
            )
        _, liu = liu_inputs(recipe, 8)
        panels[str(panel)] = {
            "generic_rng_seed": recipe["evaluation"]["generic"]["rng_seed"],
            "liu_seeds": {
                key: recipe["evaluation"]["liu"][key]
                for key in (
                    "cue_seed",
                    "support_seed",
                    "subject_encoding_seed",
                    "choice_seed",
                    "order_seed",
                    "query_shuffle_seed",
                    "evidence_shuffle_seed",
                )
            },
            "generic": generic,
            "liu": _save(
                INPUTS / str(panel) / "liu-8.npz",
                {name: value.copy() for name, value in liu.arrays.items()},
            ),
        }
    return panels


def write_source_lock() -> dict:
    commit = clean_commit()
    qualification = load_json(QUALIFICATION)
    current = sources()
    if not qualification["passed"] or qualification["sources"] != current:
        raise RuntimeError("qualification does not cover committed promotion source")
    if INPUTS.exists():
        raise RuntimeError("promotion frozen input root already exists")
    payload = {
        "schema_version": 1,
        "protocol_sha256": PROTOCOL_SHA256,
        "source_commit": commit,
        "sources": current,
        "qualification": reference(QUALIFICATION),
        "task": inherited_recipe(1)["task"],
        "panels": _freeze_inputs(),
        "scientific_outcomes_exposed": False,
    }
    write_json_exclusive(SOURCE_LOCK, payload)
    return {"source_commit": commit, "panels": len(payload["panels"])}


def _historical_source_lock() -> dict:
    require_committed(SOURCE_LOCK)
    lock = load_json(SOURCE_LOCK)
    if lock["protocol_sha256"] != PROTOCOL_SHA256:
        raise RuntimeError("promotion source protocol differs")
    for row in lock["sources"]:
        if (
            git_blob_sha256(REPO_ROOT, lock["source_commit"], row["path"])
            != row["sha256"]
        ):
            raise RuntimeError(f"promotion Git witness differs: {row['path']}")
    qualification = load_json(verify_reference(lock["qualification"]))
    if not qualification["passed"] or qualification["sources"] != lock["sources"]:
        raise RuntimeError("promotion qualification differs")
    for panel in lock["panels"].values():
        for row in panel["generic"].values():
            verify_reference(row)
        verify_reference(panel["liu"])
    return lock


def validate_source_lock() -> dict:
    lock = _historical_source_lock()
    current = sources()
    if not SOURCE_REPAIR_LOCK.exists():
        if lock["sources"] != current:
            raise RuntimeError("promotion source changed without a repair lock")
        return lock
    require_committed(SOURCE_REPAIR_LOCK)
    repair = load_json(SOURCE_REPAIR_LOCK)
    if (
        repair["parent_source_lock"] != reference(SOURCE_LOCK)
        or repair["repair"] != reference(REPAIR)
        or file_sha256(REPAIR) != REPAIR_SHA256
        or repair["sources"] != current
    ):
        raise RuntimeError("promotion reporting repair lock differs")
    for row in current:
        if (
            git_blob_sha256(REPO_ROOT, repair["source_commit"], row["path"])
            != row["sha256"]
        ):
            raise RuntimeError(f"promotion repair Git witness differs: {row['path']}")
    qualification = load_json(verify_reference(repair["qualification"]))
    if not qualification["passed"] or qualification["sources"] != current:
        raise RuntimeError("promotion repair qualification differs")
    return lock


def write_source_repair_lock() -> dict:
    parent = _historical_source_lock()
    commit = clean_commit()
    current = sources()
    qualification = load_json(REPAIR_QUALIFICATION)
    if file_sha256(REPAIR) != REPAIR_SHA256:
        raise RuntimeError("promotion repair contract changed")
    if not qualification["passed"] or qualification["sources"] != current:
        raise RuntimeError("repair qualification does not cover committed source")
    payload = {
        "schema_version": 1,
        "protocol_sha256": PROTOCOL_SHA256,
        "parent_source_lock": reference(SOURCE_LOCK),
        "repair": reference(REPAIR),
        "source_commit": commit,
        "sources": current,
        "qualification": reference(REPAIR_QUALIFICATION),
        "locked_evaluation_artifacts_unchanged": True,
        "scientific_outcomes_exposed": True,
        "parent_source_commit": parent["source_commit"],
    }
    write_json_exclusive(SOURCE_REPAIR_LOCK, payload)
    return {"source_commit": commit, "repair": REPAIR.name}


def load_generic(source: dict, panel: int, name: str) -> SinglePEpisodeBatch:
    with np.load(
        verify_reference(source["panels"][str(panel)]["generic"][name]),
        allow_pickle=False,
    ) as raw:
        return SinglePEpisodeBatch({key: raw[key] for key in raw.files})


def load_liu(source: dict, panel: int) -> EpisodeBatch:
    with np.load(
        verify_reference(source["panels"][str(panel)]["liu"]), allow_pickle=False
    ) as raw:
        return EpisodeBatch({key: raw[key] for key in raw.files})


def validate_training_run(seed: int) -> dict:
    directory = training_directory(seed)
    manifest = load_json(directory / "run.json")
    if (
        manifest["lifecycle_state"] != "complete"
        or not validate_run_manifest(directory / "run.json")["passed"]
    ):
        raise RuntimeError("promotion training run is incomplete")
    result = load_json(directory / "result.json")
    expected = (seed, PROTOCOL_SHA256)
    if tuple(result[key] for key in ("seed", "protocol_sha256")) != expected:
        raise RuntimeError("promotion training identity differs")
    rows = [
        json.loads(line)
        for line in (directory / "train_log.jsonl").read_text().splitlines()
    ]
    updates = specification()["training"]["updates"]
    if [row["step"] for row in rows] != list(range(updates)):
        raise RuntimeError("promotion training trajectory is incomplete")
    if any(value != updates for value in result["optimizer_steps"].values()):
        raise RuntimeError("promotion optimizer did not update every parameter")
    if result["parameters"] != specification()["architecture"]["parameter_count"]:
        raise RuntimeError("promotion parameter count differs")
    if not np.isfinite(result["final_eta"]):
        raise RuntimeError("promotion eta is nonfinite")
    payload = torch.load(
        verify_reference(result["checkpoint"]), map_location="cpu", weights_only=True
    )
    if payload["level"] != "M2" or payload["seed"] != seed:
        raise RuntimeError("promotion checkpoint identity differs")
    return result


def write_model_lock() -> dict:
    source = validate_source_lock()
    runs = {}
    for seed in specification()["design"]["network_seeds"]:
        metadata = validate_training_run(seed)
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
        "source_commit": source["source_commit"],
        "source_lock": reference(SOURCE_LOCK),
        "runs": runs,
        "scientific_outcomes_exposed": False,
    }
    write_json_exclusive(MODEL_LOCK, payload)
    return {"locked_models": len(runs)}


def validate_model_lock() -> tuple[dict, dict]:
    source = validate_source_lock()
    require_committed(MODEL_LOCK)
    lock = load_json(MODEL_LOCK)
    seeds = specification()["design"]["network_seeds"]
    if lock["source_lock"] != reference(SOURCE_LOCK) or set(lock["runs"]) != {
        str(seed) for seed in seeds
    }:
        raise RuntimeError("promotion joint model lock differs")
    for row in lock["runs"].values():
        for file in row["files"].values():
            verify_reference(file)
    return source, lock


def require_generic_freeze() -> dict:
    validate_model_lock()
    require_committed(GENERIC_RESULT)
    result = load_json(GENERIC_RESULT)
    if result["protocol_sha256"] != PROTOCOL_SHA256 or result["networks"] != 20:
        raise RuntimeError("generic distribution freeze differs")
    return result


__all__ = [
    "load_generic",
    "load_liu",
    "reference",
    "require_generic_freeze",
    "sources",
    "validate_model_lock",
    "validate_source_lock",
    "validate_training_run",
    "write_model_lock",
    "write_source_lock",
    "write_source_repair_lock",
]
