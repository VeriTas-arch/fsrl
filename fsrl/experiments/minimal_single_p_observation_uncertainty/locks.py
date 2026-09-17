"""Fail-closed parent, source, model, and input locks for the M2 intervention."""

from __future__ import annotations

import subprocess

import numpy as np

from fsrl.experiments.training_strategy.batches import EpisodeBatch
from fsrl.experiments.training_strategy.evaluation import write_arrays
from fsrl.experiments.training_strategy.locks import reference, verify_reference
from fsrl.infra.git_provenance import git_blob_sha256
from fsrl.infra.provenance import file_sha256, load_json, write_json_exclusive
from fsrl.paths import REPO_ROOT

from .observations import CONDITIONS, encode, epsilon_for, observation_checks
from .protocol import (
    INPUTS,
    PROTOCOL,
    PROTOCOL_SHA256,
    QUALIFICATION,
    SOURCE_INPUT_LOCK,
    specification,
)


def _git_text(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def clean_commit() -> str:
    if _git_text("branch", "--show-current") != "dev":
        raise RuntimeError("M2 observation execution requires dev")
    if _git_text("status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("commit qualified M2 observation source before locking")
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
        (
            REPO_ROOT / "tests/experiments/minimal_single_p_observation_uncertainty"
        ).rglob("*.py")
    )
    paths += [PROTOCOL, REPO_ROOT / "pyproject.toml", REPO_ROOT / ".envrc"]
    return [reference(path) for path in sorted(set(paths))]


def _parent_path(name: str):
    row = specification()["parents"][name]
    path = REPO_ROOT / row["path"]
    if file_sha256(path) != row["sha256"]:
        raise RuntimeError(f"parent record changed: {name}")
    return path


def validate_parent() -> tuple[dict, dict, dict]:
    _parent_path("M2_protocol")
    model_lock = load_json(_parent_path("M2_model_lock"))
    generic = load_json(_parent_path("M2_generic_result"))
    _parent_path("M2_clean_Liu_result")
    _parent_path("behavior_contract")
    _parent_path("historical_observation_protocol")
    _parent_path("historical_observation_result")
    seeds = specification()["design"]["network_seeds"]
    if set(model_lock["runs"]) != {str(seed) for seed in seeds}:
        raise RuntimeError("parent model cohort differs")
    if set(generic["network_results"]) != {str(seed) for seed in seeds}:
        raise RuntimeError("parent generic cohort differs")
    for seed in seeds:
        verify_reference(model_lock["runs"][str(seed)]["files"]["model.pth"])
    source_lock = load_json(verify_reference(model_lock["source_lock"]))
    for panel in specification()["design"]["evaluation_panels"]:
        verify_reference(source_lock["panels"][str(panel)]["liu"])
    return model_lock, generic, source_lock


def _load(ref: dict) -> EpisodeBatch:
    with np.load(verify_reference(ref), allow_pickle=False) as raw:
        return EpisodeBatch({name: raw[name] for name in raw.files})


def _save(path, arrays: dict[str, np.ndarray]) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_arrays(path, arrays)
    return reference(path)


def _freeze_panel(panel: int, parent_ref: dict, sigma: float) -> dict:
    cpu = _load(parent_ref)
    seed = 982000 + 100 * panel
    epsilon = epsilon_for(cpu, seed)
    rows = {}
    batches = {}
    for condition in CONDITIONS:
        batches[condition] = encode(cpu, condition, sigma, epsilon)
        rows[condition] = _save(
            INPUTS / str(panel) / f"{condition}.npz",
            batches[condition].arrays,
        )
    checks = observation_checks(batches["clean"], batches["folded"], batches["noisy"])
    if not all(
        checks[name]
        for name in (
            "nonzero_displayed_relations",
            "absolute_amplitudes_equal",
            "folded_signs_restored",
        )
    ):
        raise RuntimeError("registered observation identities failed")
    if batches["clean"].fingerprint() != cpu.fingerprint():
        raise RuntimeError("clean observation input differs from parent")
    return {
        "parent": parent_ref,
        "noise_seed": seed,
        "epsilon": _save(INPUTS / str(panel) / "epsilon.npz", {"epsilon": epsilon}),
        "conditions": rows,
        "checks": checks,
    }


def write_source_input_lock() -> dict:
    commit = clean_commit()
    require_committed(QUALIFICATION)
    qualification = load_json(QUALIFICATION)
    current = sources()
    if not qualification["passed"] or qualification["sources"] != current:
        raise RuntimeError("qualification does not cover committed source")
    if INPUTS.exists():
        raise RuntimeError("M2 observation input root already exists")
    model_lock, generic, parent_source = validate_parent()
    sigma = specification()["observation"]["sigma"]
    inputs = {
        str(panel): _freeze_panel(
            panel, parent_source["panels"][str(panel)]["liu"], sigma
        )
        for panel in specification()["design"]["evaluation_panels"]
    }
    payload = {
        "schema_version": 1,
        "protocol_sha256": PROTOCOL_SHA256,
        "source_commit": commit,
        "sources": current,
        "qualification": reference(QUALIFICATION),
        "parent_model_lock": reference(_parent_path("M2_model_lock")),
        "parent_generic_result": reference(_parent_path("M2_generic_result")),
        "models": {
            seed: {
                "checkpoint": row["files"]["model.pth"],
                "tensor_hashes": row["metadata"]["final_model"],
            }
            for seed, row in model_lock["runs"].items()
        },
        "generic_categories": {
            seed: row["category"] for seed, row in generic["network_results"].items()
        },
        "inputs": inputs,
        "scientific_outcomes_exposed": False,
    }
    write_json_exclusive(SOURCE_INPUT_LOCK, payload)
    return {
        "source_commit": commit,
        "models": len(payload["models"]),
        "panels": len(inputs),
    }


def _validate_locked_sources(lock: dict) -> None:
    if lock["sources"] != sources():
        raise RuntimeError("M2 observation source inventory differs")
    for row in lock["sources"]:
        if (
            git_blob_sha256(REPO_ROOT, lock["source_commit"], row["path"])
            != row["sha256"]
        ):
            raise RuntimeError(f"source Git witness differs: {row['path']}")
    qualification = load_json(verify_reference(lock["qualification"]))
    if not qualification["passed"] or qualification["sources"] != lock["sources"]:
        raise RuntimeError("locked qualification differs")


def _validate_locked_models(lock: dict, model_lock: dict, generic: dict) -> None:
    for seed, row in lock["models"].items():
        verify_reference(row["checkpoint"])
        if row["tensor_hashes"] != model_lock["runs"][seed]["metadata"]["final_model"]:
            raise RuntimeError("locked parent model tensors differ")
        if (
            lock["generic_categories"][seed]
            != generic["network_results"][seed]["category"]
        ):
            raise RuntimeError("locked generic category differs")


def _validate_locked_inputs(lock: dict) -> None:
    for panel in lock["inputs"].values():
        verify_reference(panel["parent"])
        verify_reference(panel["epsilon"])
        for row in panel["conditions"].values():
            verify_reference(row)


def validate_source_input_lock() -> dict:
    require_committed(SOURCE_INPUT_LOCK)
    lock = load_json(SOURCE_INPUT_LOCK)
    if lock["protocol_sha256"] != PROTOCOL_SHA256:
        raise RuntimeError("M2 observation protocol lock differs")
    _validate_locked_sources(lock)
    model_lock, generic, _ = validate_parent()
    if lock["parent_model_lock"] != reference(_parent_path("M2_model_lock")) or lock[
        "parent_generic_result"
    ] != reference(_parent_path("M2_generic_result")):
        raise RuntimeError("locked parent identity differs")
    _validate_locked_models(lock, model_lock, generic)
    _validate_locked_inputs(lock)
    return lock


def load_condition(lock: dict, panel: int, condition: str) -> EpisodeBatch:
    if condition not in CONDITIONS:
        raise ValueError("unregistered observation condition")
    return _load(lock["inputs"][str(panel)]["conditions"][condition])


__all__ = [
    "load_condition",
    "reference",
    "sources",
    "validate_parent",
    "validate_source_input_lock",
    "write_source_input_lock",
]
