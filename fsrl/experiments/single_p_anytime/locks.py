"""Source, input, schedule, and model locks for single-P anytime V1."""

from __future__ import annotations

import json
import subprocess

import numpy as np
import torch

from fsrl.experiments.clean_single_p.batches import prepare_single_p
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
from fsrl.infra.git_provenance import git_blob_sha256
from fsrl.infra.provenance import load_json, write_json_exclusive
from fsrl.paths import REPO_ROOT

from .protocol import (
    MODEL_LOCK,
    PROTOCOL,
    PROTOCOL_SHA256,
    QUALIFICATION,
    RUNS,
    SOURCE_LOCK,
    historical_recipe,
    specification,
    training_directory,
)
from .schedule import build_schedule, schedule_sha256, validate_schedule
from .streams import (
    learned_mask,
    make_generator,
    sample_fixed_edge_episodes,
)


def clean_commit() -> str:
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT):
        raise RuntimeError("commit qualified single-P anytime source before locking")
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def sources() -> list[dict]:
    paths = list((REPO_ROOT / "fsrl").rglob("*.py"))
    paths += list((REPO_ROOT / "tests/experiments/single_p_anytime").rglob("*.py"))
    paths += [PROTOCOL, REPO_ROOT / "pyproject.toml", REPO_ROOT / ".envrc"]
    return [reference(path) for path in sorted(set(paths))]


def _save_input(path, arrays: dict[str, np.ndarray]) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_arrays(path, arrays)
    return reference(path)


def _freeze_schedules(root) -> dict:
    records = {}
    for seed in specification()["design"]["network_seeds"]:
        schedule = build_schedule(seed)
        summary = validate_schedule(schedule)
        record = _save_input(
            root / "schedules" / f"seed-{seed}.npz", {"schedule": schedule}
        )
        records[str(seed)] = {
            "seed": 620000 + seed,
            "sha256": schedule_sha256(schedule),
            "summary": summary,
            "file": record,
        }
    return records


def _freeze_historical_inputs(root) -> dict:
    panels = {}
    for panel in specification()["design"]["evaluation_panels"]:
        recipe = historical_recipe(panel)
        episodes = validation_episodes(recipe)
        inputs = {}
        for length, indices in validation_groups(episodes).items():
            cpu = with_learned(tuple(episodes[index] for index in indices))
            cpu.arrays["episode_indices"] = np.asarray(indices)
            cpu = attach_noise(cpu, 7510 + panel, length)
            inputs[f"test-{length}"] = _save_input(
                root / "historical" / str(panel) / f"test-{length}.npz",
                cpu.arrays,
            )
        _, liu = liu_inputs(recipe, 8)
        liu = attach_noise(liu, 7510 + panel, replays=(0,))
        inputs["liu-8"] = _save_input(
            root / "historical" / str(panel) / "liu-8.npz", liu.arrays
        )
        panels[str(panel)] = {"inputs": inputs}
    return panels


def _freeze_anytime_inputs(root, task: dict) -> dict:
    panels = {}
    generator = make_generator(task, support_blocks=7)
    for panel in specification()["design"]["evaluation_panels"]:
        inputs = {}
        for edge_count in (7, 8, 9, 10):
            episodes = sample_fixed_edge_episodes(
                generator,
                seed=760000 + panel * 100 + edge_count,
                edge_count=edge_count,
                batch_size=64,
                validation=True,
            )
            clean = prepare_single_p(
                episodes,
                "clean",
                observation_seed=770000 + panel * 100 + edge_count,
            )
            noisy = prepare_single_p(
                episodes,
                "noisy",
                observation_seed=770000 + panel * 100 + edge_count,
            )
            mask = learned_mask(clean)
            for arm, cpu in (("clean", clean), ("noisy", noisy)):
                arrays = {name: value.copy() for name, value in cpu.arrays.items()}
                arrays["learned"] = mask
                inputs[f"E{edge_count}-{arm}"] = _save_input(
                    root / "anytime" / str(panel) / f"E{edge_count}-{arm}.npz",
                    arrays,
                )
        panels[str(panel)] = {"inputs": inputs}
    return panels


def write_source_lock() -> dict:
    commit = clean_commit()
    qualification = load_json(QUALIFICATION)
    current_sources = sources()
    if not qualification["passed"] or qualification["sources"] != current_sources:
        raise RuntimeError("qualification does not cover committed anytime source")
    root = RUNS / "inputs"
    if root.exists():
        raise RuntimeError("single-P anytime frozen input root already exists")
    task = historical_recipe(1)["task"]
    payload = {
        "schema_version": 1,
        "protocol_sha256": PROTOCOL_SHA256,
        "source_commit": commit,
        "sources": current_sources,
        "qualification": reference(QUALIFICATION),
        "task": task,
        "schedules": _freeze_schedules(root),
        "historical_panels": _freeze_historical_inputs(root),
        "anytime_panels": _freeze_anytime_inputs(root, task),
        "scientific_outcomes_exposed": False,
    }
    write_json_exclusive(SOURCE_LOCK, payload)
    return {
        "source_commit": commit,
        "schedules": len(payload["schedules"]),
        "historical_panels": len(payload["historical_panels"]),
        "anytime_panels": len(payload["anytime_panels"]),
    }


def _verify_input_tree(lock: dict) -> None:
    for row in lock["schedules"].values():
        verify_reference(row["file"])
    for family in ("historical_panels", "anytime_panels"):
        for panel in lock[family].values():
            for row in panel["inputs"].values():
                verify_reference(row)


def validate_source_lock() -> dict:
    lock = load_json(SOURCE_LOCK)
    if lock["protocol_sha256"] != PROTOCOL_SHA256:
        raise RuntimeError("single-P anytime source protocol differs")
    current = sources()
    if lock["sources"] != current:
        raise RuntimeError("single-P anytime implementation changed after source lock")
    for row in current:
        if (
            git_blob_sha256(REPO_ROOT, lock["source_commit"], row["path"])
            != row["sha256"]
        ):
            raise RuntimeError(f"single-P anytime Git witness differs: {row['path']}")
    qualification = load_json(verify_reference(lock["qualification"]))
    if not qualification["passed"] or qualification["sources"] != current:
        raise RuntimeError("single-P anytime qualification differs")
    _verify_input_tree(lock)
    for seed in specification()["design"]["network_seeds"]:
        schedule = locked_schedule(lock, seed)
        if schedule_sha256(schedule) != lock["schedules"][str(seed)]["sha256"]:
            raise RuntimeError("locked anytime schedule hash differs")
        validate_schedule(schedule)
    return lock


def locked_schedule(source: dict, seed: int) -> np.ndarray:
    row = source["schedules"][str(seed)]
    with np.load(verify_reference(row["file"]), allow_pickle=False) as raw:
        schedule = np.asarray(raw["schedule"], dtype=np.int16)
    return schedule


def validate_training_run(seed: int, recipe: str, arm: str) -> dict:
    directory = training_directory(seed, recipe, arm)
    manifest = load_json(directory / "run.json")
    if (
        manifest["lifecycle_state"] != "complete"
        or not validate_run_manifest(directory / "run.json")["passed"]
    ):
        raise RuntimeError("single-P anytime training run is incomplete")
    result = load_json(directory / "result.json")
    expected = (seed, recipe, arm, PROTOCOL_SHA256)
    observed = tuple(
        result[key] for key in ("seed", "recipe", "arm", "protocol_sha256")
    )
    if observed != expected:
        raise RuntimeError("single-P anytime training identity differs")
    spec = specification()
    rows = [
        json.loads(line)
        for line in (directory / "train_log.jsonl").read_text().splitlines()
    ]
    if [row["step"] for row in rows] != list(
        range(spec["training_schedule"]["updates"])
    ):
        raise RuntimeError("single-P anytime training trajectory is incomplete")
    schedule = build_schedule(seed)
    for step, row in enumerate(rows):
        scheduled_b, edge_count = map(int, schedule[step])
        retained_b = 4 if recipe == "fixed_horizon" else scheduled_b
        if (
            row["scheduled_B"] != scheduled_b
            or row["edge_count"] != edge_count
            or row["retained_B"] != retained_b
        ):
            raise RuntimeError("single-P anytime log differs from frozen schedule")
    if any(
        value != spec["training_schedule"]["updates"]
        for value in result["optimizer_steps"].values()
    ):
        raise RuntimeError("single-P anytime optimizer did not update every parameter")
    checkpoint = verify_reference(result["checkpoint"])
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload["recipe"] != recipe or payload["arm"] != arm:
        raise RuntimeError("single-P anytime checkpoint identity differs")
    return result


def _training_rows(seed: int, recipe: str, arm: str) -> list[dict]:
    return [
        json.loads(line)
        for line in (training_directory(seed, recipe, arm) / "train_log.jsonl")
        .read_text()
        .splitlines()
    ]


def _collect_seed_runs(seed: int, spec: dict) -> tuple[dict, dict, dict, dict]:
    runs, paired_rows, initial, max_streams = {}, {}, {}, {}
    for recipe in spec["design"]["training_recipes"]:
        for arm in spec["design"]["observation_training_arms"]:
            identity = f"{seed}/{recipe}/{arm}"
            metadata = validate_training_run(seed, recipe, arm)
            files = {
                path.name: reference(path)
                for path in sorted(training_directory(seed, recipe, arm).iterdir())
                if path.is_file()
            }
            runs[identity] = {"metadata": metadata, "files": files}
            paired_rows[identity] = _training_rows(seed, recipe, arm)
            initial[identity] = (
                metadata["initial_shadow"],
                metadata["initial_model"],
            )
            max_streams[identity] = metadata["max_stream_fingerprint"]
    return runs, paired_rows, initial, max_streams


def _validate_max_stream_pairing(paired_rows: dict, updates: int) -> None:
    keys = (
        "scheduled_B",
        "edge_count",
        "max_task_fingerprint",
        "max_clean_fingerprint",
        "max_noisy_fingerprint",
        "max_stream_fingerprint",
    )
    identities = sorted(paired_rows)
    for step in range(updates):
        common = {
            tuple(paired_rows[identity][step][key] for key in keys)
            for identity in identities
        }
        if len(common) != 1:
            raise RuntimeError(f"paired max-six stream differs at step {step}")


def _validate_b4_prefixes(seed: int, paired_rows: dict, arms: list[str]) -> None:
    for arm in arms:
        fixed = paired_rows[f"{seed}/fixed_horizon/{arm}"]
        variable = paired_rows[f"{seed}/variable_horizon/{arm}"]
        for step, (fixed_row, variable_row) in enumerate(
            zip(fixed, variable, strict=True)
        ):
            if variable_row["scheduled_B"] == 4 and (
                fixed_row["prefix_fingerprint"] != variable_row["prefix_fingerprint"]
            ):
                raise RuntimeError(
                    f"paired B4 prefixes differ for {seed}/{arm}/step{step}"
                )


def _locked_seed_runs(seed: int, spec: dict) -> dict:
    runs, paired_rows, initial, max_streams = _collect_seed_runs(seed, spec)
    if len({json.dumps(value, sort_keys=True) for value in initial.values()}) != 1:
        raise RuntimeError("paired anytime initializations differ")
    if len(set(max_streams.values())) != 1:
        raise RuntimeError("paired anytime max-six streams differ")
    _validate_max_stream_pairing(paired_rows, spec["training_schedule"]["updates"])
    _validate_b4_prefixes(
        seed, paired_rows, spec["design"]["observation_training_arms"]
    )
    return runs


def write_model_lock() -> dict:
    source, spec = validate_source_lock(), specification()
    runs = {}
    for seed in spec["design"]["network_seeds"]:
        runs.update(_locked_seed_runs(seed, spec))
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
        raise RuntimeError("single-P anytime model lock source differs")
    if len(lock["runs"]) != specification()["design"]["trained_models"]:
        raise RuntimeError("single-P anytime model lock is incomplete")
    for row in lock["runs"].values():
        for file in row["files"].values():
            verify_reference(file)
    return source, lock


__all__ = [
    "locked_schedule",
    "reference",
    "sources",
    "validate_model_lock",
    "validate_source_lock",
    "validate_training_run",
    "write_model_lock",
    "write_source_lock",
]
