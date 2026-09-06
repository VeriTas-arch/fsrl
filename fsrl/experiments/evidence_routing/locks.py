"""Prospective source/input identities and the six-model evaluation barrier."""

import hashlib
import json

import numpy as np

from fsrl.experiments.memory_structure.inputs import liu_inputs, prepare_shared
from fsrl.experiments.training_strategy.evaluation import write_arrays
from fsrl.experiments.training_strategy.generic_validation import (
    validation_episodes,
    validation_groups,
)
from fsrl.experiments.training_strategy.locks import (
    git_text,
    reference,
    scientific_inputs,
    verify_reference,
)
from fsrl.infra.file_contracts import validate_run_manifest
from fsrl.infra.provenance import load_json, write_json_exclusive
from fsrl.paths import REPO_ROOT

from .inputs import check_pair
from .protocol import (
    PROTOCOL_PATH,
    PROTOCOL_SHA256,
    RECORD_ROOT,
    RUN_ROOT,
    run_directory,
    specification,
)

SOURCE_LOCK = RECORD_ROOT / "benchmarks" / "source_lock.json"
ARTIFACT_LOCK = RECORD_ROOT / "benchmarks" / "artifact_lock.json"


def require_clean() -> str:
    if git_text("branch", "--show-current") != "dev":
        raise RuntimeError("the study requires shared dev")
    if git_text("status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("scientific execution requires committed sources and locks")
    return git_text("rev-parse", "HEAD")


def sources() -> list[dict]:
    paths = list((REPO_ROOT / "fsrl").rglob("*.py"))
    paths += list((REPO_ROOT / "tests/experiments/evidence_routing").glob("*.py"))
    paths += [REPO_ROOT / "pyproject.toml", REPO_ROOT / ".envrc"]
    return [reference(path) for path in sorted(paths)]


def freeze_inputs(spec: dict) -> dict:
    directory = RUN_ROOT / "inputs"
    directory.mkdir(parents=True, exist_ok=False)
    records = {}
    for size in (
        spec["evaluation"]["item_counts"] + spec["evaluation"]["transport_item_counts"]
    ):
        _, batch = liu_inputs(spec, size)
        check_pair(batch)
        path = directory / f"liu-{size}.npz"
        write_arrays(path, batch.arrays)
        records[f"liu-{size}"] = {
            "file": reference(path),
            "fingerprint": batch.fingerprint(),
        }
    episodes = validation_episodes(spec)
    for length, indices in validation_groups(episodes).items():
        selected = tuple(episodes[index] for index in indices)
        batch = prepare_shared(selected)
        check_pair(batch)
        batch.arrays["learned"] = np.asarray(
            [
                [
                    tuple(sorted((query.left_item, query.right_item)))
                    in {
                        tuple(sorted((trial.left_item, trial.right_item)))
                        for trial in episode.support_trials
                    }
                    for query in episode.query_trials
                ]
                for episode in selected
            ]
        )
        batch.arrays["episode_indices"] = np.asarray(indices)
        path = directory / f"generic-{length}.npz"
        write_arrays(path, batch.arrays)
        records[f"generic-{length}"] = {
            "file": reference(path),
            "fingerprint": batch.fingerprint(),
        }
    return records


def write_source_lock() -> dict:
    commit = require_clean()
    spec = specification()
    qualification = load_json(RUN_ROOT / "qualification" / "result.json")
    if not qualification["passed"] or qualification["sources"] != sources():
        raise RuntimeError("current-source qualification is required")
    inputs = freeze_inputs(spec)
    records = sources() + [reference(PROTOCOL_PATH)] + scientific_inputs()
    records += [
        reference(REPO_ROOT / row["path"]) for row in spec["references"].values()
    ]
    for record in records:
        verify_reference(record, commit=commit)
    result = {
        "protocol_sha256": PROTOCOL_SHA256,
        "source_commit": commit,
        "sources": records,
        "inputs": inputs,
        "qualification": reference(RUN_ROOT / "qualification" / "result.json"),
        "runtime": qualification["runtime"],
        "status": "frozen_before_training_and_outcomes",
    }
    write_json_exclusive(SOURCE_LOCK, result)
    return result


def validate_source_lock() -> dict:
    head = require_clean()
    verify_reference(reference(SOURCE_LOCK), commit=head)
    lock = load_json(SOURCE_LOCK)
    if lock["protocol_sha256"] != PROTOCOL_SHA256:
        raise RuntimeError("protocol identity differs from source lock")
    for record in lock["sources"]:
        verify_reference(record, commit=lock["source_commit"])
    for record in lock["inputs"].values():
        verify_reference(record["file"])
    verify_reference(lock["qualification"])
    return lock


def validate_run(seed: int, condition: str) -> dict:
    directory = run_directory(seed, condition)
    run = load_json(directory / "run.json")
    if (
        run["lifecycle_state"] != "complete"
        or not validate_run_manifest(directory / "run.json")["passed"]
    ):
        raise RuntimeError("training attempt incomplete or modified")
    metadata = load_json(directory / "config.json")
    spec = specification()
    steps = spec["optimization"]["total_steps"]
    logs = [
        json.loads(line)
        for line in (directory / "train_log.jsonl").read_text().splitlines()
    ]
    if metadata["protocol_sha256"] != PROTOCOL_SHA256 or len(logs) != steps:
        raise RuntimeError("training identity or step count differs")
    if [row["step"] for row in logs] != list(range(steps)):
        raise RuntimeError("non-contiguous training history")
    if metadata["stream_fingerprint"] != logs[-1]["stream_fingerprint"]:
        raise RuntimeError("training stream identity differs")
    counts = metadata["optimizer_steps"]
    if (
        counts["backbone.i2h.weight"] != steps
        or counts["backbone.h2DA.weight"] != steps
    ):
        raise RuntimeError("backbone was not jointly trained throughout")
    if counts["local.raw_gain"] != steps:
        raise RuntimeError("local gain was not jointly trained throughout")
    if metadata["local_scalars"] != spec["task"]["cue_size"] ** 2:
        raise RuntimeError("local state size differs")
    if metadata["seed"] != seed or metadata["condition"] != condition:
        raise RuntimeError("training identity differs")
    validate_streams(logs, metadata)
    return metadata


def validate_streams(logs: list, metadata: dict) -> None:
    for prefix, log_key in (("", "batch_fingerprint"), ("base_", "base_fingerprint")):
        digest = hashlib.sha256()
        for row in logs:
            digest.update(bytes.fromhex(row[log_key]))
            if prefix == "" and digest.hexdigest() != row["stream_fingerprint"]:
                raise RuntimeError("training stream history differs")
        if digest.hexdigest() != metadata[prefix + "stream_fingerprint"]:
            raise RuntimeError("training stream digest differs")


def write_artifact_lock() -> dict:
    source = validate_source_lock()
    spec = specification()
    runs = {}
    for seed in spec["seeds"]["mandatory"]:
        pair = {}
        for condition in spec["seeds"]["conditions"]:
            metadata = validate_run(seed, condition)
            pair[condition] = metadata
            directory = run_directory(seed, condition)
            runs[f"{seed}/{condition}"] = {
                "metadata": metadata,
                "files": [
                    reference(path)
                    for path in sorted(directory.iterdir())
                    if path.is_file()
                ],
            }
        for key in ("initial_backbone", "initial_local", "base_stream_fingerprint"):
            if pair["shared"][key] != pair["isolated"][key]:
                raise RuntimeError(f"paired training mismatch: {seed}/{key}")
    result = {
        "source_commit": source["source_commit"],
        "protocol_sha256": PROTOCOL_SHA256,
        "source_lock": reference(SOURCE_LOCK),
        "runs": runs,
        "status": "six_final_models_locked_before_evaluation",
    }
    write_json_exclusive(ARTIFACT_LOCK, result)
    return result


def validate_artifacts() -> dict:
    validate_source_lock()
    verify_reference(reference(ARTIFACT_LOCK), commit=require_clean())
    lock = load_json(ARTIFACT_LOCK)
    verify_reference(lock["source_lock"])
    if lock["protocol_sha256"] != PROTOCOL_SHA256:
        raise RuntimeError("artifact protocol differs")
    for row in lock["runs"].values():
        for record in row["files"]:
            verify_reference(record)
    return lock
