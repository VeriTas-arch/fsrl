"""Prospective implementation/input locks beside unchanged parent witnesses."""

from __future__ import annotations

import tomllib

import numpy as np

from fsrl.experiments.minimal_learner.decisions import adequate
from fsrl.experiments.training_strategy.locks import (
    git_text,
    reference,
    require_pushed_clean,
    verify_reference,
)
from fsrl.infra.provenance import file_sha256, load_json, write_json_exclusive
from fsrl.paths import REPO_ROOT, STUDIES_ROOT
from fsrl.tasks.protocol_catalog import protocol_path

RECORD_ROOT = STUDIES_ROOT / "minimal_learner_diagnostics" / "records"
PROTOCOL = RECORD_ROOT / "benchmarks/minimal_learner_diagnostics_v1.json"
PROTOCOL_COMMIT = "d8bdc3af04cc31f3ef436d751ab823caa38dd593"
PROTOCOL_HASH = "08aac3c95b5d8298c9d329ad88a6d219d6a7818bb1575f81a405009546ea3f83"
LOCK = RECORD_ROOT / "benchmarks/minimal_learner_diagnostics_v1.execution_lock.json"
PARENT = STUDIES_ROOT / "minimal_relational_learner/records"
PARENT_RESULT = PARENT / "results/minimal_relational_learner_v1.json"


def specification() -> dict:
    if file_sha256(PROTOCOL) != PROTOCOL_HASH:
        raise RuntimeError("diagnostic protocol changed")
    verify_reference(reference(PROTOCOL), commit=PROTOCOL_COMMIT)
    return load_json(PROTOCOL)


def sources() -> list[dict]:
    paths = list((REPO_ROOT / "fsrl").rglob("*.py"))
    paths.extend((REPO_ROOT / "tests/experiments/learner_diagnostics").glob("*.py"))
    paths.extend((REPO_ROOT / "pyproject.toml", REPO_ROOT / ".envrc"))
    return [reference(path) for path in sorted(paths)]


def parent_inputs() -> list[dict]:
    spec = specification()
    manifest = REPO_ROOT / spec["parent_manifest"]
    if file_sha256(manifest) != spec["parent_manifest_sha256"]:
        raise RuntimeError("parent study identity changed")
    records = tomllib.loads(manifest.read_text())["records"]
    inputs = [
        reference(manifest),
        reference(PROTOCOL),
        reference(protocol_path("liu_v2")),
    ]
    for row in records:
        ref = {
            "path": (manifest.parent / row["path"]).relative_to(REPO_ROOT).as_posix(),
            "sha256": row["sha256"],
            "bytes": row["bytes"],
        }
        verify_reference(ref, commit=spec["parent_commit"])
        inputs.append(ref)
    # Verify the original enumerated sources, not an obsolete complete file set.
    old_lock = load_json(
        PARENT / "benchmarks/minimal_relational_learner_v1.source_lock.json"
    )
    for ref in old_lock["sources"] + old_lock["inputs"]:
        verify_reference(ref, commit=old_lock["source_commit"])
    return inputs


def validate_parent() -> dict:
    spec = specification()
    original = load_json(PARENT / "benchmarks/minimal_relational_learner_v1.json")
    lock = load_json(
        PARENT / "benchmarks/minimal_relational_learner_v1.artifact_lock.json"
    )
    result = load_json(PARENT_RESULT)
    expected = {f"{s}/{c}" for s in spec["seeds"] for c in spec["conditions"]}
    if set(result["conditions"]) != expected or set(lock["runs"]) != expected:
        raise RuntimeError("mandatory six-fit inventory differs")
    for identity, row in result["conditions"].items():
        if not adequate(row, original):
            raise RuntimeError(f"parent competence failed: {identity}")
        run = lock["runs"][identity]
        if row["parameters"] != run["config"]["physical_parameters"]:
            raise RuntimeError("parameter summary differs from pre-evaluation lock")
        for ref in run["files"]:
            verify_reference(ref)
        verify_reference(row["registered_raw_arrays"], commit=spec["parent_commit"])
    return result


def write_lock() -> dict:
    commit = require_pushed_clean()
    inputs = parent_inputs()
    validate_parent()
    source = sources()
    for ref in source + inputs:
        verify_reference(ref, commit=commit)
    lock = {
        "experiment_id": specification()["experiment_id"],
        "source_commit": commit,
        "protocol_commit": PROTOCOL_COMMIT,
        "protocol_sha256": PROTOCOL_HASH,
        "sources": source,
        "inputs": inputs,
    }
    write_json_exclusive(LOCK, lock)
    return reference(LOCK)


def validate_lock(*, pushed: bool) -> dict:
    commit = require_pushed_clean() if pushed else git_text("rev-parse", "HEAD")
    lock = load_json(LOCK)
    verify_reference(reference(LOCK), commit=commit)
    if lock["sources"] != sources() or lock["inputs"] != parent_inputs():
        raise RuntimeError("locked implementation or input inventory changed")
    if (
        lock["protocol_sha256"] != PROTOCOL_HASH
        or lock["protocol_commit"] != PROTOCOL_COMMIT
    ):
        raise RuntimeError("execution lock protocol differs")
    for ref in lock["sources"] + lock["inputs"]:
        verify_reference(ref, commit=lock["source_commit"])
    validate_parent()
    return lock


def load_arrays(row: dict) -> tuple[dict, dict]:
    path = verify_reference(row["registered_raw_arrays"])
    with np.load(path, allow_pickle=False) as saved:
        arrays = {name: saved[name] for name in saved.files}
    prefix = "liu__inputs__"
    inputs = {
        name.removeprefix(prefix): a
        for name, a in arrays.items()
        if name.startswith(prefix)
    }
    return arrays, inputs
