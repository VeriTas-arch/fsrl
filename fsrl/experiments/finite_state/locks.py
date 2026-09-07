"""Prospective source, input, selection and complete-model barriers."""

import json

from fsrl.experiments.evidence_routing.locks import require_clean
from fsrl.experiments.training_strategy.locks import (
    reference,
    scientific_inputs,
    verify_reference,
)
from fsrl.experiments.write_cost.locks import completed
from fsrl.infra.provenance import load_json, write_json_exclusive
from fsrl.paths import REPO_ROOT

from .inputs import freeze_inputs
from .protocol import PROTOCOL, PROTOCOL_SHA256, RECORDS, RUNS, specification

SOURCE = RECORDS / "benchmarks/source_lock.json"
DEVELOPMENT = RECORDS / "benchmarks/development_model_lock.json"
SELECTION = RECORDS / "benchmarks/selection_lock.json"
MODELS = RECORDS / "benchmarks/model_lock.json"


def sources():
    paths = list((REPO_ROOT / "fsrl").rglob("*.py"))
    paths += list((REPO_ROOT / "tests/experiments/finite_state").glob("*.py"))
    paths += [REPO_ROOT / "pyproject.toml", REPO_ROOT / ".envrc"]
    return [reference(path) for path in sorted(paths)]


def lock_source():
    commit = require_clean()
    q = RECORDS / "benchmarks/qualification.json"
    qualification = load_json(q)
    if not qualification["passed"] or qualification["sources"] != sources():
        raise RuntimeError("qualification does not cover current sources")
    spec = specification()
    files = sources() + [reference(PROTOCOL)] + scientific_inputs()
    files += [reference(REPO_ROOT / row["path"]) for row in spec["references"].values()]
    for row in files:
        verify_reference(row, commit=commit)
    result = {
        "source_commit": commit,
        "protocol_sha256": PROTOCOL_SHA256,
        "sources": files,
        "qualification": reference(q),
        "runtime": qualification["runtime"],
        "inputs": freeze_inputs(spec),
    }
    write_json_exclusive(SOURCE, result)
    return {"source_commit": commit, "inputs": len(result["inputs"])}


def validate_source():
    head = require_clean()
    verify_reference(reference(SOURCE), commit=head)
    lock = load_json(SOURCE)
    if lock["protocol_sha256"] != PROTOCOL_SHA256:
        raise RuntimeError("source protocol mismatch")
    for row in lock["sources"]:
        verify_reference(row, commit=lock["source_commit"])
    verify_reference(lock["qualification"])
    for row in lock["inputs"].values():
        verify_reference(row["file"])
    return lock


def training_dir(seed, arm):
    return RUNS / "training" / str(seed) / arm


def run_record(seed, arm):
    directory = training_dir(seed, arm)
    metadata = completed(directory)
    logs = [
        json.loads(line)
        for line in (directory / "train_log.jsonl").read_text().splitlines()
    ]
    steps = specification()["optimization"]["total_steps"]
    if [row["step"] for row in logs] != list(range(steps)):
        raise RuntimeError("training step coverage differs")
    if (metadata["protocol_sha256"], metadata["seed"], metadata["arm"]) != (
        PROTOCOL_SHA256,
        seed,
        arm,
    ):
        raise RuntimeError("training identity differs")
    for key in ("backbone.i2h.weight", "backbone.h2DA.weight", "local.raw_gain"):
        if metadata["optimizer_steps"][key] != steps:
            raise RuntimeError("single-stage joint optimization incomplete")
    return {
        "metadata": metadata,
        "files": [reference(p) for p in sorted(directory.iterdir()) if p.is_file()],
    }


def validate_phase(path):
    validate_source()
    verify_reference(reference(path), commit=require_clean())
    phase = load_json(path)
    for row in phase.get("files", []):
        verify_reference(row)
    for run in phase.get("runs", {}).values():
        for row in run["files"]:
            verify_reference(row)
    return phase


def paired_records(jobs):
    runs = {f"{seed}/{arm}": run_record(seed, arm) for seed, arm in jobs}
    for seed in sorted({seed for seed, _ in jobs}):
        group = [
            row["metadata"] for row in runs.values() if row["metadata"]["seed"] == seed
        ]
        for key in ("initial_backbone", "initial_local", "stream_fingerprint"):
            if any(row[key] != group[0][key] for row in group):
                raise RuntimeError("paired initialization or task stream differs")
    return runs


def lock_development():
    validate_source()
    spec = specification()
    jobs = [
        (spec["seeds"]["development"], arm)
        for arm in ["shared", *(f"K-{k}" for k in spec["storage"]["state_candidates"])]
    ]
    result = {
        "runs": paired_records(jobs),
        "files": [reference(SOURCE)],
        "status": "four_development_models_locked_before_validation",
    }
    write_json_exclusive(DEVELOPMENT, result)
    return {"models": len(jobs)}


def lock_models():
    if validate_phase(SELECTION)["selected_K"] is None:
        return {"status": "not_triggered"}
    spec = specification()
    jobs = [
        (seed, arm)
        for seed in spec["seeds"]["mandatory"]
        for arm in spec["seeds"]["conditions"]
    ]
    result = {
        "runs": paired_records(jobs),
        "files": [reference(SELECTION)],
        "status": "twelve_final_models_locked_before_primary_outcomes",
    }
    write_json_exclusive(MODELS, result)
    return {"models": len(jobs)}
