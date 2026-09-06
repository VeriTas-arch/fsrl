"""Prospective phase locks and validated, write-once training identities."""

import json

from fsrl.experiments.evidence_routing.locks import require_clean
from fsrl.experiments.training_strategy.locks import (
    reference,
    scientific_inputs,
    verify_reference,
)
from fsrl.infra.file_contracts import validate_run_manifest
from fsrl.infra.provenance import load_json, write_json_exclusive
from fsrl.paths import REPO_ROOT

from .inputs import freeze_inputs
from .protocol import PROTOCOL, PROTOCOL_SHA256, RECORDS, RUNS, specification

SOURCE = RECORDS / "benchmarks/source_lock.json"
SCALE = RECORDS / "benchmarks/scale_lock.json"
SELECTION = RECORDS / "benchmarks/selection_lock.json"
MODELS = RECORDS / "benchmarks/model_lock.json"


def sources():
    paths = list((REPO_ROOT / "fsrl").rglob("*.py"))
    paths += list((REPO_ROOT / "tests/experiments/write_cost").glob("*.py"))
    paths += [REPO_ROOT / "pyproject.toml", REPO_ROOT / ".envrc"]
    return [reference(p) for p in sorted(paths)]


def lock_source():
    commit = require_clean()
    q = RECORDS / "benchmarks/qualification.final.json"
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
        "status": "locked_before_training",
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


def completed(directory):
    if not validate_run_manifest(directory / "run.json")["passed"]:
        raise RuntimeError(f"invalid run: {directory}")
    if load_json(directory / "run.json")["lifecycle_state"] != "complete":
        raise RuntimeError(f"incomplete run: {directory}")
    return load_json(directory / "result.json")


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
    if [r["step"] for r in logs] != list(range(steps)):
        raise RuntimeError("training steps differ")
    if (
        metadata["protocol_sha256"] != PROTOCOL_SHA256
        or metadata["seed"] != seed
        or metadata["arm"] != arm
    ):
        raise RuntimeError("training identity mismatch")
    for key in ("backbone.i2h.weight", "backbone.h2DA.weight", "local.raw_gain"):
        if metadata["optimizer_steps"][key] != steps:
            raise RuntimeError("incomplete joint learning")
    return {
        "metadata": metadata,
        "files": [reference(p) for p in sorted(directory.iterdir()) if p.is_file()],
    }


def validate_phase(path):
    validate_source()
    verify_reference(reference(path), commit=require_clean())
    lock = load_json(path)
    for row in lock.get("files", []):
        verify_reference(row)
    for run in lock.get("runs", {}).values():
        for row in run["files"]:
            verify_reference(row)
    return lock


def lock_scale():
    source = validate_source()
    seed = specification()["seeds"]["development"]
    run = run_record(seed, "shared")
    from .evaluation import generic_run

    row = generic_run(seed, "shared", "development", source)
    value = row["mean_cost"]
    if value <= 0 or not row["competence"]:
        raise RuntimeError(
            "development shared reference must be competent with positive cost"
        )
    result = {
        "source_commit": source["source_commit"],
        "c_ref": value,
        "coefficients": {
            str(k): k / value for k in specification()["cost"]["kappa_candidates"]
        },
        "runs": {"baseline": run},
        "generic_baseline": row,
        "files": [
            reference(RUNS / "generic/development" / str(seed) / "shared/result.json")
        ],
    }
    write_json_exclusive(SCALE, result)
    return {"c_ref": value, "coefficients": result["coefficients"]}


def select():
    source = validate_source()
    scale = validate_phase(SCALE)
    spec = specification()
    seed = spec["seeds"]["development"]
    # No cost-arm validation outcomes until every candidate checkpoint is locked.
    runs = {
        str(k): run_record(seed, f"kappa-{k}") for k in spec["cost"]["kappa_candidates"]
    }
    barrier = RECORDS / "benchmarks/development_model_lock.json"
    write_json_exclusive(barrier, {"runs": runs, "files": [reference(SCALE)]})
    from .evaluation import generic_run

    curve = []
    for k in spec["cost"]["kappa_candidates"]:
        row = generic_run(seed, f"kappa-{k}", "development", source)
        curve.append(
            {
                "kappa": k,
                "coefficient": scale["coefficients"][str(k)],
                "eligible": row["competence"]
                and row["mean_cost"] <= spec["cost"]["budget_ratio"] * scale["c_ref"],
                **row,
            }
        )
    eligible = [row for row in curve if row["eligible"]]
    selected = min(eligible, key=lambda r: (r["ce"], r["kappa"])) if eligible else None
    result = {
        "selected": selected,
        "curve": curve,
        "baseline": scale["generic_baseline"],
        "runs": runs,
        "files": [reference(SCALE), reference(barrier)],
        "outcome": "selected_on_generic_only" if selected else "no_feasible_candidate",
        "liu_evaluated": False,
    }
    write_json_exclusive(SELECTION, result)
    return {
        "outcome": result["outcome"],
        "selected_kappa": None if selected is None else selected["kappa"],
        "curve": curve,
    }


def lock_models():
    selected = validate_phase(SELECTION)["selected"]
    if selected is None:
        return {"status": "not_triggered"}
    runs = {}
    for seed in specification()["seeds"]["mandatory"]:
        pair = {arm: run_record(seed, arm) for arm in ("shared", "cost")}
        for key in ("initial_backbone", "initial_local", "stream_fingerprint"):
            if pair["shared"]["metadata"][key] != pair["cost"]["metadata"][key]:
                raise RuntimeError("paired initialization or stream mismatch")
        runs.update({f"{seed}/{arm}": row for arm, row in pair.items()})
    write_json_exclusive(
        MODELS,
        {
            "runs": runs,
            "files": [reference(SELECTION)],
            "status": "six_final_models_locked_before_outcomes",
        },
    )
    return {"models": len(runs)}
