"""Prospective cross-diagnostic protocol and inherited archive identities."""

import json

from fsrl.experiments.evidence_routing.locks import require_clean
from fsrl.experiments.training_strategy.locks import reference, verify_reference
from fsrl.experiments.write_cost.protocol import record_role
from fsrl.infra.provenance import file_sha256, load_json
from fsrl.paths import REPO_ROOT, RUNS_ROOT, STUDIES_ROOT

STUDY = "observation_crossover"
RECORDS = STUDIES_ROOT / STUDY / "records"
RUNS = RUNS_ROOT / "observation_crossover_v1"
PROTOCOL = RECORDS / "benchmarks/protocol.json"
LOCK = RECORDS / "benchmarks/execution_lock.json"
PROTOCOL_SHA256 = "5aa39260e419c18d364e3c2206b568578a9c81f1b476c36f13c9cc115a73a6d8"


def specification():
    if file_sha256(PROTOCOL) != PROTOCOL_SHA256:
        raise RuntimeError("cross-diagnostic protocol changed")
    return load_json(PROTOCOL)


def inherited():
    spec = specification()
    parents = {
        key: load_json(
            verify_reference(spec[key], commit=spec["parent_witness_commit"])
        )
        for key in (
            "parent_protocol",
            "parent_result",
            "parent_source_lock",
            "parent_model_lock",
        )
    }
    return parents


def execution_identity():
    commit, spec = require_clean(), specification()
    parents = inherited()
    previous = parents["parent_result"]
    files = parents["parent_source_lock"]["sources"] + [reference(PROTOCOL)]
    files += [
        spec[key]
        for key in spec
        if key.startswith("parent_") and isinstance(spec[key], dict)
    ]
    for base in ("fsrl/experiments", "tests/experiments"):
        files += [reference(p) for p in sorted((REPO_ROOT / base / STUDY).glob("*.py"))]
    inputs = {}
    for name, row in parents["parent_source_lock"]["inputs"].items():
        if name.startswith("test-") or name == "liu-8":
            ref = previous["artifacts"][f"inputs/{name}.npz"]
            inputs[name] = {**row, "file": ref}
            files.append(ref)
    models, old = {}, {}
    for seed in spec["design"]["seeds"]:
        for arm in ("clean", "noisy"):
            key = f"{seed}/{arm}"
            refs = {
                name: previous["artifacts"][f"training/{key}/{name}"]
                for name in ("net.pth", "local.pth", "result.json")
            }
            models[key] = refs
            files += list(refs.values())
        old[str(seed)] = {}
        for cell, row in spec["design"]["cells"].items():
            if row["reuse"] is None:
                continue
            refs = {
                phase: previous["artifacts"][f"{base}/{seed}/{row['reuse']}/raw.npz"]
                for phase, base in (("generic", "generic/test"), ("liu", "liu"))
            }
            old[str(seed)][cell] = refs
            files += list(refs.values())
    files = sorted(
        {row["path"]: row for row in files}.values(), key=lambda r: r["path"]
    )
    for row in files:
        verify_reference(row, commit=commit)
    return {
        "source_commit": commit,
        "protocol_sha256": PROTOCOL_SHA256,
        "files": files,
        "inputs": inputs,
        "models": models,
        "previous_cells": old,
        "runtime": parents["parent_source_lock"]["runtime"],
    }


def validate_lock():
    verify_reference(reference(LOCK), commit=require_clean())
    lock = load_json(LOCK)
    if lock["protocol_sha256"] != PROTOCOL_SHA256:
        raise RuntimeError("execution protocol mismatch")
    for ref in lock["files"]:
        verify_reference(ref, commit=lock["source_commit"])
    qualification = load_json(verify_reference(lock["qualification"]))
    if not qualification["passed"]:
        raise RuntimeError("qualification failed")
    return lock


def register(
    status="unresolved", finding="Prospective crossover; missing cell not evaluated."
):
    values = {
        "schema_version": 1,
        "id": STUDY,
        "title": "Training by observation crossover",
        "chapter": "algorithmic_compression",
        "order": 1070,
        "status": status,
        "review_state": "indexed",
        "question": specification()["question"],
        "finding": finding,
        "boundary": "Three previously trained paired networks; three exposed cells and one prospectively frozen cross-evaluation; no training, human calibration or promotion.",
    }
    lines = [f"{k} = {json.dumps(v)}" for k, v in values.items()]
    for path in sorted(RECORDS.rglob("*")):
        if not path.is_file():
            continue
        ref = reference(path)
        row = {
            "path": str(path.relative_to(RECORDS.parent)),
            "legacy_path": ref["path"],
            "origin": "native",
            "role": "registered_contract" if path == PROTOCOL else record_role(path),
            "sha256": ref["sha256"],
            "bytes": ref["bytes"],
            "source_ref": "sha256:" + ref["sha256"],
        }
        lines += [
            "",
            "[[records]]",
            *[f"{k} = {json.dumps(v)}" for k, v in row.items()],
        ]
    (RECORDS.parent / "study.toml").write_text("\n".join(lines) + "\n")
