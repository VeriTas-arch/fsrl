"""Prospective scope and native record ownership for the modulation schedule study."""

import json

from fsrl.experiments.linear_modulation.protocol import analysis_seed, recipe
from fsrl.experiments.training_strategy.locks import reference, verify_reference
from fsrl.experiments.write_cost.protocol import record_role
from fsrl.infra.provenance import load_json
from fsrl.paths import RUNS_ROOT, STUDIES_ROOT

STUDY = "modulation_schedule"
RECORDS = STUDIES_ROOT / STUDY / "records"
RUNS = RUNS_ROOT / "modulation_schedule_v1"
PROTOCOL = RECORDS / "benchmarks/protocol.json"
SOURCE = RECORDS / "benchmarks/source_lock.json"
CALIBRATION = RECORDS / "benchmarks/calibration_lock.json"


def specification():
    return load_json(PROTOCOL)


def parents():
    spec = specification()
    return {
        key: load_json(verify_reference(ref, commit=spec["parent_commit"]))
        for key, ref in spec["parents"].items()
    }


def register(
    status="unresolved", finding="Prospective frozen affine schedule diagnostic."
):
    values = {
        "schema_version": 1,
        "id": STUDY,
        "title": "State dependence versus stage schedules in affine modulation",
        "chapter": "algorithmic_compression",
        "order": 1130,
        "status": status,
        "review_state": "indexed",
        "question": specification()["question"],
        "finding": finding,
        "boundary": "Frozen six affine models; exposed panels; phase and conditional constant interventions. No universal necessity, human confirmation or main-model promotion. Compact diagnosis is separate.",
    }
    lines = [f"{key} = {json.dumps(value)}" for key, value in values.items()]
    for path in sorted(RECORDS.rglob("*")):
        if path.is_file():
            ref = reference(path)
            row = {
                "path": str(path.relative_to(RECORDS.parent)),
                "legacy_path": ref["path"],
                "origin": "native",
                "role": "registered_contract"
                if path == PROTOCOL
                else record_role(path),
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


__all__ = ["analysis_seed", "recipe"]
