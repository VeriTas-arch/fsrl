"""One prospective successor; prior outcomes remain immutable."""

import json

from fsrl.experiments.modulation_schedule.protocol import parents, recipe
from fsrl.experiments.training_strategy.locks import reference, verify_reference
from fsrl.experiments.write_cost.protocol import record_role
from fsrl.infra.provenance import load_json
from fsrl.paths import RUNS_ROOT, STUDIES_ROOT

RECORDS = STUDIES_ROOT / "write_budget_match/records"
RUNS = RUNS_ROOT / "write_budget_match_v1"
PROTOCOL = RECORDS / "benchmarks/protocol.json"
SOURCE = RECORDS / "benchmarks/source_lock.json"
CALIBRATION = RECORDS / "benchmarks/calibration_lock.json"


def specification():
    return load_json(PROTOCOL)


def prior_records():
    spec = specification()
    prior = parents()
    prior.update(
        {
            k: load_json(verify_reference(v, commit=spec["parent_commit"]))
            for k, v in spec["parents"].items()
        }
    )
    return prior


def register(status="unresolved", finding="Prospective actual-write matching test."):
    values = {
        "schema_version": 1,
        "id": "write_budget_match",
        "title": "Actual stage write budgets versus modulation organization",
        "chapter": "algorithmic_compression",
        "order": 1140,
        "status": status,
        "review_state": "indexed",
        "question": specification()["question"],
        "finding": finding,
        "boundary": "Frozen affine networks and exposed panels; budget-only calibration. No from-scratch training, reliability-gate necessity, main-model promotion or compact requalification.",
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


__all__ = ["recipe"]
