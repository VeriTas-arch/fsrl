"""One prospective authority for observation-uncertainty memory development."""

import json

from fsrl.experiments.training_strategy.locks import reference
from fsrl.experiments.write_cost.protocol import record_role
from fsrl.infra.provenance import file_sha256, load_json
from fsrl.paths import RUNS_ROOT, STUDIES_ROOT

STUDY = "observation_uncertainty"
RECORDS = STUDIES_ROOT / STUDY / "records"
RUNS = RUNS_ROOT / "observation_uncertainty_v1"
PROTOCOL = RECORDS / "benchmarks/protocol.json"
PROTOCOL_SHA256 = "522a6f0eaab3923a04e4e644a8033fcb917fd85b0dcd37f63a3552756baf4b92"


def specification():
    if file_sha256(PROTOCOL) != PROTOCOL_SHA256:
        raise RuntimeError("observation-uncertainty protocol changed")
    return load_json(PROTOCOL)


def register(
    status="unresolved", finding="Prospectively registered; execution pending."
):
    header = {
        "schema_version": 1,
        "id": STUDY,
        "title": "Observation uncertainty and sign-restored control",
        "chapter": "algorithmic_compression",
        "order": 1060,
        "status": status,
        "review_state": "indexed",
        "question": specification()["question"],
        "finding": finding,
        "boundary": "Development; one fixed observation-error scale, sign-restored oracle control, three paired network seeds, single-stage joint training; no human calibration or promotion.",
    }
    lines = [f"{key} = {json.dumps(value)}" for key, value in header.items()]
    for path in sorted(RECORDS.rglob("*")):
        if not path.is_file():
            continue
        row = reference(path)
        role = "registered_contract" if path == PROTOCOL else record_role(path)
        values = {
            "path": str(path.relative_to(RECORDS.parent)),
            "legacy_path": row["path"],
            "origin": "native",
            "role": role,
            "sha256": row["sha256"],
            "bytes": row["bytes"],
            "source_ref": "sha256:" + row["sha256"],
        }
        lines += ["", "[[records]]"]
        lines += [f"{key} = {json.dumps(value)}" for key, value in values.items()]
    (RECORDS.parent / "study.toml").write_text("\n".join(lines) + "\n")
