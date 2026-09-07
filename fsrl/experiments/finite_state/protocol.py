"""One prospective authority for finite-state memory development."""

import json

from fsrl.experiments.training_strategy.locks import reference
from fsrl.experiments.write_cost.protocol import record_role
from fsrl.infra.provenance import file_sha256, load_json
from fsrl.paths import RUNS_ROOT, STUDIES_ROOT

STUDY = "finite_state_memory"
RECORDS = STUDIES_ROOT / STUDY / "records"
RUNS = RUNS_ROOT / "finite_state_memory_v1"
PROTOCOL = RECORDS / "benchmarks/protocol.json"
PROTOCOL_SHA256 = "7199a7c2ca0282d78ca92c15000576b617f75560a6e5ff835b1308210e551475"


def specification():
    if file_sha256(PROTOCOL) != PROTOCOL_SHA256:
        raise RuntimeError("finite-state protocol changed")
    return load_json(PROTOCOL)


def register(
    status="unresolved", finding="Prospectively registered; execution pending."
):
    header = {
        "schema_version": 1,
        "id": STUDY,
        "title": "Finite-state fast memory and effective write cost",
        "chapter": "algorithmic_compression",
        "order": 1050,
        "status": status,
        "review_state": "indexed",
        "question": specification()["question"],
        "finding": finding,
        "boundary": "Development; one generic initialization and conditional three new four-condition fits. Fixed per-connection states, single-stage joint training, project-exposed Liu knowledge; no main-model promotion or physiological precision estimate.",
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
