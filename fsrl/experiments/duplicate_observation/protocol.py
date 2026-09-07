"""Fixed independent replication identities and native record ownership."""

import json

from fsrl.experiments.training_strategy.locks import reference
from fsrl.experiments.write_cost.protocol import record_role
from fsrl.infra.provenance import file_sha256, load_json
from fsrl.paths import RUNS_ROOT, STUDIES_ROOT

STUDY = "duplicate_observation"
RECORDS = STUDIES_ROOT / STUDY / "records"
RUNS = RUNS_ROOT / "duplicate_observation_v1"
PROTOCOL = RECORDS / "benchmarks/protocol.json"
SOURCE = RECORDS / "benchmarks/source_lock.json"
MODELS = RECORDS / "benchmarks/model_lock.json"
PROTOCOL_SHA256 = "5722e54a0f56163e15b65d5086503385e5dc06896ab30c9ecbe066f6bd86d345"


def specification():
    if file_sha256(PROTOCOL) != PROTOCOL_SHA256:
        raise RuntimeError("cross-diagnostic protocol changed")
    return load_json(PROTOCOL)


def register(
    status="unresolved",
    finding="Prospective matched duplicate-observation comparison; new outcomes not evaluated.",
):
    values = {
        "schema_version": 1,
        "id": STUDY,
        "title": "Duplicate-observation parameterization in a single-memory RNN",
        "chapter": "algorithmic_compression",
        "order": 1110,
        "status": status,
        "review_state": "indexed",
        "question": specification()["question"],
        "finding": finding,
        "boundary": "Three matched A/C instances and exposed panels; same q copied into both columns, no explicit zq hint or local memory. No human confirmation or promotion.",
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


def recipe(panel=None):
    from fsrl.experiments.observation_replication.protocol import (
        recipe as inherited_recipe,
    )

    return inherited_recipe(panel)


def analysis_seed(seed, panel):
    return seed + panel * 1000000


def parents():
    from fsrl.experiments.training_strategy.locks import verify_reference

    spec = specification()
    return {
        key: load_json(verify_reference(value, commit=spec["parent_witness_commit"]))
        for key, value in spec["parents"].items()
    }
