"""Prospective contract and native study registration."""

import json

from fsrl.experiments.training_strategy.locks import reference
from fsrl.infra.provenance import file_sha256, load_json
from fsrl.paths import REPO_ROOT, RUNS_ROOT, STUDIES_ROOT

RECORDS = STUDIES_ROOT / "weak_evidence_contribution" / "records"
RUNS = RUNS_ROOT / "weak_evidence_contribution_v1"
PROTOCOL = RECORDS / "benchmarks/protocol.json"
PROTOCOL_SHA256 = "df597a2260102b31c403095bef50b36683641b6a3e37071d437ab821b2c7f742"
LOCK = RECORDS / "benchmarks/source_lock.json"
QUALIFICATION = RECORDS / "benchmarks/qualification.json"


def specification():
    if file_sha256(PROTOCOL) != PROTOCOL_SHA256:
        raise RuntimeError("weak contribution protocol changed")
    return load_json(PROTOCOL)


def parent():
    return REPO_ROOT / specification()["parent"]


def parent_spec():
    return load_json(parent() / "benchmarks/protocol.json")


def register(
    status="unresolved", finding="Prospectively registered; execution pending."
):
    header = {
        "schema_version": 1,
        "id": "weak_evidence_contribution",
        "title": "Weak evidence contribution in frozen shared and cost RNNs",
        "chapter": "algorithmic_compression",
        "order": 1040,
        "status": status,
        "review_state": "indexed",
        "question": specification()["question"],
        "finding": finding,
        "boundary": specification()["decision"]["claim_boundary"],
    }
    lines = [f"{key} = {json.dumps(value)}" for key, value in header.items()]
    for path in sorted(RECORDS.rglob("*")):
        if not path.is_file():
            continue
        row = reference(path)
        role = "supporting_artifact"
        if path == PROTOCOL:
            role = "registered_contract"
        elif path == LOCK:
            role = "execution_lock"
        elif path == QUALIFICATION:
            role = "validation_result"
        elif path.suffix == ".md":
            role = "report"
        elif path.name == "result.json":
            role = "frozen_result"
        lines += [
            "",
            "[[records]]",
            f'path = "{path.relative_to(RECORDS.parent)}"',
            f'legacy_path = "{row["path"]}"',
            'origin = "native"',
            f'role = "{role}"',
            f'sha256 = "{row["sha256"]}"',
            f"bytes = {row['bytes']}",
            f'source_ref = "sha256:{row["sha256"]}"',
        ]
    (RECORDS.parent / "study.toml").write_text("\n".join(lines) + "\n")
