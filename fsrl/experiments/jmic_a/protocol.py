"""Frozen paths and authority for JMIC-A v1."""

from __future__ import annotations

import json
from pathlib import Path

from fsrl.experiments.training_strategy.locks import reference
from fsrl.infra.provenance import file_sha256, load_json
from fsrl.paths import STUDIES_ROOT

STUDY = "jmic_a_v1"
RECORDS = STUDIES_ROOT / STUDY / "records"
PROTOCOL = RECORDS / "benchmarks/jmic_a_v1.json"
REPAIR = RECORDS / "benchmarks/jmic_a_v1_quadrature_repair1.json"
QUALIFICATION = RECORDS / "benchmarks/qualification.json"
SOURCE_INPUT_LOCK = RECORDS / "benchmarks/source_input_lock.json"
RESULT = RECORDS / "results/jmic_a_v1.json"
PREDICTIONS = RECORDS / "results/jmic_a_v1_predictions.npz"
REPORT = RECORDS / "reports/jmic_a_v1.md"
TASK_INPUT = (
    STUDIES_ROOT
    / "magnitude_placement_human_program"
    / "records/benchmarks/magnitude_placement_behavior_v1_1.json"
)
PROTOCOL_SHA256 = "2a2724e835dd4c67bf1f98d71b76382361a4e990cf7e1f2b1c887115850080cd"
REPAIR_SHA256 = "a5ff0ce79ae6de9c145cd90cc30d3abf4e62638aa690326afe4ad3aacebf3c3b"


def specification() -> dict:
    if file_sha256(PROTOCOL) != PROTOCOL_SHA256:
        raise RuntimeError("JMIC-A protocol changed")
    if file_sha256(REPAIR) != REPAIR_SHA256:
        raise RuntimeError("JMIC-A quadrature repair changed")
    specification = load_json(PROTOCOL)
    repair = load_json(REPAIR)
    if repair["parent_protocol"]["sha256"] != PROTOCOL_SHA256:
        raise RuntimeError("JMIC-A repair parent differs")
    specification["stage_1"]["gauss_hermite_nodes"] = repair["repair"]["new"]
    return specification


def _role(path: Path) -> str:
    if path == PROTOCOL:
        return "registered_contract"
    if path == REPAIR:
        return "repair_contract"
    if path == QUALIFICATION:
        return "validation_result"
    if path == SOURCE_INPUT_LOCK:
        return "execution_lock"
    if path == RESULT:
        return "frozen_result"
    if path == REPORT:
        return "report"
    return "supporting_artifact"


def register(*, status: str = "unresolved", finding: str | None = None) -> None:
    finding = finding or "Prospectively registered computational qualification study."
    header = {
        "schema_version": 1,
        "id": STUDY,
        "title": "Analytic joint metric inference and stable commitment",
        "chapter": "algorithmic_compression",
        "order": 1330,
        "status": status,
        "review_state": "indexed",
        "question": specification()["question"],
        "finding": finding,
        "boundary": (
            "No neural network, Liu participant response file, human-phenotype "
            "fit, human collection, or JMIC-P promotion is authorized."
        ),
    }
    lines = [f"{key} = {json.dumps(value)}" for key, value in header.items()]
    for path in sorted(RECORDS.rglob("*")):
        if not path.is_file():
            continue
        row = reference(path)
        values = {
            "path": str(path.relative_to(RECORDS.parent)),
            "legacy_path": row["path"],
            "origin": "native",
            "role": _role(path),
            "sha256": row["sha256"],
            "bytes": row["bytes"],
            "source_ref": "sha256:" + row["sha256"],
        }
        lines += ["", "[[records]]"]
        lines += [f"{key} = {json.dumps(value)}" for key, value in values.items()]
    (RECORDS.parent / "study.toml").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


__all__ = [
    "PREDICTIONS",
    "PROTOCOL",
    "PROTOCOL_SHA256",
    "QUALIFICATION",
    "RECORDS",
    "REPAIR",
    "REPAIR_SHA256",
    "REPORT",
    "RESULT",
    "SOURCE_INPUT_LOCK",
    "TASK_INPUT",
    "register",
    "specification",
]
