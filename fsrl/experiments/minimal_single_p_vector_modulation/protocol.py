"""Frozen authority and paths for M2 vector modulation."""

from __future__ import annotations

import json
from pathlib import Path

from fsrl.experiments.training_strategy.locks import reference
from fsrl.infra.provenance import file_sha256, load_json
from fsrl.paths import RUNS_ROOT, STUDIES_ROOT

STUDY = "minimal_single_p_vector_modulation"
RECORDS = STUDIES_ROOT / STUDY / "records"
PROTOCOL = RECORDS / "benchmarks/minimal_single_p_vector_modulation_v1.json"
QUALIFICATION = RECORDS / "benchmarks/qualification.json"
SOURCE_LOCK = RECORDS / "benchmarks/source_input_lock.json"
MODEL_LOCK = RECORDS / "benchmarks/model_lock.json"
GENERIC_RESULT = RECORDS / "results/minimal_single_p_vector_modulation_v1.generic.json"
GENERIC_REPORT = RECORDS / "reports/minimal_single_p_vector_modulation_v1.generic.md"
RESULT = RECORDS / "results/minimal_single_p_vector_modulation_v1.json"
PAIR_TABLE = RECORDS / "results/minimal_single_p_vector_modulation_v1.pairs.npz"
WRITE_TABLE = RECORDS / "results/minimal_single_p_vector_modulation_v1.writes.npz"
REPORT = RECORDS / "reports/minimal_single_p_vector_modulation_v1.md"
RUNS = RUNS_ROOT / "minimal_single_p_vector_modulation_v1"
PROTOCOL_SHA256 = "11c11848906d9e4532f19d0cb223245b84c517e9ef16ef352ec13cbdef5d3bb9"


def specification() -> dict:
    if file_sha256(PROTOCOL) != PROTOCOL_SHA256:
        raise RuntimeError("M2 vector-modulation protocol changed")
    return load_json(PROTOCOL)


def training_directory(seed: int) -> Path:
    return RUNS / "training" / str(seed)


def generic_directory(seed: int, panel: int) -> Path:
    return RUNS / "generic" / str(seed) / str(panel)


def liu_directory(seed: int, panel: int, condition: str) -> Path:
    return RUNS / "liu" / str(seed) / str(panel) / condition


def _role(path: Path) -> str:
    if path == PROTOCOL:
        return "registered_contract"
    if path == QUALIFICATION:
        return "validation_result"
    if path == SOURCE_LOCK:
        return "execution_lock"
    if path == MODEL_LOCK:
        return "artifact_lock"
    if path in {GENERIC_RESULT, RESULT}:
        return "frozen_result"
    if path.suffix == ".md":
        return "report"
    return "supporting_artifact"


def register(*, status: str = "unresolved", finding: str | None = None) -> None:
    finding = finding or "Prospectively registered vector-modulation development study."
    header = {
        "schema_version": 1,
        "id": STUDY,
        "title": "Postsynaptic vector modulation in minimal single-P M2",
        "chapter": "algorithmic_compression",
        "order": 1320,
        "status": status,
        "review_state": "indexed",
        "question": specification()["question"],
        "finding": finding,
        "boundary": (
            "P remains the only episode-persistent plastic state. This exposed "
            "three-seed development study tests heterogeneous write allocation, "
            "not confirmation, biological implementation, or main-model promotion."
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
    (RECORDS.parent / "study.toml").write_text("\n".join(lines) + "\n")


__all__ = [
    "GENERIC_REPORT",
    "GENERIC_RESULT",
    "MODEL_LOCK",
    "PAIR_TABLE",
    "PROTOCOL",
    "PROTOCOL_SHA256",
    "QUALIFICATION",
    "RECORDS",
    "REPORT",
    "RESULT",
    "RUNS",
    "SOURCE_LOCK",
    "WRITE_TABLE",
    "generic_directory",
    "liu_directory",
    "register",
    "specification",
    "training_directory",
]
