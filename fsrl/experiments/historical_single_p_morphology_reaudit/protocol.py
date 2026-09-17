"""Frozen authority and paths for the historical morphology re-audit."""

from __future__ import annotations

import json
from pathlib import Path

from fsrl.experiments.training_strategy.locks import reference
from fsrl.infra.provenance import file_sha256, load_json
from fsrl.paths import RUNS_ROOT, STUDIES_ROOT

STUDY = "historical_single_p_morphology_reaudit"
RECORDS = STUDIES_ROOT / STUDY / "records"
PROTOCOL = RECORDS / "benchmarks/historical_single_p_morphology_reaudit_v1.json"
ORIGINAL_QUALIFICATION = RECORDS / "benchmarks/qualification.json"
REPAIR1_QUALIFICATION = RECORDS / "benchmarks/qualification_repair1.json"
REPAIR2_QUALIFICATION = RECORDS / "benchmarks/qualification_repair2.json"
QUALIFICATION = RECORDS / "benchmarks/qualification_repair3.json"
REPAIR1 = RECORDS / "benchmarks/implementation_repair1.json"
REPAIR2 = RECORDS / "benchmarks/implementation_repair2.json"
REPAIR = RECORDS / "benchmarks/implementation_repair3.json"
ORIGINAL_SOURCE_INPUT_LOCK = RECORDS / "benchmarks/source_input_lock.json"
REPAIR1_SOURCE_INPUT_LOCK = RECORDS / "benchmarks/source_input_lock_repair1.json"
REPAIR2_SOURCE_INPUT_LOCK = RECORDS / "benchmarks/source_input_lock_repair2.json"
SOURCE_INPUT_LOCK = RECORDS / "benchmarks/source_input_lock_repair3.json"
RESULT = RECORDS / "results/historical_single_p_morphology_reaudit_v1.json"
PAIR_TABLE = RECORDS / "results/historical_single_p_morphology_reaudit_v1.pairs.npz"
REPORT = RECORDS / "reports/historical_single_p_morphology_reaudit_v1.md"
RUNS = RUNS_ROOT / "historical_single_p_morphology_reaudit_v1"
PROTOCOL_SHA256 = "bc99e6e55a6ec2800f74bb0cc932a988c9d371f05e6c26c9e4e2f0afe9c3d826"
REPAIR_SHA256 = "672c2037aa39992ed6bcb29f4e367539c2202faca7c1881fabb4cad6005370c0"


def specification() -> dict:
    if file_sha256(PROTOCOL) != PROTOCOL_SHA256:
        raise RuntimeError("historical morphology re-audit protocol changed")
    return load_json(PROTOCOL)


def unit_directory(family: str, seed: int, panel: int, condition: str) -> Path:
    if family == "local_memory_removal":
        return (
            STUDIES_ROOT
            / family
            / "records/artifacts/evaluation"
            / str(seed)
            / str(panel)
            / condition
        )
    if family == "clean_single_p":
        return (
            RUNS_ROOT
            / "clean_single_p_v1/evaluation-attempt2"
            / str(seed)
            / str(panel)
            / "clean_no_time"
            / condition
        )
    raise ValueError(f"unknown historical family: {family}")


def _role(path: Path) -> str:
    if path == PROTOCOL:
        return "registered_contract"
    if path in {
        ORIGINAL_QUALIFICATION,
        REPAIR1_QUALIFICATION,
        REPAIR2_QUALIFICATION,
        QUALIFICATION,
    }:
        return "validation_result"
    if path in {REPAIR1, REPAIR2, REPAIR}:
        return "repair_contract"
    if path in {
        ORIGINAL_SOURCE_INPUT_LOCK,
        REPAIR1_SOURCE_INPUT_LOCK,
        REPAIR2_SOURCE_INPUT_LOCK,
        SOURCE_INPUT_LOCK,
    }:
        return "execution_lock"
    if path == RESULT:
        return "frozen_result"
    if path == PAIR_TABLE:
        return "supporting_artifact"
    if path.suffix == ".md":
        return "report"
    return "supporting_artifact"


def register(
    *,
    status: str = "unresolved",
    finding: str = "Prospectively registered read-only historical morphology re-audit; analysis pending.",
) -> None:
    header = {
        "schema_version": 1,
        "id": STUDY,
        "title": "Current-standard morphology re-audit of historical alpha single-P",
        "chapter": "algorithmic_compression",
        "order": 1320,
        "status": status,
        "review_state": "indexed",
        "question": specification()["question"],
        "finding": finding,
        "boundary": (
            "All 36 archived A0/Ae units from the two specified historical families "
            "are mandatory. The study replays existing choices and analyzes archived "
            "output geometry without training, checkpoint loading, model inference, "
            "rescaling, threshold search, selection, historical-record mutation, or "
            "main-model promotion."
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
    "ORIGINAL_QUALIFICATION",
    "ORIGINAL_SOURCE_INPUT_LOCK",
    "PAIR_TABLE",
    "PROTOCOL",
    "PROTOCOL_SHA256",
    "QUALIFICATION",
    "RECORDS",
    "REPAIR",
    "REPAIR1",
    "REPAIR1_QUALIFICATION",
    "REPAIR1_SOURCE_INPUT_LOCK",
    "REPAIR2",
    "REPAIR2_QUALIFICATION",
    "REPAIR2_SOURCE_INPUT_LOCK",
    "REPAIR_SHA256",
    "REPORT",
    "RESULT",
    "RUNS",
    "SOURCE_INPUT_LOCK",
    "register",
    "specification",
    "unit_directory",
]
