"""Frozen authority, paths, and native registration for the M2 intervention."""

from __future__ import annotations

import json
from pathlib import Path

from fsrl.experiments.training_strategy.locks import reference
from fsrl.infra.provenance import file_sha256, load_json
from fsrl.paths import RUNS_ROOT, STUDIES_ROOT

STUDY = "minimal_single_p_observation_uncertainty"
RECORDS = STUDIES_ROOT / STUDY / "records"
PROTOCOL = RECORDS / "benchmarks/minimal_single_p_observation_uncertainty_v1.json"
QUALIFICATION = RECORDS / "benchmarks/qualification.json"
SOURCE_INPUT_LOCK = RECORDS / "benchmarks/source_input_lock.json"
RESULT = RECORDS / "results/minimal_single_p_observation_uncertainty_v1.json"
REPORT = RECORDS / "reports/minimal_single_p_observation_uncertainty_v1.md"
RUNS = RUNS_ROOT / "minimal_single_p_observation_uncertainty_v1"
INPUTS = RUNS / "inputs"
PROTOCOL_SHA256 = "a85c5aed1c7b9e0bab453322a0f5677e23f0d94a0359f9723ddf66c789eaea96"


def specification() -> dict:
    if file_sha256(PROTOCOL) != PROTOCOL_SHA256:
        raise RuntimeError("minimal single-P observation protocol changed")
    return load_json(PROTOCOL)


def evaluation_directory(seed: int, panel: int, condition: str) -> Path:
    return RUNS / "evaluation" / str(seed) / str(panel) / condition


def _role(path: Path) -> str:
    if path == PROTOCOL:
        return "registered_contract"
    if path == QUALIFICATION:
        return "validation_result"
    if path == SOURCE_INPUT_LOCK:
        return "execution_lock"
    if path == RESULT:
        return "frozen_result"
    if path.suffix == ".md":
        return "report"
    return "supporting_artifact"


def register(
    *,
    status: str = "unresolved",
    finding: str = "Prospectively registered; implementation and execution pending.",
) -> None:
    spec = specification()
    header = {
        "schema_version": 1,
        "id": STUDY,
        "title": "Acute observation uncertainty in frozen minimal single-P M2",
        "chapter": "algorithmic_compression",
        "order": 1280,
        "status": status,
        "review_state": "indexed",
        "question": spec["scientific_question"],
        "finding": finding,
        "boundary": (
            "Frozen-checkpoint development intervention across all twenty M2 networks "
            "and three inherited Liu panels, with clean, amplitude-matched sign-restored "
            "folded, and noisy conditions at fixed sigma=1/7. No training, dose selection, "
            "human fitting, checkpoint selection, or main-model promotion."
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
    "INPUTS",
    "PROTOCOL",
    "PROTOCOL_SHA256",
    "QUALIFICATION",
    "RECORDS",
    "REPORT",
    "RESULT",
    "RUNS",
    "SOURCE_INPUT_LOCK",
    "evaluation_directory",
    "register",
    "specification",
]
