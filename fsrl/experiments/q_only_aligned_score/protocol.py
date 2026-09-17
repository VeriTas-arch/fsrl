"""Frozen authority and paths for the aligned q-only comparator."""

from __future__ import annotations

import json
from pathlib import Path

from fsrl.experiments.training_strategy.locks import reference
from fsrl.infra.provenance import file_sha256, load_json
from fsrl.paths import RUNS_ROOT, STUDIES_ROOT

STUDY = "q_only_aligned_score_comparator"
RECORDS = STUDIES_ROOT / STUDY / "records"
PROTOCOL = RECORDS / "benchmarks/q_only_aligned_score_comparator_v1.json"
QUALIFICATION = RECORDS / "benchmarks/qualification.json"
REPAIR = RECORDS / "benchmarks/implementation_repair1.json"
REPAIR_QUALIFICATION = RECORDS / "benchmarks/qualification_repair1.json"
REPAIR2 = RECORDS / "benchmarks/implementation_repair2.json"
REPAIR2_QUALIFICATION = RECORDS / "benchmarks/qualification_repair2.json"
REPAIR3 = RECORDS / "benchmarks/implementation_repair3.json"
REPAIR3_QUALIFICATION = RECORDS / "benchmarks/qualification_repair3.json"
SOURCE_INPUT_LOCK = RECORDS / "benchmarks/source_input_lock.json"
SOURCE_REPAIR_LOCK = RECORDS / "benchmarks/source_repair1.json"
SOURCE_REPAIR2_LOCK = RECORDS / "benchmarks/source_repair2.json"
SOURCE_REPAIR3_LOCK = RECORDS / "benchmarks/source_repair3.json"
MODEL_LOCK = RECORDS / "benchmarks/model_lock.json"
GENERIC_RESULT = RECORDS / "results/q_only_aligned_score_comparator_v1.generic.json"
RESULT = RECORDS / "results/q_only_aligned_score_comparator_v1.json"
PARAMETERS = RECORDS / "results/q_only_aligned_score_comparator_v1.parameters.npz"
PAIR_TABLE = RECORDS / "results/q_only_aligned_score_comparator_v1.pairs.npz"
REPORT = RECORDS / "reports/q_only_aligned_score_comparator_v1.md"
GENERIC_REPORT = RECORDS / "reports/q_only_aligned_score_comparator_v1.generic.md"
RUNS = RUNS_ROOT / "q_only_aligned_score_comparator_v1"
PROTOCOL_SHA256 = "0971de6e4812cf4f21fd5e94ae119a96697fb8810ff77f5620d1c78de44104ab"
REPAIR_SHA256 = "24c5ad60a74e01fd68253109fa9f0097351ac5a906f300a429f8e261058bd31f"
REPAIR2_SHA256 = "8c1c0ccd7b11b23192fa8f79d836acc5d5f7493ab62f766d829156d97987d630"
REPAIR3_SHA256 = "793b41be67894a8b083a0679e8af43c80e24f62509ad6f13ba8458d780c57150"


def specification() -> dict:
    if file_sha256(PROTOCOL) != PROTOCOL_SHA256:
        raise RuntimeError("aligned score comparator protocol changed")
    return load_json(PROTOCOL)


def training_directory(seed: int) -> Path:
    if seed not in specification()["design"]["training_streams"]:
        raise ValueError("unregistered aligned-score training stream")
    return RUNS / "training" / str(seed)


def generic_directory(seed: int, panel: int) -> Path:
    return RUNS / "generic" / str(seed) / str(panel)


def liu_directory(seed: int, panel: int, condition: str) -> Path:
    return RUNS / "liu" / str(seed) / str(panel) / condition


def _role(path: Path) -> str:
    if path == PROTOCOL:
        return "registered_contract"
    if path in {
        QUALIFICATION,
        REPAIR_QUALIFICATION,
        REPAIR2_QUALIFICATION,
        REPAIR3_QUALIFICATION,
    }:
        return "validation_result"
    if path in {REPAIR, REPAIR2, REPAIR3}:
        return "repair_contract"
    if path in {
        SOURCE_INPUT_LOCK,
        SOURCE_REPAIR_LOCK,
        SOURCE_REPAIR2_LOCK,
        SOURCE_REPAIR3_LOCK,
        MODEL_LOCK,
    }:
        return "execution_lock" if path == SOURCE_INPUT_LOCK else "artifact_lock"
    if path in {GENERIC_RESULT, RESULT}:
        return "frozen_result"
    if path.suffix == ".md":
        return "report"
    return "supporting_artifact"


def register(*, status: str = "unresolved", finding: str | None = None) -> None:
    finding = (
        finding
        or "Prospectively registered information-aligned comparator; execution pending."
    )
    header = {
        "schema_version": 1,
        "id": STUDY,
        "title": "Information-aligned q-only score comparator",
        "chapter": "algorithmic_compression",
        "order": 1300,
        "status": status,
        "review_state": "indexed",
        "question": specification()["question"],
        "finding": finding,
        "boundary": (
            "All twenty parent M2 training streams are replayed by frozen batch "
            "fingerprints, and all parent generic and Liu arrays are reused directly. "
            "S_q has only a 15-dimensional episode score state and two learned scalars. "
            "The result is a complete-recipe comparison under aligned exogenous "
            "information, not an operator-only ablation or main-model promotion."
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
    "PARAMETERS",
    "PROTOCOL",
    "PROTOCOL_SHA256",
    "QUALIFICATION",
    "RECORDS",
    "REPAIR",
    "REPAIR2",
    "REPAIR2_QUALIFICATION",
    "REPAIR2_SHA256",
    "REPAIR3",
    "REPAIR3_QUALIFICATION",
    "REPAIR3_SHA256",
    "REPAIR_QUALIFICATION",
    "REPAIR_SHA256",
    "REPORT",
    "RESULT",
    "RUNS",
    "SOURCE_INPUT_LOCK",
    "SOURCE_REPAIR2_LOCK",
    "SOURCE_REPAIR3_LOCK",
    "SOURCE_REPAIR_LOCK",
    "generic_directory",
    "liu_directory",
    "register",
    "specification",
    "training_directory",
]
