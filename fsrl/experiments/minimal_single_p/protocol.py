"""Frozen authority and paths for the minimal single-P ladder."""

from __future__ import annotations

from pathlib import Path

from fsrl.infra.provenance import file_sha256, load_json
from fsrl.paths import RUNS_ROOT, STUDIES_ROOT

STUDY = "minimal_single_p"
RECORDS = STUDIES_ROOT / STUDY / "records"
PROTOCOL = RECORDS / "benchmarks/minimal_single_p_v1.json"
QUALIFICATION = RECORDS / "benchmarks/qualification.json"
SOURCE_LOCK = RECORDS / "benchmarks/source_lock.json"
RUNS = RUNS_ROOT / "minimal_single_p_v1"
INPUTS = RUNS / "inputs"
PROTOCOL_SHA256 = "9136e5f877bcaf32737f4f20417a9edc1f77e25093c06d531dd36dd47d75cd5e"
LEVELS = ("C0", "M1", "M2", "M3", "M5_H100", "M5_H50")


def specification() -> dict:
    if file_sha256(PROTOCOL) != PROTOCOL_SHA256:
        raise RuntimeError("minimal single-P protocol changed")
    value = load_json(PROTOCOL)
    if tuple(value["design"]["ordered_levels"]) != LEVELS:
        raise RuntimeError("minimal single-P level order changed")
    return value


def validate_level(level: str) -> str:
    if level not in LEVELS:
        raise ValueError(f"unknown minimal single-P level: {level}")
    return level


def predecessor(level: str) -> str | None:
    index = LEVELS.index(validate_level(level))
    return None if index == 0 else LEVELS[index - 1]


def training_directory(seed: int, level: str) -> Path:
    return RUNS / "training" / validate_level(level) / str(seed)


def evaluation_directory(seed: int, panel: int, level: str) -> Path:
    return RUNS / "evaluation" / validate_level(level) / str(seed) / str(panel)


def model_lock_path(level: str) -> Path:
    return RECORDS / f"benchmarks/model_lock.{validate_level(level)}.json"


def result_path(level: str) -> Path:
    return RECORDS / f"results/minimal_single_p_v1.{validate_level(level)}.json"


def report_path(level: str) -> Path:
    return RECORDS / f"reports/minimal_single_p_v1.{validate_level(level)}.md"


__all__ = [
    "INPUTS",
    "LEVELS",
    "PROTOCOL",
    "PROTOCOL_SHA256",
    "QUALIFICATION",
    "RECORDS",
    "RUNS",
    "SOURCE_LOCK",
    "evaluation_directory",
    "model_lock_path",
    "predecessor",
    "report_path",
    "result_path",
    "specification",
    "training_directory",
    "validate_level",
]
