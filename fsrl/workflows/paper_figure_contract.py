"""Static paths, dataset contracts, and deterministic paper-figure serializers."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fsrl.infra.provenance import file_sha256, load_json
from fsrl.infra.study_registry import resolve_record
from fsrl.paths import REPO_ROOT

ROOT = REPO_ROOT
SUITE_ROOT = ROOT / "synthesis" / "figures" / "paper_alignment"
SPECIFICATION_PATH = SUITE_ROOT / "figure_spec.json"
REPLAY_CSV_PATH = SUITE_ROOT / "source" / "model_subject_pair_accuracy.csv"
REPLAY_MANIFEST_PATH = (
    SUITE_ROOT / "source" / "model_subject_pair_accuracy.manifest.json"
)
MODEL_RESULT_PATH = resolve_record(
    "results/dual_evidence_access_confirmation_v2_4.json"
)
HUMAN_BENCHMARK_PATH = resolve_record("benchmarks/liu_human_exact_v1.json")
PROTOCOL_PATH = resolve_record("benchmarks/liu_v2.json")

DATASET_ORDER = ("human", "2104", "2105")
DATASET_LABELS = {
    "human": "Human",
    "2104": "Network 2104",
    "2105": "Network 2105",
}
DATASET_COLORS = {
    "human": "#333333",
    "2104": "#2878B5",
    "2105": "#D95F02",
}
PAIR_CLASS_ORDER = (
    "high_accuracy",
    "bimodal",
    "ordinary_unimodal",
    "low_accuracy",
    "boundary",
    "not_fit",
)
PAIR_CLASS_COLORS = {
    "high_accuracy": "#A6761D",
    "bimodal": "#1B9E77",
    "ordinary_unimodal": "#BDBDBD",
    "low_accuracy": "#D95F02",
    "boundary": "#7570B3",
    "not_fit": "#F2F2F2",
}
ITEM_COLORS = (
    "#4E79A7",
    "#F28E2B",
    "#E15759",
    "#76B7B2",
    "#59A14F",
    "#EDC948",
    "#B07AA1",
    "#FF9DA7",
)


@dataclass(frozen=True)
class Dataset:
    dataset_id: str
    subjects: list[dict]
    pair_accuracy: np.ndarray
    pair_rows: list[dict]

    @property
    def eligible(self) -> np.ndarray:
        return np.asarray(
            [subject["overall_accuracy"] >= 0.5 for subject in self.subjects]
        )

    @property
    def analysis(self) -> np.ndarray:
        return self.eligible & np.asarray(
            [subject["ranking_class"] != "correct" for subject in self.subjects]
        )


def _repo_path(value: str) -> Path:
    return ROOT / value


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _format_cell(value: object) -> object:
    if isinstance(value, (float, np.floating)):
        return format(float(value), ".12g")
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    return value


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {name: _format_cell(row.get(name, "")) for name in fieldnames}
            )


def validate_specification(path: Path = SPECIFICATION_PATH) -> dict:
    specification = load_json(path)
    if specification.get("schema_version") != 1:
        raise RuntimeError("paper-alignment specification schema must be 1")
    if tuple(specification["model"]["network_seeds"]) != (2104, 2105):
        raise RuntimeError("paper alignment requires the two frozen networks")
    registrations = {
        "paper": specification["paper"],
        "human_benchmark": specification["human"]["benchmark"],
        "model_result": specification["model"]["result"],
        "model_specification": specification["model"]["specification"],
        "model_artifact_lock": specification["model"]["artifact_lock"],
    }
    checks = []
    for name, registration in registrations.items():
        path_value = registration.get("path", registration.get("local_pdf"))
        source_path = _repo_path(path_value)
        observed = file_sha256(source_path)
        expected = registration["sha256"]
        checks.append(
            {
                "name": name,
                "path": path_value,
                "expected": expected,
                "observed": observed,
                "passed": observed == expected,
            }
        )
    if not all(check["passed"] for check in checks):
        raise RuntimeError(f"paper-alignment source validation failed: {checks}")
    return {"specification": specification, "checks": checks, "passed": True}
