"""Frozen authority and paths for pair-morphology attribution."""

from __future__ import annotations

import json
from pathlib import Path

from fsrl.experiments.training_strategy.locks import reference
from fsrl.infra.provenance import file_sha256, load_json
from fsrl.paths import RUNS_ROOT, STUDIES_ROOT

STUDY = "minimal_single_p_pair_morphology_attribution"
RECORDS = STUDIES_ROOT / STUDY / "records"
PROTOCOL = RECORDS / "benchmarks/minimal_single_p_pair_morphology_attribution_v1.json"
QUALIFICATION = RECORDS / "benchmarks/qualification.json"
SOURCE_INPUT_LOCK = RECORDS / "benchmarks/source_input_lock.json"
RESULT = RECORDS / "results/minimal_single_p_pair_morphology_attribution_v1.json"
PAIR_TABLE = RECORDS / "results/minimal_single_p_pair_morphology_attribution_v1.npz"
REPORT = RECORDS / "reports/minimal_single_p_pair_morphology_attribution_v1.md"
RUNS = RUNS_ROOT / "minimal_single_p_pair_morphology_attribution_v1"
PROTOCOL_SHA256 = "d2c2c3240655cdb644e04c2240c97d4c2cb2c578491241cec56bf9a83d6da58a"


def specification() -> dict:
    if file_sha256(PROTOCOL) != PROTOCOL_SHA256:
        raise RuntimeError("pair-morphology attribution protocol changed")
    return load_json(PROTOCOL)


def m2_directory(seed: int, panel: int, condition: str) -> Path:
    return (
        RUNS_ROOT
        / "minimal_single_p_observation_uncertainty_v1/evaluation"
        / str(seed)
        / str(panel)
        / condition
    )


def m2_input(panel: int, condition: str) -> Path:
    return (
        RUNS_ROOT
        / "minimal_single_p_observation_uncertainty_v1/inputs"
        / str(panel)
        / f"{condition}.npz"
    )


def score_path(seed: int) -> Path:
    return (
        STUDIES_ROOT
        / "minimal_relational_learner/records/results"
        / f"minimal_relational_learner_v1.seed-{seed}.score_only.npz"
    )


def _role(path: Path) -> str:
    if path == PROTOCOL:
        return "registered_contract"
    if path == QUALIFICATION:
        return "validation_result"
    if path == SOURCE_INPUT_LOCK:
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
    finding: str = "Prospectively registered read-only attribution; analysis pending.",
) -> None:
    header = {
        "schema_version": 1,
        "id": STUDY,
        "title": "Pair-morphology attribution in frozen minimal single-P M2",
        "chapter": "algorithmic_compression",
        "order": 1290,
        "status": status,
        "review_state": "indexed",
        "question": specification()["question"],
        "finding": finding,
        "boundary": (
            "All 180 frozen M2 observation units are mandatory. The study replays "
            "existing choices and decomposes archived output geometry without training, "
            "model inference, rescaling, threshold search, checkpoint mutation, or "
            "main-model promotion. Score-only remains a complete-recipe comparator unless "
            "exact information-interface equivalence is established."
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
    "PAIR_TABLE",
    "PROTOCOL",
    "PROTOCOL_SHA256",
    "QUALIFICATION",
    "RECORDS",
    "REPORT",
    "RESULT",
    "RUNS",
    "SOURCE_INPUT_LOCK",
    "m2_directory",
    "m2_input",
    "register",
    "score_path",
    "specification",
]
