"""Frozen authority and paths for clean single-P development."""

from __future__ import annotations

from fsrl.infra.provenance import file_sha256, load_json
from fsrl.paths import RUNS_ROOT, STUDIES_ROOT

STUDY = "clean_single_p"
RECORDS = STUDIES_ROOT / STUDY / "records"
PROTOCOL = RECORDS / "benchmarks/clean_single_p_v1.json"
QUALIFICATION = RECORDS / "benchmarks/qualification.json"
QUALIFICATION_REPAIR = RECORDS / "benchmarks/evaluation_repair1_qualification.json"
SOURCE_LOCK = RECORDS / "benchmarks/source_lock.json"
SOURCE_REPAIR_LOCK = RECORDS / "benchmarks/source_repair1.json"
EVALUATION_REPAIR = RECORDS / "benchmarks/evaluation_repair1.json"
MODEL_LOCK = RECORDS / "benchmarks/model_lock.json"
RESULT = RECORDS / "results/clean_single_p_v1.json"
REPORT = RECORDS / "reports/clean_single_p_v1.md"
RUNS = RUNS_ROOT / "clean_single_p_v1"
EVALUATION_RUNS = RUNS / "evaluation-attempt2"
PROTOCOL_SHA256 = "211ed4f40fa685f250d364e215c00865175534dc7693d98aa25922f2e19f06bc"
EVALUATION_REPAIR_SHA256 = (
    "186cc9635175f1c37430e7402566f0ea9673338b290d4d202eca80c6b78cf870"
)


def specification() -> dict:
    if file_sha256(PROTOCOL) != PROTOCOL_SHA256:
        raise RuntimeError("clean single-P protocol changed")
    return load_json(PROTOCOL)


def inherited_recipe(panel: int = 1) -> dict:
    import copy

    from fsrl.experiments.observation_replication.protocol import recipe

    if panel not in specification()["design"]["evaluation_panels"]:
        raise ValueError("unregistered clean single-P panel")
    result = copy.deepcopy(recipe(panel))
    base = 731000 + panel * 100
    result["evaluation"]["generic"]["rng_seed"] = base
    result["evaluation"]["liu"].update(
        {
            key: base + offset
            for key, offset in {
                "cue_seed": 11,
                "support_seed": 12,
                "subject_encoding_seed": 13,
                "choice_seed": 14,
                "order_seed": 15,
                "query_shuffle_seed": 16,
                "evidence_shuffle_seed": 17,
            }.items()
        }
    )
    return result


def training_directory(seed: int, condition: str, arm: str):
    return RUNS / "training" / str(seed) / condition / arm


__all__ = [
    "EVALUATION_REPAIR",
    "EVALUATION_REPAIR_SHA256",
    "EVALUATION_RUNS",
    "MODEL_LOCK",
    "PROTOCOL",
    "PROTOCOL_SHA256",
    "QUALIFICATION",
    "QUALIFICATION_REPAIR",
    "RECORDS",
    "REPORT",
    "RESULT",
    "RUNS",
    "SOURCE_LOCK",
    "SOURCE_REPAIR_LOCK",
    "inherited_recipe",
    "specification",
    "training_directory",
]
