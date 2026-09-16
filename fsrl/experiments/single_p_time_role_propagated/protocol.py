"""Frozen authority and paths for the propagated-error successor."""

from __future__ import annotations

from fsrl.infra.provenance import file_sha256, load_json
from fsrl.paths import RUNS_ROOT, STUDIES_ROOT

STUDY = "single_p_time_role_propagated_authority"
RECORDS = STUDIES_ROOT / STUDY / "records"
PROTOCOL = RECORDS / "benchmarks/single_p_time_role_propagated_authority_v1.json"
QUALIFICATION = RECORDS / "benchmarks/qualification.json"
SOURCE_LOCK = RECORDS / "benchmarks/baseline_source_lock.json"
BASELINE_RESULT = RECORDS / "results/propagated_authority_baseline_v1.json"
BASELINE_ARTIFACT_LOCK = (
    RECORDS / "benchmarks/propagated_authority_baseline_artifact_lock.json"
)
RESULT = RECORDS / "results/single_p_time_role_propagated_authority_v1.json"
ARRAYS = RECORDS / "artifacts/single_p_time_role_propagated_authority_v1.npz"
REPORT = RECORDS / "reports/single_p_time_role_propagated_authority_v1.md"
RUNS = RUNS_ROOT / "single_p_time_role_propagated_authority_v1"
BASELINE_RUNS = RUNS / "baseline"
MECHANISM_RUNS = RUNS / "mechanism"
PROTOCOL_SHA256 = "4efac363747416ff79cdb93bedbe107ef4780bf5f4825da117d148f5dd1ca959"


def specification() -> dict:
    if file_sha256(PROTOCOL) != PROTOCOL_SHA256:
        raise RuntimeError("propagated-authority time-role protocol changed")
    return load_json(PROTOCOL)


__all__ = [
    "ARRAYS",
    "BASELINE_ARTIFACT_LOCK",
    "BASELINE_RESULT",
    "BASELINE_RUNS",
    "MECHANISM_RUNS",
    "PROTOCOL",
    "PROTOCOL_SHA256",
    "QUALIFICATION",
    "RECORDS",
    "REPORT",
    "RESULT",
    "RUNS",
    "SOURCE_LOCK",
    "specification",
]
