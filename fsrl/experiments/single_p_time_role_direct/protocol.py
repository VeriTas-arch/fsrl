"""Frozen authority and paths for the direct-authority successor."""

from __future__ import annotations

from fsrl.infra.provenance import file_sha256, load_json
from fsrl.paths import RUNS_ROOT, STUDIES_ROOT

STUDY = "single_p_time_role_direct_authority"
RECORDS = STUDIES_ROOT / STUDY / "records"
PROTOCOL = RECORDS / "benchmarks/single_p_time_role_direct_authority_v1.json"
QUALIFICATION = RECORDS / "benchmarks/qualification.json"
SOURCE_LOCK = RECORDS / "benchmarks/baseline_source_lock.json"
BASELINE_RESULT = RECORDS / "results/direct_authority_baseline_v1.json"
BASELINE_ARTIFACT_LOCK = (
    RECORDS / "benchmarks/direct_authority_baseline_artifact_lock.json"
)
RESULT = RECORDS / "results/single_p_time_role_direct_authority_v1.json"
ARRAYS = RECORDS / "artifacts/single_p_time_role_direct_authority_v1.npz"
REPORT = RECORDS / "reports/single_p_time_role_direct_authority_v1.md"
RUNS = RUNS_ROOT / "single_p_time_role_direct_authority_v1"
BASELINE_RUNS = RUNS / "baseline"
MECHANISM_RUNS = RUNS / "mechanism"
PROTOCOL_SHA256 = "e5237fa724e4ed5c95ab8083319d9fc3e7689b3768b83126246edb145b9ebe91"


def specification() -> dict:
    if file_sha256(PROTOCOL) != PROTOCOL_SHA256:
        raise RuntimeError("direct-authority time-role protocol changed")
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
