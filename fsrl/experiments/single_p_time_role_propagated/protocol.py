"""Frozen authority and paths for the propagated-error successor."""

from __future__ import annotations

from fsrl.infra.provenance import file_sha256, load_json
from fsrl.paths import RUNS_ROOT, STUDIES_ROOT

STUDY = "single_p_time_role_propagated_authority"
RECORDS = STUDIES_ROOT / STUDY / "records"
PROTOCOL = RECORDS / "benchmarks/single_p_time_role_propagated_authority_v1.json"
REPAIRS = (
    PROTOCOL.with_name("single_p_time_role_propagated_authority_v1.repair1.json"),
    PROTOCOL.with_name("single_p_time_role_propagated_authority_v1.repair2.json"),
    PROTOCOL.with_name("single_p_time_role_propagated_authority_v1.repair3.json"),
    PROTOCOL.with_name("single_p_time_role_propagated_authority_v1.repair4.json"),
    PROTOCOL.with_name("single_p_time_role_propagated_authority_v1.repair5.json"),
)
QUALIFICATION_V1 = RECORDS / "benchmarks/qualification.json"
QUALIFICATION_V2 = RECORDS / "benchmarks/qualification_v2.json"
QUALIFICATION_V3 = RECORDS / "benchmarks/qualification_v3.json"
QUALIFICATION_V4 = RECORDS / "benchmarks/qualification_v4.json"
QUALIFICATION_V5 = RECORDS / "benchmarks/qualification_v5.json"
QUALIFICATION = RECORDS / "benchmarks/qualification_v6.json"
SOURCE_LOCK_V1 = RECORDS / "benchmarks/baseline_source_lock.json"
EXECUTION_SOURCE_LOCK = RECORDS / "benchmarks/baseline_source_lock_v3.json"
SOURCE_LOCK = RECORDS / "benchmarks/baseline_finalize_source_lock_v2.json"
MECHANISM_SOURCE_LOCK = RECORDS / "benchmarks/mechanism_source_lock.json"
ATTEMPT2 = RECORDS / "results/baseline_attempt2.json"
ATTEMPT2_ARTIFACT_LOCK = RECORDS / "benchmarks/baseline_attempt2_artifact_lock.json"
BASELINE_RESULT = RECORDS / "results/propagated_authority_baseline_v1.json"
BASELINE_ARTIFACT_LOCK = (
    RECORDS / "benchmarks/propagated_authority_baseline_artifact_lock.json"
)
MECHANISM_ATTEMPT1 = RECORDS / "results/mechanism_attempt1.json"
RESULT = RECORDS / "results/single_p_time_role_propagated_authority_v1.json"
ARRAYS = RECORDS / "artifacts/single_p_time_role_propagated_authority_v1.npz"
REPORT = RECORDS / "reports/single_p_time_role_propagated_authority_v1.md"
RUNS = RUNS_ROOT / "single_p_time_role_propagated_authority_v1"
BASELINE_RUNS = RUNS / "baseline"
MECHANISM_RUNS = RUNS / "mechanism"
PROTOCOL_SHA256 = "4efac363747416ff79cdb93bedbe107ef4780bf5f4825da117d148f5dd1ca959"
REPAIR_SHA256S = (
    "eca37de2c133926fee96d2ac1c387b3b59de7878415e22d03608149d3eaa05ce",
    "5a6af0e05ca3ad0b520df303aad4877ba1bb893277c62fed2ced2afa87d5c66b",
    "40752a85fdf553b6a0d5de71ff4cbe581b8f1e17d4aec0cd746e307c97c52178",
    "7663201646bed7c11c0574f1d29561f65504500331c1197bcf975ed5f22b40d2",
    "1c83bffb6be550b581cdca6f77c899f522c3a0a8dfd3c4bfa2400e05098e6f08",
)


def specification() -> dict:
    if file_sha256(PROTOCOL) != PROTOCOL_SHA256:
        raise RuntimeError("propagated-authority time-role protocol changed")
    if tuple(file_sha256(path) for path in REPAIRS) != REPAIR_SHA256S:
        raise RuntimeError("propagated-authority verifier repairs changed")
    result = load_json(PROTOCOL)
    result["active_repairs"] = [load_json(path) for path in REPAIRS]
    return result


__all__ = [
    "ARRAYS",
    "ATTEMPT2",
    "ATTEMPT2_ARTIFACT_LOCK",
    "BASELINE_ARTIFACT_LOCK",
    "BASELINE_RESULT",
    "BASELINE_RUNS",
    "EXECUTION_SOURCE_LOCK",
    "MECHANISM_ATTEMPT1",
    "MECHANISM_RUNS",
    "MECHANISM_SOURCE_LOCK",
    "PROTOCOL",
    "PROTOCOL_SHA256",
    "QUALIFICATION",
    "QUALIFICATION_V1",
    "QUALIFICATION_V2",
    "QUALIFICATION_V3",
    "QUALIFICATION_V4",
    "QUALIFICATION_V5",
    "RECORDS",
    "REPAIRS",
    "REPAIR_SHA256S",
    "REPORT",
    "RESULT",
    "RUNS",
    "SOURCE_LOCK",
    "SOURCE_LOCK_V1",
    "specification",
]
