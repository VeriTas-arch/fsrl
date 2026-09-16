"""Frozen authority for the single-P time-role decomposition."""

from __future__ import annotations

from fsrl.infra.provenance import file_sha256, load_json
from fsrl.paths import RUNS_ROOT, STUDIES_ROOT

STUDY = "single_p_time_role_decomposition"
RECORDS = STUDIES_ROOT / STUDY / "records"
PROTOCOL = RECORDS / "benchmarks/single_p_time_role_decomposition_v1.json"
REPAIR = PROTOCOL.with_name("single_p_time_role_decomposition_v1.repair1.json")
IMPLEMENTATION_REPAIR = PROTOCOL.with_name(
    "single_p_time_role_decomposition_v1.repair2.json"
)
QUALIFICATION_REPAIR = PROTOCOL.with_name(
    "single_p_time_role_decomposition_v1.repair3.json"
)
QUALIFICATION_FIX = PROTOCOL.with_name(
    "single_p_time_role_decomposition_v1.repair4.json"
)
EXECUTION_REPAIR = PROTOCOL.with_name(
    "single_p_time_role_decomposition_v1.repair5.json"
)
QUALIFICATION = RECORDS / "benchmarks/qualification_v4.json"
SOURCE_LOCK = RECORDS / "benchmarks/source_lock_v2.json"
RESULT = RECORDS / "results/single_p_time_role_decomposition_v1.json"
ARRAYS = RECORDS / "artifacts/single_p_time_role_decomposition_v1.npz"
REPORT = RECORDS / "reports/single_p_time_role_decomposition_v1.md"
RUNS = RUNS_ROOT / "single_p_time_role_decomposition_v1"
PROTOCOL_SHA256 = "0d48ff52d0cdafbeb3ded6cddb5228b01b27db4bddca3481fc8c0957b38004fa"
REPAIR_SHA256 = "66d6a6eb6e6acb2d88db1dc2fe8f9b4434b3ac57d2d7fc7cde31fc17b1fb4e02"
IMPLEMENTATION_REPAIR_SHA256 = (
    "ad736bc5ed660f6717b3d79daafeb9905be9175579d96beb6f6b639a7c40c654"
)
QUALIFICATION_REPAIR_SHA256 = (
    "d35b60c590d6f80083d6ddc8f368dda21eae65ba65195a236df112f732cf7f67"
)
QUALIFICATION_FIX_SHA256 = (
    "81ae1a1bc1f23cab1f9c70794544ac3d6a0daf1dfb5e21565154c6e638bfbf61"
)
EXECUTION_REPAIR_SHA256 = (
    "cdbf6bcf64eeaf6c339f67a50f6dd3f263520e84b58a014f5611a79963c3c9b0"
)


def specification() -> dict:
    if file_sha256(PROTOCOL) != PROTOCOL_SHA256:
        raise RuntimeError("single-P time-role protocol changed")
    if file_sha256(REPAIR) != REPAIR_SHA256:
        raise RuntimeError("single-P time-role protocol repair changed")
    if file_sha256(IMPLEMENTATION_REPAIR) != IMPLEMENTATION_REPAIR_SHA256:
        raise RuntimeError("single-P time-role implementation repair changed")
    if file_sha256(QUALIFICATION_REPAIR) != QUALIFICATION_REPAIR_SHA256:
        raise RuntimeError("single-P time-role qualification repair changed")
    if file_sha256(QUALIFICATION_FIX) != QUALIFICATION_FIX_SHA256:
        raise RuntimeError("single-P time-role qualification fixture repair changed")
    if file_sha256(EXECUTION_REPAIR) != EXECUTION_REPAIR_SHA256:
        raise RuntimeError("single-P time-role execution repair changed")
    result = load_json(PROTOCOL)
    result["active_repair"] = load_json(REPAIR)
    result["active_implementation_repair"] = load_json(IMPLEMENTATION_REPAIR)
    result["active_qualification_repair"] = load_json(QUALIFICATION_REPAIR)
    result["active_qualification_fix"] = load_json(QUALIFICATION_FIX)
    result["active_execution_repair"] = load_json(EXECUTION_REPAIR)
    return result


__all__ = [
    "ARRAYS",
    "EXECUTION_REPAIR",
    "EXECUTION_REPAIR_SHA256",
    "IMPLEMENTATION_REPAIR",
    "IMPLEMENTATION_REPAIR_SHA256",
    "PROTOCOL",
    "PROTOCOL_SHA256",
    "QUALIFICATION",
    "QUALIFICATION_FIX",
    "QUALIFICATION_FIX_SHA256",
    "QUALIFICATION_REPAIR",
    "QUALIFICATION_REPAIR_SHA256",
    "RECORDS",
    "REPAIR",
    "REPAIR_SHA256",
    "REPORT",
    "RESULT",
    "RUNS",
    "SOURCE_LOCK",
    "specification",
]
