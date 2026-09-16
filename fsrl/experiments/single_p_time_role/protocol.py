"""Frozen authority for the single-P time-role decomposition."""

from __future__ import annotations

from fsrl.infra.provenance import file_sha256, load_json
from fsrl.paths import RUNS_ROOT, STUDIES_ROOT

STUDY = "single_p_time_role_decomposition"
RECORDS = STUDIES_ROOT / STUDY / "records"
PROTOCOL = RECORDS / "benchmarks/single_p_time_role_decomposition_v1.json"
REPAIR = PROTOCOL.with_name("single_p_time_role_decomposition_v1.repair1.json")
QUALIFICATION = RECORDS / "benchmarks/qualification.json"
SOURCE_LOCK = RECORDS / "benchmarks/source_lock.json"
RESULT = RECORDS / "results/single_p_time_role_decomposition_v1.json"
ARRAYS = RECORDS / "artifacts/single_p_time_role_decomposition_v1.npz"
REPORT = RECORDS / "reports/single_p_time_role_decomposition_v1.md"
RUNS = RUNS_ROOT / "single_p_time_role_decomposition_v1"
PROTOCOL_SHA256 = "0d48ff52d0cdafbeb3ded6cddb5228b01b27db4bddca3481fc8c0957b38004fa"
REPAIR_SHA256 = "66d6a6eb6e6acb2d88db1dc2fe8f9b4434b3ac57d2d7fc7cde31fc17b1fb4e02"


def specification() -> dict:
    if file_sha256(PROTOCOL) != PROTOCOL_SHA256:
        raise RuntimeError("single-P time-role protocol changed")
    if file_sha256(REPAIR) != REPAIR_SHA256:
        raise RuntimeError("single-P time-role protocol repair changed")
    result = load_json(PROTOCOL)
    result["active_repair"] = load_json(REPAIR)
    return result


__all__ = [
    "ARRAYS",
    "PROTOCOL",
    "PROTOCOL_SHA256",
    "QUALIFICATION",
    "RECORDS",
    "REPAIR",
    "REPAIR_SHA256",
    "REPORT",
    "RESULT",
    "RUNS",
    "SOURCE_LOCK",
    "specification",
]
