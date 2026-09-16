"""Frozen authority for the exact P/L reparameterization study."""

from __future__ import annotations

from copy import deepcopy

from fsrl.infra.provenance import file_sha256, load_json
from fsrl.paths import STUDIES_ROOT

PROTOCOL_PATH = (
    STUDIES_ROOT
    / "pl_exact_reparameterization"
    / "records"
    / "benchmarks"
    / "pl_exact_reparameterization_v1.json"
)
PROTOCOL_SHA256 = "799fd9a4ca1814e8ad5b4662c5e0a481182cfa26969802547a949e5af46f8773"
PROTOCOL_COMMIT = "36ece9a00fba9df2df8dfa21b77f4518182fdc7f"
REPAIR_PATH = PROTOCOL_PATH.with_name("pl_exact_reparameterization_v1.repair1.json")
REPAIR_SHA256 = "fbf85c72011959c76cf7c62ab8fca7f1f5cc59a907c88787edb01e33464c60d1"
REPAIR_COMMIT = "95b86823bc028f54d3984cc6e8a2e2417585751d"
NUMERICAL_REPAIR_PATH = PROTOCOL_PATH.with_name(
    "pl_exact_reparameterization_v1.repair2.json"
)
NUMERICAL_REPAIR_SHA256 = (
    "3d8af0d10f31c5129b333a3501838e11c36be9a23ca9c6e9ae8f27649bb0161e"
)
NUMERICAL_REPAIR_COMMIT = "8eeee46f34960ce5d2a53f647c44436f7d54acf5"


def load_specification() -> dict:
    if file_sha256(PROTOCOL_PATH) != PROTOCOL_SHA256:
        raise RuntimeError("the frozen exact-reparameterization contract has changed")
    if file_sha256(REPAIR_PATH) != REPAIR_SHA256:
        raise RuntimeError("the frozen timestep-accounting repair has changed")
    if file_sha256(NUMERICAL_REPAIR_PATH) != NUMERICAL_REPAIR_SHA256:
        raise RuntimeError("the frozen numerical repair has changed")
    specification = deepcopy(load_json(PROTOCOL_PATH))
    repair = load_json(REPAIR_PATH)
    specification["active_repair"] = {
        "repair_id": repair["repair_id"],
        "path": REPAIR_PATH.relative_to(STUDIES_ROOT.parent).as_posix(),
        "sha256": REPAIR_SHA256,
        "commit": REPAIR_COMMIT,
    }
    specification["complete_timestep_accounting"] = deepcopy(repair["repair"])
    specification["successor_constraint"] = repair["successor_constraint"]
    numerical_repair = load_json(NUMERICAL_REPAIR_PATH)
    specification["active_numerical_repair"] = {
        "repair_id": numerical_repair["repair_id"],
        "path": NUMERICAL_REPAIR_PATH.relative_to(STUDIES_ROOT.parent).as_posix(),
        "sha256": NUMERICAL_REPAIR_SHA256,
        "commit": NUMERICAL_REPAIR_COMMIT,
    }
    return specification
