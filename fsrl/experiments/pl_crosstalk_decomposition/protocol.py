"""Frozen authority for the exact P/L cross-talk decomposition."""

from __future__ import annotations

from fsrl.infra.provenance import file_sha256, load_json
from fsrl.paths import STUDIES_ROOT

PROTOCOL_PATH = (
    STUDIES_ROOT
    / "pl_crosstalk_decomposition"
    / "records"
    / "benchmarks"
    / "pl_crosstalk_decomposition_v1.json"
)
PROTOCOL_SHA256 = "0a8aa21c8798811817d980c77701193ceedc9c75c7644eb4dd32577c54dd27ba"
PROTOCOL_COMMIT = "3d9784f1427e7e0961d62acb3d207ec857a1dca2"
REPAIR_PATH = PROTOCOL_PATH.with_name("pl_crosstalk_decomposition_v1.repair1.json")
REPAIR_SHA256 = "5fbc3dde30e0aea87a01b377ef42cfe85d9608f794f8a02fbc9c63448de38282"
REPAIR_COMMIT = "7bd3544582390bb12adc99441b03d3d1e8dda23c"


def load_specification() -> dict:
    if file_sha256(PROTOCOL_PATH) != PROTOCOL_SHA256:
        raise RuntimeError("the frozen cross-talk decomposition contract changed")
    if file_sha256(REPAIR_PATH) != REPAIR_SHA256:
        raise RuntimeError("the frozen cross-talk implementation repair changed")
    specification = load_json(PROTOCOL_PATH)
    if specification["registration_parent_commit"] != (
        "a95648553dafba6d4b56377cc19a55fbb32b1efd"
    ):
        raise RuntimeError("the decomposition cites a different parent freeze")
    repair = load_json(REPAIR_PATH)
    if repair["original_protocol"]["sha256"] != PROTOCOL_SHA256:
        raise RuntimeError("the implementation repair cites a different protocol")
    specification["active_repair"] = {
        "repair_id": repair["repair_id"],
        "path": REPAIR_PATH.relative_to(STUDIES_ROOT.parent).as_posix(),
        "sha256": REPAIR_SHA256,
        "commit": REPAIR_COMMIT,
    }
    return specification


def registered_seeds(specification: dict | None = None) -> tuple[int, ...]:
    frozen = load_specification() if specification is None else specification
    return tuple(int(seed) for seed in frozen["design"]["mandatory_network_seeds"])
