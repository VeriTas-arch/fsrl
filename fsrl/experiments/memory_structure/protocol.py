"""One frozen authority for the memory-structure development study."""

from fsrl.infra.provenance import file_sha256, load_json
from fsrl.paths import RUNS_ROOT, STUDIES_ROOT

RECORD_ROOT = STUDIES_ROOT / "matched_memory_structure" / "records"
RUN_ROOT = RUNS_ROOT / "matched_memory_structure_v1"
PROTOCOL_PATH = RECORD_ROOT / "benchmarks" / "protocol.json"
PROTOCOL_SHA256 = "d68792bf141a7f92ef368789aac24556736b7f1314df21294f571e63c21b6a6d"


def specification() -> dict:
    if file_sha256(PROTOCOL_PATH) != PROTOCOL_SHA256:
        raise RuntimeError("memory-structure protocol has changed")
    return load_json(PROTOCOL_PATH)


def run_directory(seed: int, condition: str):
    seeds = specification()["seeds"]
    if seed not in seeds["mandatory"] or condition not in seeds["conditions"]:
        raise ValueError("unregistered training identity")
    return RUN_ROOT / "training" / str(seed) / condition
