"""One frozen authority for the evidence-routing development study."""

from fsrl.infra.provenance import file_sha256, load_json
from fsrl.paths import RUNS_ROOT, STUDIES_ROOT

RECORD_ROOT = STUDIES_ROOT / "weak_evidence_routing" / "records"
RUN_ROOT = RUNS_ROOT / "weak_evidence_routing_v1"
PROTOCOL_PATH = RECORD_ROOT / "benchmarks" / "protocol.json"
PROTOCOL_SHA256 = "d5c958674b6ca4985ac953cc30afb37dfd93b7b94906498bdfb25940ef400c57"


def specification() -> dict:
    if file_sha256(PROTOCOL_PATH) != PROTOCOL_SHA256:
        raise RuntimeError("evidence-routing protocol has changed")
    return load_json(PROTOCOL_PATH)


def run_directory(seed: int, condition: str):
    seeds = specification()["seeds"]
    if seed not in seeds["mandatory"] or condition not in seeds["conditions"]:
        raise ValueError("unregistered training identity")
    return RUN_ROOT / "training" / str(seed) / condition
