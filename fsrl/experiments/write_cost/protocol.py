"""Protocol identity and native record indexing for one independent study."""

from fsrl.experiments.training_strategy.locks import reference
from fsrl.infra.provenance import file_sha256, load_json
from fsrl.paths import RUNS_ROOT, STUDIES_ROOT

RECORDS = STUDIES_ROOT / "effective_write_cost" / "records"
RUNS = RUNS_ROOT / "effective_write_cost_v1"
PROTOCOL = RECORDS / "benchmarks/protocol.json"
PROTOCOL_SHA256 = "e1234f0afd67efd5fc3e55aa4e7ec06f7722a26052411eb73199f84d0308d7a0"


def specification():
    if file_sha256(PROTOCOL) != PROTOCOL_SHA256:
        raise RuntimeError("write-cost protocol changed")
    return load_json(PROTOCOL)


def register(
    status="unresolved", finding="Prospectively registered; execution pending."
):
    header = {
        "schema_version": 1,
        "id": "effective_write_cost",
        "title": "Effective global write cost and learned evidence selection",
        "chapter": "algorithmic_compression",
        "order": 1030,
        "status": status,
        "review_state": "indexed",
        "question": specification()["question"],
        "finding": finding,
        "boundary": "One generic development initialization, conditional three fresh paired initializations; unchanged shared effective evidence and joint training. Global modification proxy, no physiological budget or automatic main-model promotion.",
    }
    import json

    lines = [f"{key} = {json.dumps(value)}" for key, value in header.items()]
    for path in sorted(RECORDS.rglob("*")):
        if not path.is_file():
            continue
        row = reference(path)
        role = record_role(path)
        lines += [
            "",
            "[[records]]",
            f'path = "{path.relative_to(RECORDS.parent)}"',
            f'legacy_path = "{row["path"]}"',
            'origin = "native"',
            f'role = "{role}"',
            f'sha256 = "{row["sha256"]}"',
            f"bytes = {row['bytes']}",
            f'source_ref = "sha256:{row["sha256"]}"',
        ]
    (RECORDS.parent / "study.toml").write_text("\n".join(lines) + "\n")


def record_role(path):
    if path == PROTOCOL:
        return "registered_contract"
    if path.suffix == ".md":
        return "report"
    if path.suffix == ".pth":
        return "frozen_parameters"
    if path.name == "result.json":
        return "frozen_result"
    if "lock" in path.name:
        return "execution_lock" if "source" in path.name else "artifact_lock"
    return "supporting_artifact"
