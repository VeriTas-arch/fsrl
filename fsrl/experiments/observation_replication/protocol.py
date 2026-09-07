"""Fixed independent replication identities and native record ownership."""

import json

from fsrl.experiments.training_strategy.locks import reference
from fsrl.experiments.write_cost.protocol import record_role
from fsrl.infra.provenance import file_sha256, load_json
from fsrl.paths import RUNS_ROOT, STUDIES_ROOT

STUDY = "observation_replication"
RECORDS = STUDIES_ROOT / STUDY / "records"
RUNS = RUNS_ROOT / "observation_replication_v1"
PROTOCOL = RECORDS / "benchmarks/protocol.json"
SOURCE = RECORDS / "benchmarks/source_lock.json"
MODELS = RECORDS / "benchmarks/model_lock.json"
PROTOCOL_SHA256 = "e614380a31485c6748ea626c90f89233d8d3465748e5635ef15e38d4ff6cbe16"


def specification():
    if file_sha256(PROTOCOL) != PROTOCOL_SHA256:
        raise RuntimeError("cross-diagnostic protocol changed")
    return load_json(PROTOCOL)


def register(
    status="unresolved",
    finding="Prospective independent replication; outcomes not evaluated.",
):
    values = {
        "schema_version": 1,
        "id": STUDY,
        "title": "Independent observation replication",
        "chapter": "algorithmic_compression",
        "order": 1080,
        "status": status,
        "review_state": "indexed",
        "question": specification()["question"],
        "finding": finding,
        "boundary": "Three new paired training instances crossed with three new simulated panels; no network pooling, human confirmation or promotion.",
    }
    lines = [f"{k} = {json.dumps(v)}" for k, v in values.items()]
    for path in sorted(RECORDS.rglob("*")):
        if not path.is_file():
            continue
        ref = reference(path)
        row = {
            "path": str(path.relative_to(RECORDS.parent)),
            "legacy_path": ref["path"],
            "origin": "native",
            "role": "registered_contract" if path == PROTOCOL else record_role(path),
            "sha256": ref["sha256"],
            "bytes": ref["bytes"],
            "source_ref": "sha256:" + ref["sha256"],
        }
        lines += [
            "",
            "[[records]]",
            *[f"{k} = {json.dumps(v)}" for k, v in row.items()],
        ]
    (RECORDS.parent / "study.toml").write_text("\n".join(lines) + "\n")


def recipe(panel=None):
    import copy

    from fsrl.experiments.training_strategy.locks import verify_reference

    spec = specification()
    result = copy.deepcopy(
        load_json(
            verify_reference(
                spec["parent_protocol"], commit=spec["parent_witness_commit"]
            )
        )
    )
    if panel is not None:
        if panel not in spec["design"]["panels"]:
            raise ValueError("unregistered panel")
        rng = spec["rng"]
        base = rng["panel_base"] + panel * rng["panel_stride"]
        result["evaluation"]["generic"]["rng_seed"] = base + rng["generic_offset"]
        result["evaluation"]["liu"].update(
            {key: base + offset for key, offset in rng["liu_offsets"].items()}
        )
    return result


def analysis_seed(seed, panel):
    return seed + panel * 1000000
