"""Read-only v1 audit preserving its pre-JSON publication order and sources."""

import json

from fsrl.experiments.cohort_diagnostic.execution import verify_shard
from fsrl.experiments.cohort_diagnostic.locks import LOCK, validate_lock
from fsrl.experiments.cohort_diagnostic.protocol import (
    PROTOCOL_HASH,
    load_parameters,
    specification,
)
from fsrl.experiments.cohort_diagnostic.reporting import (
    REPORT,
    RESULT,
    render_report,
    summarize,
)
from fsrl.experiments.training_strategy.evaluation import json_ready
from fsrl.experiments.training_strategy.locks import reference, verify_reference
from fsrl.infra.formal_runtime import configure_formal_runtime
from fsrl.infra.provenance import load_json


def check_summary_and_report(result: dict, rebuilt: dict, saved_report: str) -> None:
    for key, value in rebuilt.items():
        if json_ready(value) != result[key]:
            raise RuntimeError("cohort diagnostic statistics do not reconstruct")
    # publish() rendered before sorted-key JSON serialization. Recompute that
    # original insertion order, without sorting or weakening text comparison.
    if saved_report != render_report({**result, **rebuilt}):
        raise RuntimeError("cohort diagnostic report differs")


def audit() -> dict:
    configure_formal_runtime()
    lock = validate_lock()
    result = load_json(RESULT)
    if (
        result["execution_lock"] != reference(LOCK)
        or result["protocol_sha256"] != PROTOCOL_HASH
        or len(result["shards"]) != len(lock["cohort_shards"])
    ):
        raise RuntimeError("result lock, protocol or shard inventory differs")
    configs = load_parameters()
    shards = [load_json(verify_reference(ref)) for ref in result["shards"]]
    size = specification()["cohorts"]["shard_size"]
    checks = [
        verify_shard(shard, input_ref, index * size, configs)
        for index, (shard, input_ref) in enumerate(
            zip(shards, lock["cohort_shards"], strict=True)
        )
    ]
    if checks != result["verification"]:
        raise RuntimeError("saved independent verification differs")
    check_summary_and_report(result, summarize(shards), REPORT.read_text())
    return {
        "passed": True,
        "fits": len(configs),
        "cohorts_per_fit": lock["cohort_count"],
        "outcome": result["outcome"],
        "max_recurrence_error": max(row["max_recurrence_error"] for row in checks),
        "report_exact_bytes_reconstructed": True,
    }


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
