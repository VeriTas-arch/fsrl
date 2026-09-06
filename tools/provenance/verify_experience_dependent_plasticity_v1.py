"""Read-only audit preserving the pilot's pre-JSON report ordering."""

import json

from fsrl.experiments.adaptive_plasticity.cohorts import (
    COHORT_SHARD_SIZE,
    INPUT_LOCK,
    summarize_points,
    validate_input_lock,
)
from fsrl.experiments.adaptive_plasticity.evidence import validate_artifacts
from fsrl.experiments.adaptive_plasticity.generic_selection import validate_selection
from fsrl.experiments.adaptive_plasticity.protocol import (
    CODEBOOK,
    COHORTS,
    CONDITIONS,
    DESIGN_HASH,
    SEEDS,
)
from fsrl.experiments.adaptive_plasticity.reporting import (
    REPORT,
    RESULT,
    decision,
    render_report,
    verify_shard,
)
from fsrl.experiments.training_strategy.evaluation import json_ready
from fsrl.experiments.training_strategy.locks import reference, verify_reference
from fsrl.infra.formal_runtime import configure_formal_runtime
from fsrl.infra.provenance import load_json


def audit() -> dict:
    configure_formal_runtime()
    lock = validate_input_lock()
    artifacts = validate_artifacts()
    result = load_json(RESULT)
    if (
        result["contract_sha256"] != DESIGN_HASH
        or result["input_lock"] != reference(INPUT_LOCK)
        or result["selected_scheduler"] != lock["selected_scheduler"]
        or result["participants_pooled"]
        or result["main_model_promoted"]
    ):
        raise RuntimeError("published adaptive-plasticity result identity differs")
    checks, points = [], []
    for position, (record_ref, input_ref) in enumerate(
        zip(result["shards"], lock["cohort_shards"], strict=True)
    ):
        shard = load_json(verify_reference(record_ref))
        checks.append(
            verify_shard(
                shard,
                input_ref,
                position * COHORT_SHARD_SIZE,
                artifacts,
                lock["selected_scheduler"],
            )
        )
        points.extend(shard["points"])
    if checks != result["verification"]:
        raise RuntimeError("saved recurrence verification differs")
    fits = summarize_points(points)
    selection = validate_selection()["summary"]
    rebuilt = {"fits": fits, "decision": decision(fits, selection)}
    if any(json_ready(value) != result[key] for key, value in rebuilt.items()):
        raise RuntimeError("published adaptive-plasticity summary differs")
    # publish() rendered before sorted-key JSON serialization. Recompute the
    # original insertion order without changing the frozen report or result.
    if REPORT.read_text() != render_report({**result, **rebuilt}):
        raise RuntimeError("adaptive-plasticity report differs")
    return {
        "passed": True,
        "outcome": result["decision"]["outcome"],
        "fits": len(SEEDS) * len(CONDITIONS),
        "cohorts_per_fit": COHORTS,
        "max_recurrence_error": max(row["max_recurrence_error"] for row in checks),
        "report_exact_bytes_reconstructed": True,
        "codebook_unchanged": list(CODEBOOK),
    }


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
