"""Canonical result and concise report for functional replication."""

from __future__ import annotations

from fsrl.infra.provenance import load_json, write_json_exclusive

from .estimands import cohort_outcome
from .evaluation import evaluation_directory, json_ready
from .locks import (
    ARTIFACT_LOCK_PATH,
    RECORD_ROOT,
    reference,
    validate_artifact_lock,
)
from .protocol import (
    EXECUTION_REPAIR_PATH,
    PROTOCOL_SHA256,
    REPAIR_SHA256,
    load_execution_repair,
    load_specification,
    registered_seeds,
)
from .transport import transport_directory, validate_transport

RESULT_PATH = RECORD_ROOT / "results" / "pl_functional_replication_v1.json"
REPORT_PATH = RECORD_ROOT / "reports" / "pl_functional_replication_v1.md"


def _training_summary(metadata: dict) -> dict:
    return {
        "architecture": metadata["architecture"],
        "initial_shadow": metadata["initial_shadow"],
        "initial_backbone": metadata["initial_backbone"],
        "initial_local": metadata["initial_local"],
        "final_backbone": metadata["final_backbone"],
        "final_local": metadata["final_local"],
        "stream_fingerprint": metadata["stream_fingerprint"],
        "local_gain": metadata["local_gain"],
        "episode_exposures": metadata["episode_exposures"],
        "checkpoint_sha256": metadata["checkpoint"]["sha256"],
        "checkpoint_bytes": metadata["checkpoint"]["bytes"],
        "optimizer_parameter_steps": metadata["optimizer_parameter_steps"],
        "cost": metadata["cost"],
        "training_seconds": metadata["training_seconds"],
        "warmup_seconds": metadata["warmup_seconds"],
        "peak_allocated_bytes": metadata["peak_allocated_bytes"],
        "peak_reserved_bytes": metadata["peak_reserved_bytes"],
    }


def _evaluation_result(seed: int) -> dict:
    result = load_json(evaluation_directory(seed) / "result.json")
    if result["seed"] != seed:
        raise RuntimeError("functional evaluation seed identity differs")
    return result


def _transport_summary(transport: dict) -> dict:
    return {
        "outcome": transport["decision"]["outcome"],
        "decision": transport["decision"],
        "graph_validation": transport["graph_validation"],
        "per_seed_size": {
            seed: {
                size: {
                    "decision": cell["decision"],
                    "integrity": cell["integrity"],
                    "size_specific_metrics": cell["size_specific_metrics"],
                    "condition_summaries": {
                        condition: values["summary"]
                        for condition, values in cell["metrics"]["conditions"].items()
                    },
                    "constructive": cell["metrics"]["constructive"]["summary"],
                    "global_relation_LOO": cell["metrics"]["global_relation_LOO"][
                        "summary"
                    ],
                    "contrasts": cell["metrics"]["contrasts"],
                    "local_exactness": cell["metrics"]["local_exactness"],
                }
                for size, cell in seed_row["sizes"].items()
            }
            for seed, seed_row in transport["seeds"].items()
        },
        "runtime_result": reference(transport_directory() / "result.json"),
    }


def assemble_result() -> dict:
    specification = load_specification()
    artifact_lock = validate_artifact_lock()
    rows = {
        str(seed): _evaluation_result(seed) for seed in registered_seeds(specification)
    }
    outcome = cohort_outcome(rows)
    primary_replication = all(
        row["primary_functional_replication"] for row in rows.values()
    )
    if primary_replication:
        transport = _transport_summary(validate_transport(artifact_lock))
        transport_triggered = True
    else:
        transport = {
            "outcome": "not_triggered",
            "reason": "At least one fresh seed failed primary functional replication.",
        }
        transport_triggered = False
    return {
        "schema_version": 1,
        "experiment_id": specification["experiment_id"],
        "outcome": outcome,
        "protocol_sha256": PROTOCOL_SHA256,
        "repair_sha256": REPAIR_SHA256,
        "artifact_lock": reference(ARTIFACT_LOCK_PATH),
        "source_commit": artifact_lock["source_commit"],
        "evaluation_source_commit": artifact_lock["active_evaluation_source_commit"],
        "source_repair": artifact_lock["source_repair"],
        "execution_repair": reference(EXECUTION_REPAIR_PATH),
        "noninterpretable_attempt": load_execution_repair()["failure"],
        "seeds": list(registered_seeds(specification)),
        "all_seed_rule": "No pooling, majority vote, filtering, replacement, or successful-seed repair.",
        "primary_functional_replication": primary_replication,
        "transport_triggered": transport_triggered,
        "per_seed": {
            seed: {
                "outcome": row["outcome"],
                "primary_functional_replication": row["primary_functional_replication"],
                "training": _training_summary(row["metadata"]),
                "integrity": row["integrity"],
                "competence": row["competence"],
                "global_path": row["global_path"],
                "four_v2_4_links": row["four_v2_4_links"],
                "omitted_local_materiality": row["omitted_local_materiality"],
                "retained_local_materiality_diagnostic": row[
                    "retained_local_materiality_diagnostic"
                ],
                "contrasts": row["contrasts"],
                "probability_summaries": {
                    condition: values["summary"]
                    for condition, values in row["probability"].items()
                },
                "causal_field_summaries": {
                    condition: values["summary"]
                    for condition, values in row["causal_fields"].items()
                },
                "summaries": row["summaries"],
                "behavior": row["behavior"],
                "P_off_dual_nonlearned_sampled": row["P_off_dual_nonlearned_sampled"],
                "generic_stream_fingerprints": row["generic_stream_fingerprints"],
                "runtime_result": reference(
                    evaluation_directory(int(seed)) / "result.json"
                ),
                "raw_arrays": row["raw_arrays"],
                "sampled_behavior": row["sampled_behavior"],
            }
            for seed, row in rows.items()
        },
        "triggered_item_count_transport": transport,
        "claim_boundary": specification["claim_boundary"],
        "retention": {
            "canonical": "This JSON and its Markdown report are the canonical study result.",
            "runtime": "Checkpoints, logs, sampled behavior, dense arrays, and any triggered transport remain under the ignored artifact root and are hash-locked.",
            "historical": "No historical checkpoint is deleted, relabeled, or superseded by this study; cleanup requires a separate retention migration.",
        },
        "next_step": (
            "Review promotion and open a separate retention migration; do not infer biological or structural necessity."
            if outcome == "clean_no_time_pl_functional_replication"
            else "Retain the complete result and stop this frozen protocol without tuning."
        ),
    }


def _flag(value: bool) -> str:
    return "PASS" if value else "FAIL"


def report_text(result: dict) -> str:
    lines = [
        "# Clean no-time P/L functional replication",
        "",
        f"Registered outcome: **{result['outcome']}**.",
        "",
        "All three candidates used the frozen 32-channel, no-time architecture, four support microsteps, two query microsteps, 40,000-scalar P state, 105-scalar packed L state, and 1,500 joint updates. Seeds 3001--3003 remain the separate historical training-parameterization failure.",
        "",
        "The first seed-3004 evaluation attempt stopped before any result write because an integrity diagnostic did not restore serialized nulls to NaN. The failed attempt is retained as noninterpretable; attempt2 replays every seed over the unchanged jointly locked checkpoints under the append-only source repair.",
        "",
        "| Seed | Competence | Global P path | Four v2.4 links | Omitted L materiality | 9/9 qualitative | Quantitative calibration (report only) | Local gain |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for seed, row in result["per_seed"].items():
        behavior = row["behavior"]
        lines.append(
            f"| {seed} | {_flag(row['competence']['passed'])} | {_flag(row['global_path']['passed'])} | {_flag(row['four_v2_4_links']['passed'])} | {_flag(row['omitted_local_materiality']['passed'])} | {_flag(behavior['all_nine_qualitative_pass'])} | {behavior['quantitative_calibration_pass_count_report_only']}/{behavior['quantitative_calibration_total']} | {row['training']['local_gain']:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Triggered transport",
            "",
            f"Item-count transport: **{result['triggered_item_count_transport']['outcome']}**. It was evaluated only when all three primary functional replications passed and cannot repair the N=8 decision.",
            "",
            "## Boundary",
            "",
            "A pass supports the imposed functional division between P-mediated global assembly and query-addressed L-mediated direct fidelity in this frozen task and seed scope. It does not establish biological dual stores, architectural necessity or minimality, L necessity for every direct relation, quantitative human equivalence, spontaneous emergence without priors, or network-population prevalence.",
            "",
            f"Next step: {result['next_step']}",
            "",
        ]
    )
    return "\n".join(lines)


def write_report() -> dict:
    result = assemble_result()
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_json_exclusive(RESULT_PATH, json_ready(result))
    with REPORT_PATH.open("x", encoding="utf-8") as handle:
        handle.write(report_text(result))
    return {
        "outcome": result["outcome"],
        "result": reference(RESULT_PATH),
        "report": reference(REPORT_PATH),
    }
