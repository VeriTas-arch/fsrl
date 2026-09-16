"""Canonical result and concise report for direct P/L training."""

from __future__ import annotations

from pathlib import Path

from fsrl.infra.provenance import load_json, write_json_exclusive

from .evaluation import evaluation_directory, json_ready
from .locks import RECORD_ROOT, artifact_lock_path, reference, validate_artifact_lock
from .protocol import (
    PROTOCOL_SHA256,
    REPAIR_SHA256,
    load_specification,
    registered_seeds,
)


def result_path(cohort: str) -> Path:
    return RECORD_ROOT / "results" / f"pl_direct_training_v1.{cohort}.json"


def report_path(cohort: str) -> Path:
    return RECORD_ROOT / "reports" / f"pl_direct_training_v1.{cohort}.md"


def _evaluation_result(seed: int, cohort: str) -> dict:
    path = evaluation_directory(seed, cohort) / "result.json"
    if not path.is_file():
        raise RuntimeError(f"missing direct-training evaluation for {cohort}/{seed}")
    result = load_json(path)
    if result["seed"] != seed or result["cohort"] != cohort:
        raise RuntimeError("direct-training evaluation identity differs")
    return result


def _training_summary(metadata: dict) -> dict:
    return {
        "architecture": metadata["architecture"],
        "initial_shadow": metadata["initial_shadow"],
        "initial_backbone": metadata["initial_backbone"],
        "initial_shared": metadata["initial_shared"],
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


def cohort_outcome(cohort: str, rows: dict[str, dict]) -> str:
    if all(row["admitted"] for row in rows.values()):
        return (
            "development_admitted"
            if cohort == "development"
            else "direct_no_time_model_confirmed"
        )
    order = (
        "noninterpretable",
        "training_parameterization_failure",
        "no_time_recipe_failure",
        "competent_but_time_noninferior_failure",
        "alternative_no_time_organization",
        "behavior_incomplete",
    )
    observed = {row["outcome"] for row in rows.values()}
    return next(label for label in order if label in observed)


def assemble_result(cohort: str) -> dict:
    specification = load_specification()
    artifact_lock = validate_artifact_lock(cohort)
    rows = {
        str(seed): _evaluation_result(seed, cohort)
        for seed in registered_seeds(specification, cohort)
    }
    outcome = cohort_outcome(cohort, rows)
    return {
        "schema_version": 1,
        "experiment_id": specification["experiment_id"],
        "cohort": cohort,
        "outcome": outcome,
        "protocol_sha256": PROTOCOL_SHA256,
        "repair_sha256": REPAIR_SHA256,
        "artifact_lock": reference(artifact_lock_path(cohort)),
        "source_commit": artifact_lock["source_commit"],
        "seeds": [int(seed) for seed in rows],
        "all_seed_rule": "No pooling, majority vote, filtering, replacement, or result-dependent continuation.",
        "per_seed": {
            seed: {
                "outcome": row["outcome"],
                "admitted": row["admitted"],
                "paired": row["paired"],
                "paired_decision": row["paired_decision"],
                "conditions": {
                    condition: {
                        "training": _training_summary(condition_row["metadata"]),
                        "summaries": condition_row["summaries"],
                        "effects": condition_row["effects"],
                        "behavior": condition_row["behavior"],
                        "decisions": condition_row["decisions"],
                        "generic_stream_fingerprints": condition_row[
                            "generic_stream_fingerprints"
                        ],
                    }
                    for condition, condition_row in row["conditions"].items()
                },
            }
            for seed, row in rows.items()
        },
        "retention": {
            "canonical": "This JSON and its Markdown report are the canonical cohort result.",
            "runtime": "Checkpoints, full logs, sampled behavior, and dense arrays remain under the ignored locked runtime root.",
            "historical": "No historical checkpoint is deleted or superseded by this development result.",
        },
        "next_step": (
            "Run reserved seeds 3004--3006 unchanged."
            if outcome == "development_admitted"
            else "Stop this fixed recipe without tuning and retain the complete negative or mixed outcome."
            if cohort == "development"
            else "Open a separate reviewed retention migration before deleting any historical checkpoint."
            if outcome == "direct_no_time_model_confirmed"
            else "Stop without historical checkpoint cleanup."
        ),
    }


def _flag(value: bool) -> str:
    return "PASS" if value else "FAIL"


def report_text(result: dict) -> str:
    lines = [
        f"# Direct P/L training: {result['cohort']}",
        "",
        f"Registered outcome: **{result['outcome']}**.",
        "",
        "Both conditions were freshly initialized from paired shadow-legacy draws and trained for 1,500 joint updates. The control retains separate normalized time; the candidate has no time parameter or time argument. Both retain four support steps, two query steps, a 40,000-scalar P state, and a 105-scalar packed L state.",
        "",
        "| Seed | Condition | Competence | P/L organization | Behavior | Paired noninferiority | Local gain |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for seed, row in result["per_seed"].items():
        for condition, condition_row in row["conditions"].items():
            decision = condition_row["decisions"]
            lines.append(
                f"| {seed} | {condition} | {_flag(decision['competence']['passed'])} | {_flag(decision['mechanism']['passed'])} | {_flag(decision['behavior']['passed'])} | {_flag(row['paired_decision']['passed'])} | {condition_row['training']['local_gain']:.6g} |"
            )
    lines.extend(
        [
            "",
            "## Boundaries",
            "",
            "Every gate is evaluated within seed and condition. Participant uncertainty is not network-population uncertainty, and no successful network repairs a failed mandatory network. Human intervals and temperature are inherited descriptive references rather than newly fitted targets.",
            "",
            f"Next step: {result['next_step']}",
            "",
            "The canonical JSON retains endpoint estimates, bootstrap bounds, behavior flags, stream fingerprints, tensor identities, optimizer counters, and runtime costs. Dense arrays and checkpoints remain hash-locked runtime artifacts.",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(cohort: str) -> dict:
    result = assemble_result(cohort)
    result_target = result_path(cohort)
    report_target = report_path(cohort)
    result_target.parent.mkdir(parents=True, exist_ok=True)
    report_target.parent.mkdir(parents=True, exist_ok=True)
    write_json_exclusive(result_target, json_ready(result))
    with report_target.open("x", encoding="utf-8") as handle:
        handle.write(report_text(result))
    return {
        "cohort": cohort,
        "outcome": result["outcome"],
        "result": reference(result_target),
        "report": reference(report_target),
    }
