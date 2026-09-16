"""Canonical report-only promotion of compact-model cohort results."""

from __future__ import annotations

from pathlib import Path

from fsrl.infra.provenance import load_json, write_json_exclusive
from fsrl.paths import STUDIES_ROOT

from .evaluation import json_ready
from .locks import RUN_ROOT, artifact_lock_path, reference, validate_artifact_lock
from .protocol import (
    PROTOCOL_SHA256,
    REPAIR_SHA256,
    load_specification,
    registered_seeds,
)

RECORD_ROOT = STUDIES_ROOT / "compact_global_local_model" / "records"


def result_path(cohort: str) -> Path:
    return RECORD_ROOT / "results" / f"compact_global_local_model_v1.{cohort}.json"


def report_path(cohort: str) -> Path:
    return RECORD_ROOT / "reports" / f"compact_global_local_model_v1.{cohort}.md"


def _evaluation_result(seed: int, cohort: str) -> dict:
    path = RUN_ROOT / cohort / "evaluation" / f"seed-{seed}" / "result.json"
    if not path.is_file():
        raise RuntimeError(f"missing evaluation result for {cohort}/{seed}")
    result = load_json(path)
    if result["seed"] != seed or result["cohort"] != cohort:
        raise RuntimeError("evaluation identity differs")
    return result


def _training_summary(result: dict) -> dict:
    metadata = result["metadata"]
    return {
        "architecture": metadata["architecture"],
        "initial_backbone": metadata["initial_backbone"],
        "initial_local": metadata["initial_local"],
        "final_backbone": metadata["final_backbone"],
        "final_local": metadata["final_local"],
        "stage_boundary_backbone": metadata["stage_boundary_backbone"],
        "stream_fingerprint": metadata["stream_fingerprint"],
        "local_gain": metadata["local_gain"],
        "episode_exposures": metadata["episode_exposures"],
        "checkpoint_sha256": metadata["checkpoint"]["sha256"],
        "checkpoint_bytes": metadata["checkpoint"]["bytes"],
        "phase_stats": metadata["phase_stats"],
        "total_seconds": metadata["total_seconds"],
    }


def _cohort_outcome(cohort: str, rows: dict[str, dict]) -> str:
    decisions = [row["decisions"] for row in rows.values()]
    if all(row["admitted"] for row in decisions):
        return (
            "development_admitted"
            if cohort == "development"
            else "compact_model_confirmed"
        )
    if cohort == "confirmation":
        return "development_candidate_only"
    if not all(row["competence"]["passed"] for row in decisions):
        return "compact_model_rejected"
    if not all(row["mechanism"]["passed"] for row in decisions):
        return "competent_alternative_organization"
    return "behavior_incomplete"


def assemble_result(cohort: str) -> dict:
    specification = load_specification()
    artifact_lock = validate_artifact_lock(cohort)
    rows = {
        str(seed): _evaluation_result(seed, cohort)
        for seed in registered_seeds(specification, cohort)
    }
    outcome = _cohort_outcome(cohort, rows)
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
        "all_seed_rule": "No pooling, majority vote, filtering, or replacement.",
        "per_seed": {
            seed: {
                "training": _training_summary(row),
                "summaries": row["summaries"],
                "effects": row["effects"],
                "behavior": row["behavior"],
                "decisions": row["decisions"],
                "generic_stream_fingerprints": row["generic_stream_fingerprints"],
            }
            for seed, row in rows.items()
        },
        "retention": {
            "canonical": "This JSON and its Markdown report are the canonical cohort result.",
            "transient": "Checkpoints, full logs, sampled behavior, and dense arrays remain runtime-only and may be removed only under the registered cleanup gate.",
            "reconstruction": "Retrain from the locked source, protocol, repair, seed, RNG rule, and final-step recipe; verify the recorded stream and tensor hashes.",
        },
        "next_step": (
            "Run the unchanged reserved confirmation seeds 2804--2806."
            if outcome == "development_admitted"
            else "Stop without tuning and retain this outcome as evidence."
            if cohort == "development"
            else "Proceed to report-oriented runtime and historical-checkpoint retention cleanup."
            if outcome == "compact_model_confirmed"
            else "Stop without cleanup of historical P/L checkpoints."
        ),
    }


def _flag(value: bool) -> str:
    return "PASS" if value else "FAIL"


def report_text(result: dict) -> str:
    lines = [
        f"# Compact global/local model: {result['cohort']}",
        "",
        f"Registered outcome: **{result['outcome']}**.",
        "",
        "The candidate has 32 external channels, three support steps, two query steps, one scalar global margin, a 200-unit recurrent P state, and a 105-scalar exact packed local trace. The parent P/L evidence and historical checkpoints remain unchanged by this report.",
        "",
        "| Seed | Competence | P/L causal organization | Nine-row behavior preservation | Admitted | Local gain |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for seed, row in result["per_seed"].items():
        decision = row["decisions"]
        lines.append(
            f"| {seed} | {_flag(decision['competence']['passed'])} | {_flag(decision['mechanism']['passed'])} | {_flag(decision['behavior']['passed'])} | {_flag(decision['admitted'])} | {row['training']['local_gain']:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Boundaries",
            "",
            "Every decision is per network; no participant pooling across networks or seed majority vote is used. Human intervals and the Liu temperature are inherited descriptive references, not newly fitted confirmation targets. A causal gate failure denotes a different computational organization, not necessarily task incompetence.",
            "",
            f"Next step: {result['next_step']}",
            "",
            "The canonical JSON retains all endpoint estimates, bootstrap bounds, behavior flags, training costs, stream fingerprints, tensor hashes, and the exact artifact/source witnesses. Runtime checkpoints and dense arrays are deliberately not promoted into the report-oriented record.",
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
