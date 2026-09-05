"""Immutable complete results beside the frozen recurrent-network evidence."""

import shutil
from typing import cast

from fsrl.experiments.training_strategy.evaluation import json_ready
from fsrl.infra.provenance import file_sha256, load_json, write_json_exclusive
from fsrl.paths import REPO_ROOT

from .decisions import adequate, competence, pair_analysis, study_outcome
from .evaluation import validate_evaluation
from .locks import ARTIFACT_LOCK, reference, validate_artifacts
from .protocol import PROTOCOL_SHA256, RECORD_ROOT, specification

RESULT_PATH = RECORD_ROOT / "results" / "minimal_relational_learner_v1.json"
REPORT_PATH = RECORD_ROOT / "reports" / "minimal_relational_learner_v1.md"


def historical_comparison() -> dict:
    path = (
        REPO_ROOT
        / "studies/joint_training_strategy/records/results/joint_training_strategy_v1.json"
    )
    original = load_json(path)
    return {
        "source": reference(path),
        "outcome": original["outcome"],
        "comparison_scope": "Frozen descriptive references with different training seeds; not paired network effects or simultaneous runtime benchmarks.",
        "models": {
            f"{seed}/{condition}": {
                "behavior": row["behavior"],
                "cost": row["cost"],
                "probability": row["summaries"]["liu"]["intact"]["probability"],
            }
            for seed, pair in original["seeds"].items()
            for condition, row in pair["conditions"].items()
        },
    }


def assemble_result(conditions: dict, lock: dict, spec: dict) -> dict:
    pairs = {}
    for seed in spec["seeds"]["mandatory"]:
        trace, score = (
            conditions[f"{seed}/{condition}"]
            for condition in ("score_trace", "score_only")
        )
        pairs[str(seed)] = pair_analysis(trace, score, seed, spec)
    return {
        "experiment_id": spec["experiment_id"],
        "protocol_sha256": PROTOCOL_SHA256,
        "source_commit": lock["source_commit"],
        "artifact_lock": reference(ARTIFACT_LOCK),
        "conditions": conditions,
        "pairs": pairs,
        "outcome": study_outcome(pairs, spec),
        "historical": historical_comparison(),
        "claim_boundary": spec["claim_boundary"],
        "stop_rule": spec["decision_contract"]["stop_rule"],
    }


def flag(value) -> str:
    return "PASS" if value else "FAIL"


def number(value) -> str:
    return "undefined" if value is None else f"{value:.5f}"


def estimate_row(name, row) -> str:
    return f"| {name} | {number(row['mean'])} | {number(row['bootstrap']['lower'])} | {number(row['bootstrap']['upper'])} | {row['subjects']} |"


def _estimate_table(title, rows) -> list[str]:
    return [
        "",
        title,
        "",
        "| Endpoint | Mean | 95% lower | 95% upper | N |",
        "| --- | --- | --- | --- | --- |",
        *(estimate_row(name, row) for name, row in rows),
    ]


def _pair_section(seed, pair) -> list[str]:
    lines = ["", f"## Training stream {seed}: `{pair['outcome']}`"]
    rows = [
        (f"{domain}/{name}", row)
        for domain, groups in pair["paired_probability"].items()
        for name, row in groups.items()
    ]
    lines.extend(
        _estimate_table(
            "Score-trace minus independently fitted score-only correct probability:",
            rows,
        )
    )
    lines.extend(
        _estimate_table(
            "Acute local use (separate from independent fitting):",
            pair["trace_acute_effects"].items(),
        )
    )
    lines.extend(
        [
            "",
            "| Local-support requirement | Lower bound | Threshold | Pass |",
            "| --- | --- | --- | --- |",
        ]
    )
    for name, check in pair["local_support"].items():
        sign = ">" if check["strict"] else ">="
        lines.append(
            f"| {name} | {number(check['lower'])} | {sign} {check['threshold']} | {flag(check['passed'])} |"
        )
    return lines


def _condition_section(identity, row) -> list[str]:
    lines = [
        "",
        f"## {identity}: complete behavior map",
        "",
        "| Row | Qualitative | Frozen quantitative classifier |",
        "| --- | --- | --- |",
    ]
    for name, flags in row["behavior"]["flags"].items():
        lines.append(
            f"| {name} | {flag(flags['qualitative'])} | {flag(flags['calibration'])} |"
        )
    lines.extend(
        [
            "",
            f"Constrained parameters: `{row['parameters']}`.",
            f"Inference seconds per episode (warm compiled batch): `{row['inference_seconds_per_episode']}`.",
            f"Float64 analytic bridge versus float32 rollout maximum absolute error: `{row['history']['float64_to_float32_max_abs_error']}`.",
        ]
    )
    lines.extend(
        _estimate_table("History effect magnitudes:", row["history"]["summary"].items())
    )
    return lines


def report_text(result: dict, spec: dict) -> str:
    lines = [
        "# Minimal metric-error relational learner",
        "",
        f"Registered outcome: **`{result['outcome']}`**.",
        "",
        "Independent fixed-recipe test: normalized online score learning with an optional unchanged conjunctive local trace. This does not repair the old joint-training outcome or reconstruct P trajectories.",
        "Three paired training streams (2111/2112/2113), identical scalar initialization, 48,000 episodes per model. All six artifacts locked before evaluation. Participant bootstrap is separate within each fit, never pooled across training streams.",
        "",
        "## Per-run behavior and competence",
        "",
        "| Stream/recipe | Competent | Nine-row adequacy | Learned scalars | Persistent state entries | Warm training seconds |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for identity, row in result["conditions"].items():
        cost = row["cost"]
        state = cost["global_persistent_entries"] + cost["local_persistent_entries"]
        lines.append(
            f"| {identity} | {flag(competence(row, spec)['passed'])} | {flag(adequate(row, spec))} | {cost['trainable_parameters']} | {state} | {number(cost['warm_training_seconds'])} |"
        )
    for seed, pair in result["pairs"].items():
        lines.extend(_pair_section(seed, pair))
    for identity, row in result["conditions"].items():
        lines.extend(_condition_section(identity, row))
    lines.extend(
        [
            "",
            "## Interpretation boundaries",
            "",
            "Global additivity is imposed, not an emergent Hodge result. The exact derivative decomposes fixed-encoding influence into direct cue overlap and future-update context; a nonzero history component is not by itself correct semantic inference. Raw signed sensitivities and their exact margin bridge are preserved in typed NPZ.",
            "The score-only baseline can win: it is not required to preserve an unnecessary local branch. Conversely, an acute L-off effect is not evidence that independently optimized score-only cannot solve the task. Missing behavior under a fixed recipe does not establish universal architectural impossibility.",
            "All nine qualitative rows determine adequacy; all nine quantitative classifications are descriptive historical interval membership, not model-human equivalence. The historical temperature 0.25 and encoding distribution are inherited assumptions, not newly fitted parameters.",
            "The JSON retains complete per-participant endpoints, denominators/exclusions, headroom, geometry, all local controls, paired uncertainty and historical RNN behavior/costs. Headroom is reported without claiming it explains an observed effect. Different training seeds and historical timing runs prevent paired RNN-effect or direct speed-benchmark claims.",
            "",
            "## Frozen next step",
            "",
            result["stop_rule"],
            "",
        ]
    )
    return "\n".join(lines)


def write_report() -> dict:
    lock = validate_artifacts()
    spec = specification()
    conditions = {
        f"{seed}/{condition}": validate_evaluation(seed, condition, lock)
        for seed in spec["seeds"]["mandatory"]
        for condition in spec["seeds"]["conditions"]
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    for identity, row in conditions.items():
        seed, name = identity.split("/")
        source = REPO_ROOT / row["raw_arrays"]["path"]
        target = (
            RESULT_PATH.parent / f"minimal_relational_learner_v1.seed-{seed}.{name}.npz"
        )
        with source.open("rb") as reader, target.open("xb") as writer:
            shutil.copyfileobj(reader, writer)
        if file_sha256(source) != file_sha256(target):
            raise RuntimeError("registered arrays differ from evaluated arrays")
        row["registered_raw_arrays"] = reference(target)
    result = cast(dict, json_ready(assemble_result(conditions, lock, spec)))
    write_json_exclusive(RESULT_PATH, result)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("x") as handle:
        handle.write(report_text(result, spec))
    return {
        "outcome": result["outcome"],
        "result": reference(RESULT_PATH),
        "report": reference(REPORT_PATH),
    }
