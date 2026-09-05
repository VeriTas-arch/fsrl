"""Write-once paired results and a complete per-seed evidence report."""

from __future__ import annotations

import shutil

import numpy as np

from fsrl.infra.provenance import file_sha256, write_json_exclusive
from fsrl.paths import REPO_ROOT

from . import decisions
from .evaluation import evaluation_directory, json_ready, validate_evaluation
from .locks import ARTIFACT_LOCK_PATH, RECORD_ROOT, reference, validate_artifact_lock
from .protocol import PROTOCOL_SHA256, load_specification
from .summaries import paired_endpoints

RESULT_PATH = RECORD_ROOT / "results" / "joint_training_strategy_v1.json"
REPORT_PATH = RECORD_ROOT / "reports" / "joint_training_strategy_v1.md"


def verify_matched_evaluation(staged: dict, joint: dict) -> None:
    if staged["generic_stream_fingerprints"] != joint["generic_stream_fingerprints"]:
        raise RuntimeError("paired generic evaluation streams differ")
    keys = (
        "retention",
        "probabilities",
        "cue_codes",
        "support_pairs",
        "observed_signed_evidence",
        "natural_local_evidence",
        "shuffled_local_evidence",
        "evidence_routing",
        "query_routing",
    )
    with (
        np.load(REPO_ROOT / staged["raw_arrays"]["path"], allow_pickle=False) as first,
        np.load(REPO_ROOT / joint["raw_arrays"]["path"], allow_pickle=False) as second,
    ):
        for key in keys:
            np.testing.assert_array_equal(first[f"liu__{key}"], second[f"liu__{key}"])


def assemble_result(conditions: dict, artifact_lock: dict, specification: dict) -> dict:
    seeds = {}
    for seed in specification["seeds"]["mandatory"]:
        staged, joint = (
            conditions[f"{seed}/{name}"] for name in ("matched_staged", "joint")
        )
        verify_matched_evaluation(staged, joint)
        paired = paired_endpoints(
            joint["raw_endpoints"], staged["raw_endpoints"], seed, specification
        )
        noninferiority = decisions.noninferiority(paired, specification)
        fitted = {
            name: conditions[f"{seed}/{name}"]
            for name in specification["seeds"]["conditions"]
        }
        seeds[str(seed)] = {
            "conditions": fitted,
            "paired_noninferiority": {"estimates": paired, "decision": noninferiority},
            "cost": decisions.cost_comparison(staged["cost"], joint["cost"]),
            "outcome": decisions.outcome(
                {name: row["decisions"] for name, row in fitted.items()}, noninferiority
            ),
        }
    return {
        "experiment_id": specification["experiment_id"],
        "protocol_sha256": PROTOCOL_SHA256,
        "source_commit": artifact_lock["source_commit"],
        "artifact_lock": reference(ARTIFACT_LOCK_PATH),
        "outcome": decisions.study_outcome(seeds, specification),
        "seeds": seeds,
        "measured_efficiency_advantage": all(
            row["cost"]["efficiency_advantage"] for row in seeds.values()
        ),
        "claim_boundary": specification["claim_boundary"],
        "stop_rule": specification["decision_contract"]["stop_rule"],
    }


def _flag(value: bool) -> str:
    return "PASS" if value else "FAIL"


def _number(value) -> str:
    return "undefined" if value is None else f"{value:.5f}"


def report_text(result: dict) -> str:
    lines = [
        "# Matched staged versus single-stage joint training",
        "",
        f"Registered outcome: **`{result['outcome']}`**.",
        "",
        "## Question and frozen comparison",
        "",
        "Can one final query objective replace sequential global/local fitting at the same generic episode budget, while preserving competence, causal organization, and the frozen Liu behavior map?",
        "Both conditions use the same imposed P/L structure, final evidence admission, paired initialization and task stream. Each sees 48,000 training episodes. Staged fitting updates the backbone 1,000 times and gain 500 times; joint fitting updates both 1,500 times. This is not an order-only or equal-FLOPs experiment.",
        "",
        f"Protocol SHA-256: `{result['protocol_sha256']}`. Implementation witness: `{result['source_commit']}`.",
        "All six final artifacts were jointly locked and pushed before any new evaluation. Bootstrap is within each network (10,000 draws), with no participant pooling or network-population bootstrap. Probability endpoints average the two orientation-specific sigmoids; decision ties count as one half.",
        "",
        "## Per-seed primary results",
        "",
        "| Seed | Staged competence | Joint competence | Paired NI | Staged mechanism | Joint mechanism | Joint behavior | Outcome |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for seed, row in result["seeds"].items():
        staged, joint = (
            row["conditions"][name]["decisions"] for name in ("matched_staged", "joint")
        )
        flags = [
            staged["competence"]["passed"],
            joint["competence"]["passed"],
            row["paired_noninferiority"]["decision"]["passed"],
            staged["mechanism"]["passed"],
            joint["mechanism"]["passed"],
            joint["behavior"]["passed"],
        ]
        lines.append(
            f"| {seed} | {' | '.join(_flag(value) for value in flags)} | `{row['outcome']}` |"
        )
    for seed, row in result["seeds"].items():
        lines.extend(_seed_sections(seed, row))
    lines.extend(
        [
            "## Supported, rejected, and unidentified claims",
            "",
            "The tables retain every registered link independently. A passed competence or noninferiority comparison does not by itself establish preserved mechanism; preserved mechanism does not substitute for the nine-row behavior map. No successful network repairs a failed mandatory network.",
            "Single-stage preservation, if all registered gates pass, establishes recipe-level feasibility under imposed structural priors and can motivate independent confirmation. It does not establish minimal architecture, autonomous emergence of two memories, human neural implementation, or a population-level network effect.",
            "If a stronger conjunction fails, the passing links remain positive evidence at their registered scope. The outcome labels distinguish recipe insufficiency, comparator insufficiency, noninferiority failure, an alternative computation, and incomplete behavior preservation.",
            "",
            "## Costs, calibration, and next step",
            "",
            f"Measured joint efficiency advantage in every pair: **{_flag(result['measured_efficiency_advantage'])}**. Stage count alone is not a compute claim. Compilation/warmup, measured training, peak allocated/reserved memory, and parameter/update counts are retained in the numerical records.",
            "Liu temperature 0.25 is inherited historical human-informed calibration; it was not refitted. Human interval membership remains descriptive, not model-human equivalence. Old distance, serial-position, and self-inconsistency mismatches were not required to persist.",
            result["stop_rule"],
            "",
            "The JSON result retains exact estimates, denominators, bootstrap bounds, human-map metrics, secondary qualification/projection diagnostics, and provenance. Registered typed NPZ files preserve raw oriented margins, controls, LOO arrays, evidence/routing, and generic inputs; sampled behavior is available through its verified runtime manifest.",
            "",
        ]
    )
    return "\n".join(lines)


def _seed_sections(seed: str, row: dict) -> list[str]:
    lines = [
        "",
        f"## Seed {seed}",
        "",
        "### Paired correct-probability noninferiority",
        "",
        "| Endpoint | Joint minus staged | 95% lower | 95% upper | N | Pass (LB ≥ -0.02) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for domain, groups in row["paired_noninferiority"]["estimates"].items():
        for group, estimate in groups.items():
            passed = row["paired_noninferiority"]["decision"]["checks"][
                f"{domain}_{group}"
            ]["passed"]
            lines.append(
                f"| {domain}/{group} | {_number(estimate['mean'])} | {_number(estimate['bootstrap']['lower'])} | {_number(estimate['bootstrap']['upper'])} | {estimate['subjects']} | {_flag(passed)} |"
            )
    for name, condition in row["conditions"].items():
        lines.extend(
            [
                "",
                f"### {name}: causal links",
                "",
                "| Link / endpoint | Bound value | Registered criterion | Pass |",
                "| --- | --- | --- | --- |",
            ]
        )
        for link, decision in condition["decisions"]["mechanism"]["links"].items():
            for endpoint, check in decision["checks"].items():
                lines.append(
                    f"| {link}/{endpoint} | {_number(check['value'])} | {check['statistic']} {check['operator']} {check['threshold']} | {_flag(check['passed'])} |"
                )
        lines.extend(
            [
                "",
                f"### {name}: all nine behavior rows",
                "",
                "| Row | Qualitative | Frozen quantitative classifier |",
                "| --- | --- | --- |",
            ]
        )
        for metric, flags in condition["behavior"]["flags"].items():
            lines.append(
                f"| {metric} | {_flag(flags['qualitative'])} | {_flag(flags['calibration'])} |"
            )
        lines.append(
            f"\nOwn-global legacy qualification (secondary): {_flag(condition['legacy_qualification']['qualification']['passed'])}."
        )
    lines.extend(
        ["", f"Joint/staged cost ratios: `{row['cost']['joint_over_staged']}`."]
    )
    return lines


def write_report() -> dict:
    lock = validate_artifact_lock()
    specification = load_specification()
    conditions = {
        f"{seed}/{condition}": validate_evaluation(seed, condition, lock)
        for seed in specification["seeds"]["mandatory"]
        for condition in specification["seeds"]["conditions"]
    }
    result = assemble_result(conditions, lock, specification)
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    for identity, condition in conditions.items():
        seed, name = identity.split("/")
        source = evaluation_directory(int(seed), name) / "raw.npz"
        target = (
            RESULT_PATH.parent / f"joint_training_strategy_v1.seed-{seed}.{name}.npz"
        )
        with source.open("rb") as reader, target.open("xb") as writer:
            shutil.copyfileobj(reader, writer)
        if file_sha256(source) != file_sha256(target):
            raise RuntimeError("registered scientific arrays changed during promotion")
        condition["registered_raw_arrays"] = reference(target)
    write_json_exclusive(RESULT_PATH, json_ready(result))
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("x", encoding="utf-8") as handle:
        handle.write(report_text(result))
    return {
        "outcome": result["outcome"],
        "result": reference(RESULT_PATH),
        "report": reference(REPORT_PATH),
    }
