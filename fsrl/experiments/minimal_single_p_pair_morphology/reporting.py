"""Canonical result and concise interpretation for pair-morphology attribution."""

from __future__ import annotations

import shutil
from collections import defaultdict

import numpy as np

from fsrl.infra.provenance import file_sha256, load_json, write_json_exclusive

from .locks import reference, validate_source_input_lock
from .protocol import (
    PAIR_TABLE,
    REPORT,
    RESULT,
    RUNS,
    register,
    specification,
)


def _copy_exclusive(source, target) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as reader, target.open("xb") as writer:
        shutil.copyfileobj(reader, writer)
    if file_sha256(source) != file_sha256(target):
        raise RuntimeError("registered pair table differs from analysis output")


def _means(result: dict) -> dict:
    transitions = defaultdict(list)
    for row in result["transitions"].values():
        for condition, values in row.items():
            for name, value in values.items():
                transitions[(condition, name)].append(value)
    geometry = defaultdict(list)
    for row in result["units"]:
        for name, value in row["geometry"].items():
            geometry[(row["condition"], name)].append(value)
    return {
        "transitions": {
            condition: {
                name: float(np.mean(values))
                for (current, name), values in transitions.items()
                if current == condition
            }
            for condition in ("folded", "noisy")
        },
        "geometry": {
            condition: {
                name: float(np.mean(values))
                for (current, name), values in geometry.items()
                if current == condition
            }
            for condition in ("clean", "folded", "noisy")
        },
    }


def _conclusion(outcome: str) -> str:
    return {
        "direction_absent": (
            "The dominant failure precedes margin strength: too few pairs form both "
            "correct- and incorrect-direction latent groups."
        ),
        "weak_margin": (
            "The dominant failure is weak negative margin: directional disagreement is "
            "present across enough pairs, but too few pairs contain both stable-error and "
            "stable-correct latent probabilities."
        ),
        "latent_shape": (
            "Strong two-sided latent cases exist across enough pairs, but their across-subject "
            "probability distributions do not acquire the frozen bimodal shape."
        ),
        "finite_sampling": (
            "The latent probability distributions meet the frozen bimodality target; the "
            "remaining deficit arises in the fixed ten-choice realization/classification."
        ),
        "already_latent_and_sampled": (
            "The archived sampled and latent distributions both meet the pair-count target."
        ),
        "mixed_or_unidentified": (
            "No single registered stage explains at least two thirds of the frozen noisy "
            "network-panel units; the cohort-level attribution remains mixed."
        ),
    }[outcome]


def _render(result: dict) -> str:
    aggregate = result["aggregate"]
    outcome = aggregate["outcome"]
    means = result["descriptive_means"]
    lines = [
        "# Pair-morphology attribution in frozen minimal single-P M2",
        "",
        f"Registered primary attribution: `{outcome}`.",
        "",
        _conclusion(outcome),
        "",
        "## Noisy-condition stage distribution",
        "",
        "| stage | network-panel units |",
        "|---|---:|",
    ]
    for name, count in aggregate["noisy_stage_counts"].items():
        lines.append(f"| {name} | {count}/60 |")
    lines += [
        "",
        "| pair-level stage count | mean | median | range |",
        "|---|---:|---:|---:|",
    ]
    for name, row in aggregate["noisy_pair_stage_counts"].items():
        lines.append(
            f"| {name} | {row['mean']:.3f} | {row['median']:.3f} | "
            f"{row['minimum']}--{row['maximum']} |"
        )
    lines += [
        "",
        "## Condition-paired localization",
        "",
        "| condition minus clean | direction flips | stable-category crossings | gradient delta RMS | residual delta RMS | mean absolute evidence delta |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for condition in ("folded", "noisy"):
        row = means["transitions"][condition]
        lines.append(
            f"| {condition} | {row['direction_flip_fraction']:.5f} | "
            f"{row['stable_category_crossing_fraction']:.5f} | "
            f"{row['gradient_delta_rms']:.5f} | {row['residual_delta_rms']:.5f} | "
            f"{row['mean_absolute_evidence_delta']:.5f} |"
        )
    lines += [
        "",
        (
            "The Hodge rows localize changes in the archived query field. They are not a "
            "causal mediation analysis of P writes or a claim that residual energy is "
            "biological noise."
        ),
        "",
        "## Score-only comparison boundary",
        "",
        f"Interface outcome: `{result['score_interface']['outcome']}`.",
        "",
    ]
    mismatches = []
    for panel, checks in result["score_interface"]["panels"].items():
        mismatches.extend(
            f"panel {panel}: {name}" for name, passed in checks.items() if not passed
        )
    if not result["score_interface"]["semantic_checks"]["admission_semantics_equal"]:
        mismatches.append("admission semantics")
    lines.extend(f"- {item}" for item in mismatches)
    lines += [
        "",
        (
            "Because the interface gate did not pass, no pair-geometry difference is "
            "attributed specifically to recurrent P versus the score update. The existing "
            "score-only positive result remains a complete-recipe reference."
            if result["score_interface"]["outcome"] == "complete_recipe_only"
            else "The exact interface gate passed, so the archived geometry comparison is qualified."
        ),
        "",
        "## Integrity and stop boundary",
        "",
        (
            "All 180 frozen M2 units were retained. Both full and global sampled behaviors "
            "were deterministically replayed, and archived fields, probabilities, potentials, "
            "and Hodge identities passed the registered tolerances."
        ),
        "",
        (
            "No training, model inference, rescaling, extra Monte Carlo choices, parameter "
            "selection, or checkpoint mutation was performed. This study stops here and does "
            "not promote a main model or authorize a successor rule."
        ),
        "",
    ]
    return "\n".join(lines)


def write_report() -> dict:
    validate_source_input_lock()
    runtime_result = RUNS / "analysis/result.json"
    runtime_table = RUNS / "analysis/pair_table.npz"
    result = load_json(runtime_result)
    if len(result["units"]) != specification()["design"]["M2_units"]:
        raise RuntimeError("analysis did not retain all mandatory M2 units")
    _copy_exclusive(runtime_table, PAIR_TABLE)
    result["pair_table"] = reference(PAIR_TABLE)
    result["descriptive_means"] = _means(result)
    write_json_exclusive(RESULT, result)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with REPORT.open("x", encoding="utf-8") as handle:
        handle.write(_render(result))
    outcome = result["aggregate"]["outcome"]
    status = "unresolved" if outcome == "mixed_or_unidentified" else "supporting"
    register(
        status=status,
        finding=(
            f"Read-only attribution outcome {outcome}: noisy stage counts were "
            f"{result['aggregate']['noisy_stage_counts']}. The score-only interface outcome "
            f"was {result['score_interface']['outcome']}; no training or model execution was performed."
        ),
    )
    return {
        "outcome": outcome,
        "score_interface": result["score_interface"]["outcome"],
        "result": reference(RESULT),
        "report": reference(REPORT),
    }


__all__ = ["write_report"]
