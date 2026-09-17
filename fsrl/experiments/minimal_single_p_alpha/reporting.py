"""Canonical generic and final M2-alpha reports."""

from __future__ import annotations

import json
from collections import Counter

import numpy as np

from fsrl.experiments.training_strategy.evaluation import json_ready, write_arrays
from fsrl.experiments.write_cost.locks import completed
from fsrl.infra.provenance import load_json, write_json_exclusive

from .decisions import all_nine, evidence_binding, generic_category, outcome
from .locks import (
    reference,
    require_generic_freeze,
    validate_model_lock,
    verify_reference,
)
from .protocol import (
    GENERIC_REPORT,
    GENERIC_RESULT,
    MODEL_LOCK,
    PAIR_TABLE,
    PARAMETERS,
    PROTOCOL_SHA256,
    REPORT,
    RESULT,
    generic_directory,
    liu_directory,
    register,
    specification,
)


def report_generic() -> dict:
    _, lock = validate_model_lock()
    if GENERIC_RESULT.exists():
        return load_json(GENERIC_RESULT)
    networks = {}
    counts = Counter()
    for seed in specification()["design"]["network_seeds"]:
        panels = {
            str(panel): completed(generic_directory(seed, panel))
            for panel in specification()["design"]["generic_panels"]
        }
        category = generic_category(panels)
        counts[category] += 1
        networks[str(seed)] = {
            "category": category,
            "eta": lock["runs"][str(seed)]["metadata"]["final_eta"],
            "panels": panels,
        }
    payload = {
        "schema_version": 1,
        "protocol_sha256": PROTOCOL_SHA256,
        "model_lock": reference(MODEL_LOCK),
        "networks": networks,
        "category_counts": dict(sorted(counts.items())),
        "selection_performed": False,
        "liu_outcomes_exposed": False,
    }
    write_json_exclusive(GENERIC_RESULT, json_ready(payload))
    lines = [
        "# Generic qualification of paired M2-alpha",
        "",
        "All twenty final checkpoints were evaluated on all three frozen M2 generic panels before Liu evaluation.",
        "",
        "| category | networks |",
        "|---|---:|",
    ]
    for name in ("stable_constructive", "panel_variable", "nonconstructive"):
        lines.append(f"| {name} | {counts[name]}/20 |")
    lines += ["", "Every network proceeds to Liu without selection.", ""]
    GENERIC_REPORT.parent.mkdir(parents=True, exist_ok=True)
    GENERIC_REPORT.write_text("\n".join(lines), encoding="utf-8")
    register(
        finding=f"Generic M2-alpha freeze: {counts['stable_constructive']}/20 stable_constructive; Liu pending."
    )
    return payload


def _unit(seed, panel, condition, generic, result, parent) -> tuple[dict, list]:
    pairs = json.loads(
        (liu_directory(seed, panel, condition) / "pairs.json").read_text()
    )
    nine = all_nine(result)
    binding = evidence_binding(result)
    bad_pair = any(row["sampled_class"] in {0, 2} for row in pairs)
    morphology = result["morphology"]
    stable = generic["networks"][str(seed)]["category"] == "stable_constructive"
    constrained = (
        stable
        and binding
        and nine
        and morphology["latent_bimodal_pairs"] >= 15
        and morphology["sampled_bimodal_pairs"] >= 15
        and not bad_pair
    )
    inflated = morphology["sampled_bimodal_pairs"] > parent["counts"][
        "sampled_bimodal_pairs"
    ] and (not nine or bad_pair)
    row = {
        "seed": seed,
        "panel": panel,
        "condition": condition,
        "generic_category": generic["networks"][str(seed)]["category"],
        "all_nine_qualitative": nine,
        "evidence_binding": binding,
        "bad_sampled_pair": bad_pair,
        "constrained_morphology": constrained,
        "error_inflation": inflated,
        "stage": morphology["stage"],
        "M2_stage": parent["stage"],
        "counts": {
            key: morphology[key]
            for key in (
                "direction_present_pairs",
                "strong_two_sided_pairs",
                "latent_bimodal_pairs",
                "sampled_bimodal_pairs",
                "strong_error_subject_prevalence",
                "strong_error_top5_share",
            )
        },
        "M2_counts": parent["counts"],
        "by_distance": morphology["by_distance"],
    }
    return row, pairs


def report_final() -> dict:
    source, lock = validate_model_lock()
    generic = require_generic_freeze()
    if RESULT.exists():
        return load_json(RESULT)
    parent_result = load_json(
        verify_reference(source["parents"]["pair_morphology_result"])
    )
    parent = {
        (row["seed"], row["panel"], row["condition"]): row
        for row in parent_result["units"]
    }
    units = []
    columns = {
        name: []
        for name in (
            "seed",
            "panel",
            "condition",
            "pair_first",
            "pair_second",
            "symbolic_distance",
            "exact_mean",
            "fraction_wrong_direction",
            "fraction_strong_error",
            "fraction_strong_correct",
            "latent_class",
            "sampled_class",
        )
    }
    conditions = specification()["design"]["liu_conditions"]
    for seed in specification()["design"]["network_seeds"]:
        for panel in specification()["design"]["liu_panels"]:
            for condition_index, condition in enumerate(conditions):
                result = completed(liu_directory(seed, panel, condition))
                row, pairs = _unit(
                    seed,
                    panel,
                    condition,
                    generic,
                    result,
                    parent[(seed, panel, condition)],
                )
                units.append(row)
                for pair in pairs:
                    values = {
                        "seed": seed,
                        "panel": panel,
                        "condition": condition_index,
                        "pair_first": pair["pair"][0],
                        "pair_second": pair["pair"][1],
                        "symbolic_distance": pair["symbolic_distance"],
                        **{
                            key: pair[key]
                            for key in (
                                "exact_mean",
                                "fraction_wrong_direction",
                                "fraction_strong_error",
                                "fraction_strong_correct",
                                "latent_class",
                                "sampled_class",
                            )
                        },
                    }
                    for key, value in values.items():
                        columns[key].append(value)
    noisy = [row for row in units if row["condition"] == "noisy"]
    interpretable = sum(
        row["generic_category"] == "stable_constructive" and row["evidence_binding"]
        for row in noisy
    )
    constrained = sum(row["constrained_morphology"] for row in noisy)
    inflated = sum(row["error_inflation"] for row in noisy)
    registered = outcome(interpretable, constrained, inflated)
    payload = {
        "schema_version": 1,
        "protocol_sha256": PROTOCOL_SHA256,
        "model_lock": reference(MODEL_LOCK),
        "generic_result": reference(GENERIC_RESULT),
        "outcome": registered,
        "aggregate": {
            "generic_categories": generic["category_counts"],
            "interpretable_noisy_units": interpretable,
            "constrained_noisy_units": constrained,
            "error_inflation_units": inflated,
            "noisy_stage_counts": dict(
                sorted(Counter(row["stage"] for row in noisy).items())
            ),
            "all_nine_noisy_units": sum(row["all_nine_qualitative"] for row in noisy),
            "evidence_binding_noisy_units": sum(
                row["evidence_binding"] for row in noisy
            ),
        },
        "units": units,
        "mechanism_control_authorized": registered == "robust_constrained_rescue",
        "main_model_promoted": False,
        "selection_performed": False,
    }
    write_arrays(PAIR_TABLE, {key: np.asarray(value) for key, value in columns.items()})
    seeds = specification()["design"]["network_seeds"]
    write_arrays(
        PARAMETERS,
        {
            "seed": np.asarray(seeds),
            "eta": np.asarray(
                [lock["runs"][str(seed)]["metadata"]["final_eta"] for seed in seeds]
            ),
            "alpha_mean": np.asarray(
                [
                    lock["runs"][str(seed)]["metadata"]["final_model_summary"][
                        "alpha_mean"
                    ]
                    for seed in seeds
                ]
            ),
            "alpha_std": np.asarray(
                [
                    lock["runs"][str(seed)]["metadata"]["final_model_summary"][
                        "alpha_std"
                    ]
                    for seed in seeds
                ]
            ),
        },
    )
    write_json_exclusive(RESULT, json_ready(payload))
    lines = [
        "# Paired dense-alpha add-back to minimal single-P M2",
        "",
        f"Registered outcome: `{registered}`.",
        "",
        f"- Generic categories: {generic['category_counts']}",
        f"- Interpretable noisy units: {interpretable}/60",
        f"- All-nine noisy units: {payload['aggregate']['all_nine_noisy_units']}/60",
        f"- Constrained noisy units: {constrained}/60",
        f"- Error-inflation units: {inflated}/60",
        f"- Noisy stage counts: {payload['aggregate']['noisy_stage_counts']}",
        "",
        "Alpha remained a slow parameter and P the only episode-persistent plastic state. This paired complete-recipe result does not identify alpha and P separately or establish a biological mechanism.",
        "",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    status = {
        "robust_constrained_rescue": "supporting",
        "partial_or_mixed": "mixed",
        "generic_inadequate": "unresolved",
    }.get(registered, "valid_negative")
    register(status=status, finding=f"Frozen paired M2-alpha outcome: {registered}.")
    return payload


__all__ = ["report_final", "report_generic"]
