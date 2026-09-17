"""Canonical generic and final summaries for the aligned comparator."""

from __future__ import annotations

import json
from collections import Counter

import numpy as np

from fsrl.experiments.training_strategy.evaluation import json_ready, write_arrays
from fsrl.experiments.write_cost.locks import completed
from fsrl.infra.provenance import load_json, write_json_exclusive

from .decisions import generic_category
from .locks import reference, validate_model_lock, verify_reference
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
    validate_model_lock()
    if GENERIC_RESULT.exists() or GENERIC_REPORT.exists():
        return load_json(GENERIC_RESULT)
    streams = {}
    categories = Counter()
    for seed in specification()["design"]["training_streams"]:
        panels = {
            str(panel): completed(generic_directory(seed, panel))
            for panel in specification()["design"]["generic_panels"]
        }
        category = generic_category(sum(row["passed"] for row in panels.values()))
        categories[category] += 1
        streams[str(seed)] = {"category": category, "panels": panels}
    payload = {
        "schema_version": 1,
        "protocol_sha256": PROTOCOL_SHA256,
        "model_lock": reference(MODEL_LOCK),
        "streams": streams,
        "category_counts": dict(sorted(categories.items())),
        "selection_performed": False,
        "liu_outcomes_exposed": False,
    }
    write_json_exclusive(GENERIC_RESULT, json_ready(payload))
    lines = [
        "# Generic qualification of the information-aligned q-only comparator",
        "",
        "All twenty final two-scalar fits were evaluated on all three frozen parent generic panels before any Liu evaluation.",
        "",
        "| category | streams |",
        "|---|---:|",
    ]
    for name in ("stable_competent", "panel_variable", "noncompetent"):
        lines.append(f"| {name} | {categories[name]}/20 |")
    lines += [
        "",
        "Every stream proceeds to Liu without selection. Additive coherence is an integrity property of the score-difference representation, not an emergent result.",
    ]
    GENERIC_REPORT.parent.mkdir(parents=True, exist_ok=True)
    GENERIC_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    register(
        finding=(
            f"Generic evaluation frozen: {categories['stable_competent']}/20 stable_competent, "
            f"{categories['panel_variable']}/20 panel_variable, and "
            f"{categories['noncompetent']}/20 noncompetent; Liu evaluation pending."
        )
    )
    return payload


def _all_qualitative(result: dict) -> bool:
    flags = result["liu"]["routes"]["full"]["behavior"]["historical_nine_rows"]["flags"]
    return all(row["qualitative"] for row in flags.values())


def _binding(result: dict) -> bool:
    lower = result["liu"]["effects"]["intact_minus_evidence_shuffle_learned"][
        "interval"
    ]["lower"]
    return lower is not None and lower > 0.0


def report_final() -> dict:
    source, lock = validate_model_lock()
    generic = load_json(GENERIC_RESULT)
    if RESULT.exists():
        return load_json(RESULT)
    m2 = load_json(verify_reference(source["parents"]["pair_morphology_result"]))
    m2_units = {
        (row["seed"], row["panel"], row["condition"]): row for row in m2["units"]
    }
    units = []
    pair_columns: dict[str, list] = {
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
    for seed in specification()["design"]["training_streams"]:
        for panel in specification()["design"]["liu_panels"]:
            for condition_index, condition in enumerate(
                specification()["design"]["liu_conditions"]
            ):
                result = completed(liu_directory(seed, panel, condition))
                pairs = json.loads(
                    (liu_directory(seed, panel, condition) / "pairs.json").read_text()
                )
                all_nine = _all_qualitative(result)
                binding = _binding(result)
                bad_pair = any(row["sampled_class"] in {0, 2} for row in pairs)
                morphology = result["morphology"]
                constrained = (
                    all_nine
                    and binding
                    and morphology["latent_bimodal_pairs"] >= 15
                    and morphology["sampled_bimodal_pairs"] >= 15
                    and not bad_pair
                )
                m2_row = m2_units[(seed, panel, condition)]
                units.append(
                    {
                        "seed": seed,
                        "panel": panel,
                        "condition": condition,
                        "generic_category": generic["streams"][str(seed)]["category"],
                        "all_nine_qualitative": all_nine,
                        "evidence_binding": binding,
                        "constrained_morphology": constrained,
                        "bad_sampled_pair": bad_pair,
                        "stage": morphology["stage"],
                        "M2_stage": m2_row["stage"],
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
                        "by_distance": morphology["by_distance"],
                        "analytic": result["analytic"],
                    }
                )
                for row in pairs:
                    values = {
                        "seed": seed,
                        "panel": panel,
                        "condition": condition_index,
                        "pair_first": row["pair"][0],
                        "pair_second": row["pair"][1],
                        "symbolic_distance": row["symbolic_distance"],
                        **{
                            key: row[key]
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
                        pair_columns[key].append(value)
    noisy = [row for row in units if row["condition"] == "noisy"]
    competent_streams = {
        int(seed)
        for seed, row in generic["streams"].items()
        if row["category"] == "stable_competent"
    }
    binding_streams = {
        seed
        for seed in competent_streams
        if all(
            next(row for row in noisy if row["seed"] == seed and row["panel"] == panel)[
                "evidence_binding"
            ]
            for panel in specification()["design"]["liu_panels"]
        )
    }
    interpretable = [row for row in noisy if row["seed"] in binding_streams]
    broader_streams = [
        seed
        for seed in binding_streams
        if all(row["constrained_morphology"] for row in noisy if row["seed"] == seed)
    ]
    inflated = sum(
        row["counts"]["sampled_bimodal_pairs"]
        > m2_units[(row["seed"], row["panel"], "noisy")]["counts"][
            "sampled_bimodal_pairs"
        ]
        and (not row["all_nine_qualitative"] or row["bad_sampled_pair"])
        for row in noisy
    )
    aligned = sum(row["stage"] == row["M2_stage"] for row in interpretable)
    if broader_streams:
        outcome = "broader_constrained_morphology"
    elif inflated >= 40:
        outcome = "error_inflation"
    elif len(interpretable) >= 40 and aligned >= 40:
        outcome = "aligned_distance_restriction"
    elif len(interpretable) < 40:
        outcome = "comparator_inadequate"
    else:
        outcome = "mixed_or_unidentified"
    stage_counts = Counter(row["stage"] for row in noisy)
    payload = {
        "schema_version": 1,
        "protocol_sha256": PROTOCOL_SHA256,
        "model_lock": reference(MODEL_LOCK),
        "generic_result": reference(GENERIC_RESULT),
        "outcome": outcome,
        "main_model_promoted": False,
        "M3_authorized": False,
        "selection_performed": False,
        "aggregate": {
            "generic_categories": generic["category_counts"],
            "noisy_stage_counts": dict(sorted(stage_counts.items())),
            "interpretable_noisy_units": len(interpretable),
            "aligned_stage_units": aligned,
            "error_inflation_units": inflated,
            "broader_constrained_streams": broader_streams,
            "noisy_constrained_units": sum(
                row["constrained_morphology"] for row in noisy
            ),
        },
        "units": units,
    }
    write_arrays(
        PAIR_TABLE, {key: np.asarray(value) for key, value in pair_columns.items()}
    )
    seeds = specification()["design"]["training_streams"]
    write_arrays(
        PARAMETERS,
        {
            "seed": np.asarray(seeds),
            "eta": np.asarray(
                [
                    lock["runs"][str(seed)]["metadata"]["physical_parameters"]["eta"]
                    for seed in seeds
                ]
            ),
            "gamma": np.asarray(
                [
                    lock["runs"][str(seed)]["metadata"]["physical_parameters"]["gamma"]
                    for seed in seeds
                ]
            ),
        },
    )
    write_json_exclusive(RESULT, json_ready(payload))
    lines = [
        "# Information-aligned q-only score comparator",
        "",
        f"Registered outcome: `{outcome}`.",
        "",
        "## Frozen cohort",
        "",
        f"- Generic categories: {generic['category_counts']}",
        f"- Interpretable noisy units: {len(interpretable)}/60",
        f"- Noisy stage counts: {dict(sorted(stage_counts.items()))}",
        f"- Noisy constrained units: {sum(row['constrained_morphology'] for row in noisy)}/60",
        f"- Error-inflation units: {inflated}/60",
        f"- Streams constrained in all three noisy panels: {broader_streams}",
        "",
        "## Boundary",
        "",
        "The comparator used the same hash-verified generic training streams and the exact frozen generic/Liu evaluation arrays available to M2, but remains a complete-recipe comparison. Its additive field and unit Hodge coherence are structural. The result neither promotes a main model nor authorizes M3.",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    register(
        status="mixed" if outcome == "mixed_or_unidentified" else "completed",
        finding=f"Frozen aligned q-only comparator outcome: {outcome}.",
    )
    return payload


__all__ = ["report_final", "report_generic"]
