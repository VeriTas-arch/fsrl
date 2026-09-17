"""Canonical generic and final vector-modulation reports."""

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
    PROTOCOL_SHA256,
    REPORT,
    RESULT,
    WRITE_TABLE,
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
            "paired_preserved": all(row["paired_preserved"] for row in panels.values()),
            "vector_used": any(
                row["write_geometry"]["vector_used"] for row in panels.values()
            ),
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
        "# Generic qualification of M2 vector modulation",
        "",
        "All three final checkpoints were evaluated on all three frozen M2 generic panels before Liu evaluation.",
        "",
        "| category | networks |",
        "|---|---:|",
    ]
    for name in ("stable_constructive", "panel_variable", "nonconstructive"):
        lines.append(f"| {name} | {counts[name]}/3 |")
    lines += ["", "Every network proceeds to Liu without selection.", ""]
    GENERIC_REPORT.parent.mkdir(parents=True, exist_ok=True)
    GENERIC_REPORT.write_text("\n".join(lines), encoding="utf-8")
    register(
        finding=(
            "Generic vector-modulation development freeze: "
            f"{counts['stable_constructive']}/3 stable_constructive; Liu pending."
        )
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
    damage = bad_pair or not result["paired_exact_preserved"]
    row = {
        "seed": seed,
        "panel": panel,
        "condition": condition,
        "generic_category": generic["networks"][str(seed)]["category"],
        "all_nine_qualitative": nine,
        "evidence_binding": binding,
        "bad_sampled_pair": bad_pair,
        "constrained_morphology": constrained,
        "damage_guard_failed": damage,
        "paired_M2_exact_delta": result["paired_M2_exact_delta"],
        "vector_used": result["write_geometry"]["vector_used"],
        "write_geometry": result["write_geometry"],
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
    source, _ = validate_model_lock()
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
    pair_columns = {
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
    write_columns = {
        name: []
        for name in (
            "seed",
            "panel",
            "condition",
            "active_fraction",
            "R_mean",
            "R_maximum",
            "U_norm_mean",
            "clipping_residual_ratio_mean",
            "clamp_fraction_mean",
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
                geometry = result["write_geometry"]
                write_values = {
                    "seed": seed,
                    "panel": panel,
                    "condition": condition_index,
                    "active_fraction": geometry["active_fraction"],
                    "R_mean": geometry["R"]["mean"],
                    "R_maximum": geometry["R"]["maximum"],
                    "U_norm_mean": geometry["U_norm"]["mean"],
                    "clipping_residual_ratio_mean": geometry["clipping_residual_ratio"][
                        "mean"
                    ],
                    "clamp_fraction_mean": geometry["clamp_fraction"]["mean"],
                }
                for key, value in write_values.items():
                    write_columns[key].append(value)
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
                        pair_columns[key].append(value)
    noisy = [row for row in units if row["condition"] == "noisy"]
    network_rows = {}
    for seed in specification()["design"]["network_seeds"]:
        rows = [row for row in noisy if row["seed"] == seed]
        stable = generic["networks"][str(seed)]["category"] == "stable_constructive"
        vector_used = generic["networks"][str(seed)]["vector_used"] or any(
            row["vector_used"] for row in rows
        )
        has_constrained = any(row["constrained_morphology"] for row in rows)
        preserved = generic["networks"][str(seed)]["paired_preserved"] and not any(
            row["damage_guard_failed"] for row in rows
        )
        network_rows[str(seed)] = {
            "stable_constructive": stable,
            "vector_used": vector_used,
            "has_constrained_noisy_panel": has_constrained,
            "preserved": preserved,
            "rescued": stable and vector_used and has_constrained and preserved,
            "damaging_rescue": stable
            and vector_used
            and has_constrained
            and not preserved,
        }
    stable = sum(row["stable_constructive"] for row in network_rows.values())
    rescued = sum(row["rescued"] for row in network_rows.values())
    damaging = sum(row["damaging_rescue"] for row in network_rows.values())
    vector_used = sum(row["vector_used"] for row in network_rows.values())
    registered = outcome(stable, rescued, damaging, vector_used)
    payload = {
        "schema_version": 1,
        "protocol_sha256": PROTOCOL_SHA256,
        "model_lock": reference(MODEL_LOCK),
        "generic_result": reference(GENERIC_RESULT),
        "outcome": registered,
        "aggregate": {
            "generic_categories": generic["category_counts"],
            "stable_constructive_networks": stable,
            "vector_used_networks": vector_used,
            "rescued_networks": rescued,
            "damaging_rescue_networks": damaging,
            "constrained_noisy_units": sum(
                row["constrained_morphology"] for row in noisy
            ),
            "all_nine_noisy_units": sum(row["all_nine_qualitative"] for row in noisy),
            "noisy_stage_counts": dict(
                sorted(Counter(row["stage"] for row in noisy).items())
            ),
        },
        "networks": network_rows,
        "units": units,
        "fresh_confirmation_authorized": registered == "candidate_rescue",
        "mechanism_control_authorized": False,
        "main_model_promoted": False,
        "selection_performed": False,
    }
    write_arrays(
        PAIR_TABLE, {key: np.asarray(value) for key, value in pair_columns.items()}
    )
    write_arrays(
        WRITE_TABLE, {key: np.asarray(value) for key, value in write_columns.items()}
    )
    write_json_exclusive(RESULT, json_ready(payload))
    lines = [
        "# Postsynaptic vector modulation in minimal single-P M2",
        "",
        f"Registered development outcome: `{registered}`.",
        "",
        f"- Generic categories: {generic['category_counts']}",
        f"- Vector-used networks: {vector_used}/3",
        f"- Rescued networks: {rescued}/3",
        f"- Damaging rescue networks: {damaging}/3",
        f"- All-nine noisy units: {payload['aggregate']['all_nine_noisy_units']}/9",
        f"- Constrained noisy units: {payload['aggregate']['constrained_noisy_units']}/9",
        f"- Noisy stage counts: {payload['aggregate']['noisy_stage_counts']}",
        "",
        "P remained the only episode-persistent plastic state. This exposed three-seed result tests one fixed vector-write recipe and cannot confirm a population effect or establish a biological mechanism.",
        "",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    status = {
        "candidate_rescue": "supporting",
        "mixed_or_damaging": "mixed",
        "generic_inadequate": "unresolved",
    }.get(registered, "valid_negative")
    register(
        status=status,
        finding=f"Frozen vector-modulation development outcome: {registered}.",
    )
    return payload


__all__ = ["report_final", "report_generic"]
