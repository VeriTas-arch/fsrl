"""Unchanged nine-row behavior and the candidate's prospective admission gates."""

import numpy as np

from fsrl.experiments.training_strategy.estimands import paired_estimate

BEHAVIOR_ROWS = {
    "learned_accuracy",
    "nonlearned_accuracy",
    "symbolic_distance_effect",
    "serial_position_effect",
    "difficult_pair_bimodality",
    "stable_within_subject_errors",
    "self_consistent_vs_inconsistent_errors",
    "hodge_reconstructed_subjective_ranking",
    "inter_subject_ranking_diversity",
}


def above(value, threshold) -> bool:
    return value is not None and bool(np.isfinite(value) and value > threshold)


def fit_decision(result: dict, spec: dict, seed: int) -> dict:
    competence = {
        group: above(
            result["endpoints"]["generic"]["intact"]["exact_decision"][group][
                "bootstrap"
            ]["lower"],
            0.5,
        )
        for group in ("learned", "nonlearned")
    }
    effects = {}
    for domain in ("generic", "liu"):
        values = result["raw_endpoints"][domain]
        effects[domain] = paired_estimate(
            values["intact"]["probability"]["overall"],
            values["shuffled"]["probability"]["overall"],
            seed=(86000 if domain == "generic" else 85000) + seed,
            statistics=spec["statistics"],
        )
    flags = result["behavior"]["flags"]
    if set(flags) != BEHAVIOR_ROWS:
        raise RuntimeError("behavior evaluation must retain every original row")
    return {
        "competence": competence,
        "binding_effects": effects,
        "binding_passed": all(
            above(row["bootstrap"]["lower"], 0) for row in effects.values()
        ),
        "qualitative_passed": all(row["qualitative"] for row in flags.values()),
        "quantitative_passed": all(row["calibration"] for row in flags.values()),
    }


def recipe_decision(fits: dict, seeds: list[int], recovery_outcome: str) -> dict:
    if set(fits) != {str(seed) for seed in seeds}:
        raise RuntimeError("recipe classification requires every mandatory fit")
    competence = all(
        all(row["decision"]["competence"].values()) for row in fits.values()
    )
    behavior = all(
        row["decision"]["qualitative_passed"] and row["decision"]["quantitative_passed"]
        for row in fits.values()
    )
    binding = all(row["decision"]["binding_passed"] for row in fits.values())
    if not competence:
        outcome = "incompetent_recipe"
    elif not behavior:
        outcome = "partial_behavioral_reproduction"
    elif binding and recovery_outcome == "distinguishable_on_registered_screen":
        outcome = "complete_pilot_reproduction_recoverable_codec"
    else:
        outcome = "complete_pilot_reproduction_specificity_unresolved"
    return {
        "outcome": outcome,
        "eligible_for_unchanged_replication": competence and behavior and binding,
        "main_model_promoted": False,
    }
