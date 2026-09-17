"""Pure hierarchical decision rules for single-P anytime V1."""

from __future__ import annotations

OUTCOME_PRECEDENCE = (
    "noninterpretable",
    "training_recipe_failure",
    "historical_task_degradation",
    "anytime_competence_failure",
    "no_horizon_benefit",
    "horizon_tradeoff",
    "anytime_admitted",
)


def classify(
    *,
    integrity: bool,
    fresh_fixed_valid: bool,
    historical_preserved: bool,
    anytime_competent: bool,
    short_improved: bool,
    long_improved: bool,
) -> str:
    if not integrity:
        return "noninterpretable"
    if not fresh_fixed_valid:
        return "training_recipe_failure"
    if not historical_preserved:
        return "historical_task_degradation"
    if not anytime_competent:
        return "anytime_competence_failure"
    if not short_improved and not long_improved:
        return "no_horizon_benefit"
    if short_improved != long_improved:
        return "horizon_tradeoff"
    return "anytime_admitted"


def interval_gate(
    *,
    ce_upper: float,
    probability_lower: float,
    ce_limit: float = 0.0,
    probability_margin: float = -0.02,
) -> bool:
    return ce_upper < ce_limit and probability_lower >= probability_margin


__all__ = ["OUTCOME_PRECEDENCE", "classify", "interval_gate"]
