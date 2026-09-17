"""Frozen M2-alpha decision rules."""

from __future__ import annotations

from fsrl.experiments.minimal_single_p_promotion.decisions import generic_category


def evidence_binding(result: dict) -> bool:
    lower = result["liu"]["effects"]["intact_minus_evidence_shuffle_learned"][
        "bootstrap"
    ]["lower"]
    return lower is not None and lower > 0.0


def all_nine(result: dict) -> bool:
    flags = result["liu"]["routes"]["full"]["behavior"]["historical_nine_rows"]["flags"]
    return all(row["qualitative"] for row in flags.values())


def outcome(interpretable: int, constrained: int, inflated: int) -> str:
    if interpretable < 40:
        return "generic_inadequate"
    if constrained >= 40:
        return "robust_constrained_rescue"
    if inflated >= 40:
        return "error_inflation"
    if constrained:
        return "partial_or_mixed"
    return "no_constrained_rescue"


__all__ = ["all_nine", "evidence_binding", "generic_category", "outcome"]
