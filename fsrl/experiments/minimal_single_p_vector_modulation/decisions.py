"""Frozen vector-modulation development decision rules."""

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


def outcome(
    stable: int,
    rescued_networks: int,
    damaging_rescued_networks: int,
    vector_used_networks: int,
) -> str:
    if stable < 2:
        return "generic_inadequate"
    if damaging_rescued_networks:
        return "mixed_or_damaging"
    if rescued_networks >= 2:
        return "candidate_rescue"
    if vector_used_networks:
        return "used_without_rescue"
    return "vector_not_used"


__all__ = ["all_nine", "evidence_binding", "generic_category", "outcome"]
