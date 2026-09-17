"""Pure registered decisions for the historical morphology re-audit."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from fsrl.experiments.minimal_single_p_promotion.decisions import ROWS


def all_nine(result: Mapping) -> bool:
    flags = result["liu"]["routes"]["full"]["behavior"]["historical_nine_rows"]["flags"]
    return set(flags) == set(ROWS) and all(row["qualitative"] for row in flags.values())


def evidence_binding(result: Mapping) -> bool:
    lower = result["liu"]["effects"]["intact_minus_evidence_shuffle_learned"][
        "bootstrap"
    ]["lower"]
    return lower is not None and lower > 0.0


def inherited_competence(result: Mapping) -> bool:
    return bool(result["generic"]["competence"]) and bool(
        result["liu"]["routes"]["full"]["competence"]
    )


def constrained_unit(
    *,
    competent: bool,
    binding: bool,
    nine: bool,
    latent_bimodal: int,
    sampled_bimodal: int,
    bad_pair: bool,
) -> bool:
    return (
        competent
        and binding
        and nine
        and latent_bimodal >= 15
        and sampled_bimodal >= 15
        and not bad_pair
    )


def error_inflation(
    *, sampled_bimodal: int, control_sampled_bimodal: int, nine: bool, bad_pair: bool
) -> bool:
    return sampled_bimodal > control_sampled_bimodal and (not nine or bad_pair)


def family_replicated(units: Sequence[Mapping], seeds: Sequence[int]) -> bool:
    return all(
        any(row["seed"] == seed and row["constrained_morphology"] for row in units)
        for seed in seeds
    )


def outcome(constrained: int, replicated_families: int) -> str:
    if replicated_families == 2:
        return "cross_family_current_standard_precedent"
    if replicated_families == 1:
        return "single_family_current_standard_precedent"
    if constrained:
        return "isolated_current_standard_precedent"
    return "no_current_standard_precedent"


__all__ = [
    "all_nine",
    "constrained_unit",
    "error_inflation",
    "evidence_binding",
    "family_replicated",
    "inherited_competence",
    "outcome",
]
