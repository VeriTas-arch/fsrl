"""Prospective sufficiency outcomes, never a retrospective threshold search."""

import numpy as np

from fsrl.experiments.training_strategy.estimands import estimate, paired_estimate


def competence(record: dict, spec: dict) -> dict:
    criteria = spec["decision_contract"]["competence"]
    checks = {}
    for domain, groups in (
        ("generic", ("learned", "nonlearned")),
        ("liu", ("overall", "nonlearned")),
    ):
        for group in groups:
            value = record["endpoints"][domain]["intact"]["exact_decision"][group][
                "mean"
            ]
            threshold = criteria[f"{domain}_{group}"]
            checks[f"{domain}_{group}"] = {
                "value": value,
                "minimum": threshold,
                "passed": value is not None
                and np.isfinite(value)
                and value >= threshold,
            }
    return {
        "passed": all(check["passed"] for check in checks.values()),
        "checks": checks,
    }


def adequate(record: dict, spec: dict) -> bool:
    flags = record["behavior"]["flags"]
    required = {
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
    return (
        competence(record, spec)["passed"]
        and set(flags) == required
        and all(row["qualitative"] for row in flags.values())
    )


def pair_analysis(trace: dict, score: dict, seed: int, spec: dict) -> dict:
    if (
        trace["input_fingerprint"] != score["input_fingerprint"]
        or trace["generic_fingerprints"] != score["generic_fingerprints"]
    ):
        raise RuntimeError("paired evaluation inputs differ")
    settings = spec["statistics"]
    paired = {}
    for domain, groups in (
        ("generic", ("learned", "nonlearned")),
        ("liu", ("learned", "nonlearned", "retained", "omitted")),
    ):
        paired[domain] = {
            group: paired_estimate(
                trace["raw_endpoints"][domain]["intact"]["probability"][group],
                score["raw_endpoints"][domain]["intact"]["probability"][group],
                seed=(85000 if domain == "liu" else 86000) + seed,
                statistics=settings,
            )
            for group in groups
        }
    probability = {
        name: row["probability"] for name, row in trace["raw_endpoints"]["liu"].items()
    }
    effects = {
        f"intact_minus_{control}_{group}": paired_estimate(
            probability["intact"][group],
            probability[control][group],
            seed=85000 + seed,
            statistics=settings,
        )
        for control, group in (
            ("local_off", "retained"),
            ("local_off", "omitted"),
            ("query_shuffle", "learned"),
            ("evidence_shuffle", "learned"),
        )
    }
    headroom = {
        group: estimate(
            1 - np.asarray(probability["local_off"][group], dtype=np.float64),
            seed=85000 + seed,
            statistics=settings,
        )
        for group in ("retained", "omitted")
    }
    bounds = {
        "omitted_contribution": (effects["intact_minus_local_off_omitted"], 0.0, True),
        "query_specificity": (effects["intact_minus_query_shuffle_learned"], 0.0, True),
        "evidence_specificity": (
            effects["intact_minus_evidence_shuffle_learned"],
            0.0,
            True,
        ),
        "between_recipe": (paired["liu"]["omitted"], 0.0, True),
        "retained_preservation": (paired["liu"]["retained"], -0.02, False),
    }
    checks = {}
    for name, (value, threshold, strict) in bounds.items():
        lower = value["bootstrap"]["lower"]
        checks[name] = {
            "lower": lower,
            "threshold": threshold,
            "strict": strict,
            "passed": lower is not None
            and np.isfinite(lower)
            and (lower > threshold if strict else lower >= threshold),
        }
    return {
        "paired_probability": paired,
        "trace_acute_effects": effects,
        "trace_global_probability_headroom": headroom,
        "local_support": checks,
        "outcome": outcome(trace, score, checks, spec),
    }


def outcome(trace: dict, score: dict, checks: dict, spec: dict) -> str:
    if adequate(score, spec):
        return "score_only_sufficient"
    if adequate(trace, spec):
        if len(checks) == 5 and all(row["passed"] for row in checks.values()):
            return "compact_dual_state_candidate"
        return "compact_behavior_solution_mechanism_unresolved"
    if any(competence(record, spec)["passed"] for record in (trace, score)):
        return "competent_behavior_incomplete"
    return "fixed_recipe_insufficient"


def study_outcome(pairs: dict, spec: dict) -> str:
    if set(pairs) != {str(seed) for seed in spec["seeds"]["mandatory"]}:
        raise RuntimeError("every mandatory training stream is required")
    labels = {pair["outcome"] for pair in pairs.values()}
    return next(iter(labels)) if len(labels) == 1 else "heterogeneous"
