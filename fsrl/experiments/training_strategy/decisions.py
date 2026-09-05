"""Frozen per-network decisions; no pooled or majority-vote rescue."""

from __future__ import annotations

import math


def _comparison(value: float | None, threshold: float, operator: str) -> bool:
    if value is None or not math.isfinite(value):
        return False
    return {
        ">=": value >= threshold,
        ">": value > threshold,
        "<=": value <= threshold,
    }[operator]


def criterion(
    summary: dict, threshold: float, *, statistic: str, operator: str
) -> dict:
    value = summary["mean"] if statistic == "mean" else summary["bootstrap"][statistic]
    return {
        "value": value,
        "statistic": statistic,
        "operator": operator,
        "threshold": threshold,
        "subjects": summary["subjects"],
        "passed": _comparison(value, threshold, operator),
    }


def competence(summaries: dict, specification: dict) -> dict:
    contract = specification["decision_contract"]["per_condition_competence"]
    rules = {}
    for domain in ("generic", "liu"):
        minima = contract[f"{domain}_intact_exact_decision_mean_minimum"]
        for group, minimum in minima.items():
            rules[f"{domain}_{group}"] = criterion(
                summaries[domain]["intact"]["exact_decision"][group],
                minimum,
                statistic="mean",
                operator=">=",
            )
    rules["liu_transitivity"] = criterion(
        summaries["constructive"]["intact_transitive_triplet_fraction"],
        contract["liu_intact_transitive_triplet_mean_minimum"],
        statistic="mean",
        operator=">=",
    )
    return {"checks": rules, "passed": all(row["passed"] for row in rules.values())}


def mechanism(effects: dict) -> dict:
    # Numeric translations of the immutable v1 contract's named causal rules.
    rules = {
        "global_necessity": (
            ("intact_minus_P_off_nonlearned", "lower", ">=", 0.10),
            ("P_off_nonlearned", "upper", "<=", 0.55),
        ),
        "remote_reassembly": (
            ("global_remote_absolute", "lower", ">", 0.01),
            ("global_third_party_relational", "lower", ">", 0.05),
        ),
        "direct_local_fidelity": (
            ("intact_minus_local_off_retained", "lower", ">=", 0.01),
            ("intact_minus_local_off_omitted", "lower", ">=", 0.01),
        ),
        "query_evidence_specificity": (
            ("intact_minus_query_shuffle_learned", "lower", ">=", 0.01),
            ("intact_minus_evidence_shuffle_learned", "lower", ">=", 0.01),
        ),
        "local_only_partition": (
            ("P_off_learned", "lower", ">", 0.50),
            ("P_off_nonlearned", "upper", "<=", 0.55),
            ("local_remote_minus_quarter_combined", "upper", "<=", 0.0),
        ),
    }
    links = {}
    for name, tests in rules.items():
        checks = {
            key: criterion(effects[key], bound, statistic=statistic, operator=operator)
            for key, statistic, operator, bound in tests
        }
        links[name] = {
            "checks": checks,
            "passed": all(check["passed"] for check in checks.values()),
        }
    return {"links": links, "passed": all(link["passed"] for link in links.values())}


def noninferiority(paired: dict, specification: dict) -> dict:
    contract = specification["decision_contract"]["paired_noninferiority"]
    checks = {
        f"{domain}_{group}": criterion(
            paired[domain][group],
            contract["minimum_lower_bound"],
            statistic="lower",
            operator=">=",
        )
        for domain in ("generic", "liu")
        for group in contract[f"{domain}_groups"]
    }
    return {
        "checks": checks,
        "passed": all(check["passed"] for check in checks.values()),
    }


def behavior_preservation(record: dict, specification: dict) -> dict:
    required = specification["decision_contract"]["behavior"][
        "historically_quantitative_rows"
    ]
    flags = record["flags"]
    # The nine row identities are fixed, including the three old mismatches.
    names = set(required) | {
        "symbolic_distance_effect",
        "serial_position_effect",
        "self_consistent_vs_inconsistent_errors",
    }
    if set(flags) != names:
        raise ValueError("behavior decision requires exactly the nine registered rows")
    qualitative = {name: flags[name]["qualitative"] is True for name in sorted(names)}
    quantitative = {name: flags[name]["calibration"] is True for name in required}
    return {
        "qualitative": qualitative,
        "historically_quantitative": quantitative,
        "passed": all(qualitative.values()) and all(quantitative.values()),
    }


def outcome(condition_decisions: dict, paired_decision: dict) -> str:
    staged, joint = (condition_decisions[name] for name in ("matched_staged", "joint"))
    ordered = (
        (joint["competence"]["passed"], "joint_recipe_insufficient"),
        (staged["competence"]["passed"], "matched_comparator_insufficient"),
        (paired_decision["passed"], "competent_but_not_noninferior"),
        (
            staged["mechanism"]["passed"] and joint["mechanism"]["passed"],
            "alternative_computational_solution",
        ),
        (joint["behavior"]["passed"], "mechanism_preserved_behavior_incomplete"),
    )
    return next(
        (label for passed, label in ordered if not passed),
        "single_stage_preserves_mechanism_and_behavior",
    )


def study_outcome(seed_results: dict, specification: dict) -> str:
    expected = {str(seed) for seed in specification["seeds"]["mandatory"]}
    if set(seed_results) != expected:
        raise ValueError("all three mandatory paired seeds must be reported")
    labels = {row["outcome"] for row in seed_results.values()}
    order = specification["decision_contract"]["outcomes"]
    if not labels.issubset(order):
        raise ValueError("unregistered outcome")
    return next(label for label in order if label in labels)


def cost_comparison(staged: dict, joint: dict) -> dict:
    metrics = ("warm_training_seconds", "peak_allocated_bytes")
    if any(
        not math.isfinite(row[key]) or row[key] <= 0
        for row in (staged, joint)
        for key in metrics
    ):
        raise ValueError("measured training time and allocated memory must be positive")
    ratios = {key: joint[key] / staged[key] for key in metrics}
    return {
        "joint_over_staged": ratios,
        "efficiency_advantage": all(value <= 1 for value in ratios.values()),
    }
