"""Original cohort point estimands, with whole-cohort simulation uncertainty."""

from collections import Counter

import numpy as np
from scipy.stats import norm

from fsrl.experiments.confirmation.reproduction_map import (
    endpoint_statistics,
    position_profile,
)
from fsrl.experiments.training_strategy.behavior import classify_rows


def extract_values(metrics: dict) -> dict:
    return {
        "learned_accuracy": metrics["learned_accuracy"]["point"],
        "nonlearned_accuracy": metrics["nonlearned_accuracy"]["point"],
        "symbolic_distance_effect": metrics["symbolic_distance_effect"]["mean"],
        "serial_position_effect": metrics["serial_position_effect"]["profiles"]["all"][
            "mean_endpoint_contrast"
        ],
        "stable_within_subject_errors": metrics["stable_within_subject_errors"][
            "point"
        ],
        **{
            key: metrics["self_consistent_vs_inconsistent_errors"][key]["point"]
            for key in ("self_consistent_incorrect", "self_inconsistent")
        },
        "correct_ranker": metrics["hodge_reconstructed_subjective_ranking"][
            "correct_ranker"
        ]["point"],
        "inter_subject_ranking_diversity": metrics["inter_subject_ranking_diversity"][
            "point"
        ],
    }


def cohort_record(behavior: dict, references: dict) -> dict:
    subjects, summary = behavior["subjects"], behavior["summary"]
    eligible = [row for row in subjects if row["overall_accuracy"] >= 0.5]
    classes = summary["ranking_class_counts"]
    proportions = {
        key: {"point": count / len(eligible) if eligible else None}
        for key, count in classes.items()
    }
    metrics: dict = {
        key: {"point": float(np.mean([row[key] for row in subjects]))}
        for key in ("learned_accuracy", "nonlearned_accuracy")
    }
    metrics.update(
        {
            "symbolic_distance_effect": summary["symbolic_distance_slope"],
            "serial_position_effect": {
                "profiles": {
                    "all": endpoint_statistics(
                        position_profile(behavior["pairs"], "mean_accuracy_all")
                    )
                }
            },
            "stable_within_subject_errors": {
                "point": summary["stable_error_subject_prevalence"]["80"]["analysis"]
            },
            "self_consistent_vs_inconsistent_errors": {
                key: proportions[key]
                for key in ("self_consistent_incorrect", "self_inconsistent")
            },
            "hodge_reconstructed_subjective_ranking": {
                "correct_ranker": proportions["correct"],
                "all_orders_are_permutations": bool(eligible)
                and all(
                    sorted(row["subjective_order_high_to_low"]) == list(range(8))
                    for row in eligible
                ),
            },
            "inter_subject_ranking_diversity": {
                "point": summary["mean_inter_subject_kendall_tau"]
            },
            "difficult_pair_bimodality": summary["beta_pair_class_counts_analysis"],
        }
    )
    return {
        "values": extract_values(metrics),
        "flags": classify_rows(metrics, references),
        "distance_test": metrics["symbolic_distance_effect"],
        "morphology": metrics["difficult_pair_bimodality"],
        "eligible_subjects": len(eligible),
        "analysis_subjects": summary["analysis_subjects_excluding_correct_rankers"],
    }


def reference_intervals(references: dict) -> dict:
    aliases = {
        "symbolic_distance_effect": "symbolic_distance_slope",
        "stable_within_subject_errors": "stable_error_80_analysis_proportion",
        "self_consistent_incorrect": "self_consistent_incorrect_proportion",
        "self_inconsistent": "self_inconsistent_proportion",
        "correct_ranker": "correct_ranker_proportion",
    }
    names = ["learned_accuracy", "nonlearned_accuracy", *aliases]
    result = {
        name: {
            key: references["intervals"][aliases.get(name, name)][key]
            for key in ("lower", "upper")
        }
        for name in names
    }
    return {
        **result,
        "serial_position_effect": references["serial"],
        "inter_subject_ranking_diversity": references["tau"],
    }


def interval_classification(interval, reference) -> str:
    if interval is None:
        return "unresolved"
    if interval["lower"] > reference["upper"]:
        return "sustained_above_reference"
    if interval["upper"] < reference["lower"]:
        return "sustained_below_reference"
    if (
        interval["lower"] >= reference["lower"]
        and interval["upper"] <= reference["upper"]
    ):
        return "mean_within_reference"
    return "boundary_unresolved"


def wilson(values) -> dict:
    values = np.asarray(values, dtype=bool)
    count, successes = len(values), int(values.sum())
    z = norm.ppf(0.975)
    rate = successes / count
    denominator = 1 + z * z / count
    center = (rate + z * z / (2 * count)) / denominator
    radius = z * np.sqrt(rate * (1 - rate) / count + z * z / (4 * count**2))
    return {
        "successes": successes,
        "cohorts": count,
        "rate": rate,
        "lower": 0.0
        if successes == 0
        else max(0.0, float(center - radius / denominator)),
        "upper": 1.0
        if successes == count
        else min(1.0, float(center + radius / denominator)),
    }


def continuous_summary(rows, seed, spec, references) -> dict:
    names = spec["estimands"]["continuous"]
    values = np.asarray(
        [[row["values"][name] for name in names] for row in rows], float
    )
    settings = spec["estimands"]["uncertainty"]
    rng = np.random.default_rng(settings["seed_base"] + seed)
    counts = rng.multinomial(
        len(rows), np.full(len(rows), 1 / len(rows)), size=settings["samples"]
    )
    means = counts @ values / len(rows)
    intervals, result = reference_intervals(references), {}
    for column, name in enumerate(names):
        undefined = np.flatnonzero(~np.isfinite(values[:, column])).tolist()
        interval = None
        point = None
        if not undefined:
            low, high = np.quantile(means[:, column], [0.025, 0.975])
            interval = {"lower": float(low), "upper": float(high)}
            point = float(values[:, column].mean())
        result[name] = {
            "mean": point,
            "interval": interval,
            "reference": intervals[name],
            "undefined_cohorts": undefined,
            "classification": interval_classification(interval, intervals[name]),
        }
    return result


def summarize_fit(rows, seed, spec, references) -> dict:
    if [row["cohort"] for row in rows] != list(range(spec["cohorts"]["count"])):
        raise RuntimeError("diagnostic summary requires every registered cohort")
    continuous = continuous_summary(rows, seed, spec, references)
    flags = {
        name: {
            kind: wilson([row["flags"][name][kind] for row in rows])
            for kind in ("qualitative", "calibration")
        }
        for name in rows[0]["flags"]
    }
    all_flags = {
        kind: [all(flag[kind] for flag in row["flags"].values()) for row in rows]
        for kind in ("qualitative", "calibration")
    }
    joint = np.asarray(all_flags["qualitative"]) & all_flags["calibration"]
    morphology = Counter(
        tuple(
            row["morphology"][key]
            for key in ("bimodal", "ordinary_unimodal", "low_accuracy")
        )
        for row in rows
    )
    distance = continuous["symbolic_distance_effect"]
    return {
        "outcome": distance["classification"],
        "continuous": continuous,
        "primary_mean_minus_upper": None
        if distance["mean"] is None
        else distance["mean"] - distance["reference"]["upper"],
        "pass_rates": flags,
        "all_nine": {
            **{k: wilson(v) for k, v in all_flags.items()},
            "joint": wilson(joint),
        },
        "morphology_joint": [
            {"counts_bimodal_unimodal_low": list(key), "cohorts": value}
            for key, value in sorted(morphology.items())
        ],
        "main_model_promoted": False,
        "parent_outcome_changed": False,
    }
