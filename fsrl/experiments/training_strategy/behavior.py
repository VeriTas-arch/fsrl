"""Frozen nine-row behavior definitions with explicit undefined-cohort handling."""

from __future__ import annotations

import math

import numpy as np

from fsrl.analysis.behavioral import analyze_sampled_query_policy
from fsrl.analysis.policy import bundle_logits
from fsrl.experiments.confirmation.reproduction_map import (
    endpoint_statistics,
    mean_pairwise_tau_interval,
    position_profile,
)
from fsrl.infra.provenance import load_json
from fsrl.infra.study_registry import resolve_record
from fsrl.paths import REPO_ROOT
from fsrl.tasks.protocol import ordered_pairs

from .estimands import estimate


def human_references(specification: dict) -> dict:
    contract = specification["decision_contract"]["behavior"]
    source = load_json(REPO_ROOT / contract["reference_contract"])
    frozen = load_json(REPO_ROOT / contract["reference_result"])["human_reference"]
    benchmark = load_json(
        resolve_record(source["registered_sources"]["human_benchmark"]["path"])
    )
    return {
        "intervals": benchmark["bootstrap"]["metrics"],
        "serial": frozen["serial_position_effect"]["interval"],
        "tau": frozen["inter_subject_ranking_diversity"]["interval"],
    }


def _point_interval(values, seed: int, statistics: dict) -> dict:
    result = estimate(
        np.asarray(values, dtype=np.float64), seed=seed, statistics=statistics
    )
    return {
        "point": result["mean"],
        "interval": {key: result["bootstrap"][key] for key in ("lower", "upper")},
        "subjects": result["subjects"],
    }


def behavior_metrics(behavior: dict, seed: int, statistics: dict) -> dict:
    subjects = behavior["subjects"]
    eligible = [subject for subject in subjects if subject["overall_accuracy"] >= 0.5]
    analysis = [
        subject for subject in eligible if subject["ranking_class"] != "correct"
    ]
    classes = {
        name: _point_interval(
            [subject["ranking_class"] == name for subject in eligible], seed, statistics
        )
        for name in ("correct", "self_consistent_incorrect", "self_inconsistent")
    }
    tau = {"point": None, "interval": {"lower": None, "upper": None}}
    if len(analysis) >= 2:
        tau = mean_pairwise_tau_interval(
            [subject["subjective_order_high_to_low"] for subject in analysis],
            samples=statistics["samples"],
            seed=seed,
        )
        if not np.isclose(
            tau["point"], behavior["summary"]["mean_inter_subject_kendall_tau"]
        ):
            raise RuntimeError("behavioral ranking diversity does not reconstruct")
    summary = behavior["summary"]
    metrics = {
        name: _point_interval([subject[name] for subject in subjects], seed, statistics)
        for name in ("learned_accuracy", "nonlearned_accuracy")
    }
    metrics.update(
        {
            "symbolic_distance_effect": summary["symbolic_distance_slope"],
            "serial_position_effect": {
                "profiles": {
                    name: endpoint_statistics(
                        position_profile(
                            behavior["pairs"], "mean_accuracy_all", learned=selector
                        )
                    )
                    for name, selector in (
                        ("all", None),
                        ("learned", True),
                        ("nonlearned", False),
                    )
                },
                "participant_interval": "not registered; frozen point-profile classifier",
            },
            "difficult_pair_bimodality": summary["beta_pair_class_counts_analysis"],
            "stable_within_subject_errors": _point_interval(
                [subject["stable_error_pair_counts"]["80"] > 0 for subject in analysis],
                seed,
                statistics,
            ),
            "self_consistent_vs_inconsistent_errors": {
                name: classes[name]
                for name in ("self_consistent_incorrect", "self_inconsistent")
            },
            "hodge_reconstructed_subjective_ranking": {
                "correct_ranker": classes["correct"],
                "all_orders_are_permutations": bool(eligible)
                and all(
                    sorted(subject["subjective_order_high_to_low"]) == list(range(8))
                    for subject in eligible
                ),
            },
            "inter_subject_ranking_diversity": {**tau, "subjects": len(analysis)},
        }
    )
    return {
        "eligible_subjects": len(eligible),
        "analysis_subjects_excluding_correct_rankers": len(analysis),
        "metrics": metrics,
    }


def _number(value) -> float:
    return float("nan") if value is None else float(value)


def _inside(value, interval: dict) -> bool:
    value = _number(value)
    return math.isfinite(value) and interval["lower"] <= value <= interval["upper"]


def classify_rows(metrics: dict, references: dict) -> dict:
    """Same point classifiers as model_record; undefined values never pass."""

    human = references["intervals"]
    flags = {
        name: {
            "qualitative": _number(metrics[name]["point"]) > 0.5,
            "calibration": _inside(metrics[name]["point"], human[name]),
        }
        for name in ("learned_accuracy", "nonlearned_accuracy")
    }
    distance = metrics["symbolic_distance_effect"]
    serial = metrics["serial_position_effect"]["profiles"]["all"]
    beta = metrics["difficult_pair_bimodality"]
    bimodal = (
        beta["bimodal"] >= 15
        and beta["ordinary_unimodal"] == 0
        and beta["low_accuracy"] == 0
    )
    stable = metrics["stable_within_subject_errors"]["point"]
    classes = metrics["self_consistent_vs_inconsistent_errors"]
    consistent, inconsistent = (
        classes[name]["point"]
        for name in ("self_consistent_incorrect", "self_inconsistent")
    )
    rank = metrics["hodge_reconstructed_subjective_ranking"]
    correct = rank["correct_ranker"]["point"]
    tau = metrics["inter_subject_ranking_diversity"]["point"]
    pairs = {
        "symbolic_distance_effect": (
            _number(distance["mean"]) > 0 and _number(distance["p_vs_zero"]) < 0.05,
            _inside(distance["mean"], human["symbolic_distance_slope"]),
        ),
        "serial_position_effect": (
            serial["both_endpoints_above_interior"],
            _inside(serial["mean_endpoint_contrast"], references["serial"]),
        ),
        "difficult_pair_bimodality": (bimodal, bimodal),
        "stable_within_subject_errors": (
            _number(stable) >= 0.80,
            _inside(stable, human["stable_error_80_analysis_proportion"]),
        ),
        "self_consistent_vs_inconsistent_errors": (
            _number(consistent) > _number(inconsistent),
            _inside(consistent, human["self_consistent_incorrect_proportion"])
            and _inside(inconsistent, human["self_inconsistent_proportion"]),
        ),
        "hodge_reconstructed_subjective_ranking": (
            rank["all_orders_are_permutations"] and _number(correct) < 0.5,
            _inside(correct, human["correct_ranker_proportion"]),
        ),
        "inter_subject_ranking_diversity": (
            _number(tau) < 0.80,
            _inside(tau, references["tau"]),
        ),
    }
    flags.update(
        {
            name: {"qualitative": bool(qualitative), "calibration": bool(calibration)}
            for name, (qualitative, calibration) in pairs.items()
        }
    )
    return flags


def evaluate_behavior(
    bundle: dict, protocol, network_seed: int, specification: dict
) -> dict:
    settings = specification["evaluation"]["liu"]
    schedules = (ordered_pairs(protocol.n_items),) * bundle["logits"].shape[0]
    behavior = analyze_sampled_query_policy(
        protocol,
        bundle_logits(bundle, schedules),
        seed=settings["choice_seed"],
        temperature=settings["temperature"],
    )
    record = behavior_metrics(
        behavior, 85000 + network_seed, specification["statistics"]
    )
    record["flags"] = classify_rows(record["metrics"], human_references(specification))
    record["seed"] = network_seed
    record["bootstrap_scope"] = (
        "Frozen cohort/point definitions; v1 study participant seed; no human interval refit."
    )
    return {"sampled_behavior": behavior, "record": record}
