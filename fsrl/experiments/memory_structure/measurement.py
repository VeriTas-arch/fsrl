"""Task ability and organized errors, separated from human-interval matching."""

from itertools import combinations

import numpy as np

from fsrl.analysis.behavioral import analyze_sampled_query_policy, kendall_tau_positions
from fsrl.analysis.hodge import build_complete_graph_geometry, gradient_energy_fraction
from fsrl.analysis.policy import bundle_logits
from fsrl.analysis.statistics import bootstrap_counts
from fsrl.experiments.training_strategy.behavior import (
    behavior_metrics,
    classify_rows,
    human_references,
)
from fsrl.experiments.training_strategy.estimands import estimate
from fsrl.tasks.protocol import ordered_pairs


def diversity(orders: list, seed: int, samples: int) -> dict:
    count = len(orders)
    if count < 2:
        return {"point": None, "interval": {"lower": None, "upper": None}}
    positions = np.argsort(np.asarray(orders), axis=1)
    matrix = np.eye(count)
    for first, second in combinations(range(count), 2):
        matrix[first, second] = matrix[second, first] = kendall_tau_positions(
            positions[first], positions[second]
        )
    counts = bootstrap_counts(np.random.default_rng(seed), samples, count)
    draws = (
        np.einsum("bi,ij,bj->b", counts, matrix, counts, optimize=True) - counts.sum(1)
    ) / (count * (count - 1))
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return {
        "point": float(matrix[np.triu_indices(count, 1)].mean()),
        "interval": {"lower": float(lower), "upper": float(upper)},
    }


def summarize_behavior(bundle: dict, protocol, spec: dict, seed: int) -> tuple:
    settings = spec["evaluation"]["liu"]
    behavior = analyze_sampled_query_policy(
        protocol,
        bundle_logits(
            bundle, (ordered_pairs(protocol.n_items),) * bundle["logits"].shape[0]
        ),
        seed=settings["choice_seed"],
        temperature=settings["temperature"],
    )
    subjects = behavior["subjects"]
    eligible = [s for s in subjects if s["overall_accuracy"] >= 0.5]
    analysis = [s for s in eligible if s["ranking_class"] != "correct"]
    geometry = build_complete_graph_geometry(protocol)
    field = (bundle["logits"][:, ::2] - bundle["logits"][:, 1::2]) / 2
    raw = {
        "coherence": gradient_energy_fraction(field, geometry),
        "stable_prevalence": np.asarray(
            [s["stable_error_pair_counts"]["80"] > 0 for s in analysis], dtype=float
        ),
        "organized_error_difference": np.asarray(
            [
                int(s["ranking_class"] == "self_consistent_incorrect")
                - int(s["ranking_class"] == "self_inconsistent")
                for s in eligible
            ],
            dtype=float,
        ),
        "stable_error_density": np.asarray(
            [
                s["stable_error_pair_counts"]["80"] / len(geometry.pairs)
                for s in subjects
            ]
        ),
    }
    bootstrap_seed = spec["statistics"]["seed_offset"] + seed
    result: dict = {
        key: estimate(value, seed=bootstrap_seed, statistics=spec["statistics"])
        for key, value in raw.items()
    }
    result["diversity"] = diversity(
        [s["subjective_order_high_to_low"] for s in analysis],
        bootstrap_seed,
        spec["statistics"]["samples"],
    )
    result["eligible_subjects"] = len(eligible)
    result["analysis_subjects"] = len(analysis)
    if protocol.n_items == 8:
        historical = behavior_metrics(behavior, bootstrap_seed, spec["statistics"])
        historical["flags"] = classify_rows(
            historical["metrics"], human_references(spec)
        )
        result["historical_nine_rows"] = historical
    return result, raw, behavior


def above(row: dict, threshold: float) -> bool:
    value = row["bootstrap"]["lower"]
    return value is not None and value > threshold


def competence(endpoints: dict) -> bool:
    return all(
        above(endpoints["probability"][group], 0.5)
        for group in ("learned", "nonlearned")
    )


def core_flags(summary: dict, behavior: dict, generic_passed: bool) -> dict:
    tau = behavior["diversity"]["interval"]["upper"]
    return {
        "competence": generic_passed and competence(summary),
        "coherence": above(behavior["coherence"], 0.95),
        "stable_errors": above(behavior["stable_prevalence"], 0.0),
        "organized_errors": above(behavior["organized_error_difference"], 0.0),
        "individual_diversity": tau is not None and tau < 0.8,
    }


def removal_effects(intact: np.ndarray, removed: np.ndarray, protocol) -> dict:
    pairs = ordered_pairs(protocol.n_items)
    delta = np.abs(intact[None] - removed)
    direct, remote = [], []
    for index, relation in enumerate(protocol.support_pairs_higher_lower):
        direct_mask = np.asarray([set(pair) == set(relation) for pair in pairs])
        remote_mask = np.asarray(
            [not set(pair).intersection(relation) for pair in pairs]
        )
        direct.append(delta[index][:, direct_mask].mean(1))
        remote.append(delta[index][:, remote_mask].mean(1))
    return {
        "direct_absolute_margin_change": np.asarray(direct).mean(0),
        "disjoint_remote_absolute_margin_change": np.asarray(remote).mean(0),
    }
