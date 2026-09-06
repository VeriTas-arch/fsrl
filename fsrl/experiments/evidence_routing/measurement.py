"""Paired selection-aware order contrasts and fixed-panel internal preferences."""

from itertools import combinations

import numpy as np

from fsrl.analysis.behavioral import kendall_tau_positions
from fsrl.analysis.hodge import build_complete_graph_geometry, hodge_potentials
from fsrl.analysis.policy import exact_probability
from fsrl.analysis.statistics import bootstrap_counts
from fsrl.experiments.memory_structure.measurement import diversity
from fsrl.experiments.training_strategy.estimands import estimate


def tau_matrix(orders: np.ndarray) -> np.ndarray:
    positions = np.argsort(orders, axis=1)
    result = np.eye(len(orders))
    for first, second in combinations(range(len(orders)), 2):
        result[first, second] = result[second, first] = kendall_tau_positions(
            positions[first], positions[second]
        )
    return result


def weighted_tau(matrix: np.ndarray, counts: np.ndarray) -> np.ndarray:
    sizes = counts.sum(1)
    numerator = np.einsum("bi,ij,bj->b", counts, matrix, counts, optimize=True) - sizes
    return np.divide(
        numerator,
        sizes * (sizes - 1),
        out=np.full(len(counts), np.nan),
        where=sizes >= 2,
    )


def paired_tau(
    shared_orders, isolated_orders, shared_mask, isolated_mask, seed, statistics
):
    """Resample original IDs jointly before applying each observed inclusion rule."""
    n = len(shared_orders)
    if len(isolated_orders) != n:
        raise ValueError("paired orders need identical subject axes")
    counts = bootstrap_counts(np.random.default_rng(seed), statistics["samples"], n)
    draws, points = [], []
    for orders, mask in (
        (shared_orders, shared_mask),
        (isolated_orders, isolated_mask),
    ):
        matrix = tau_matrix(np.asarray(orders))
        mask = np.asarray(mask, dtype=bool)
        draws.append(weighted_tau(matrix, counts * mask))
        points.append(weighted_tau(matrix, mask[None, :].astype(float))[0])
    delta = draws[1] - draws[0]
    invalid = int((~np.isfinite(delta)).sum())
    interval = [None, None] if invalid else np.quantile(delta, [0.025, 0.975]).tolist()
    point = points[1] - points[0]
    return {
        "point": float(point) if np.isfinite(point) else None,
        "interval": {"lower": interval[0], "upper": interval[1]},
        "undefined_draws": invalid,
        "resampled_subjects": n,
        "shared_analysis_subjects": int(np.sum(shared_mask)),
        "isolated_analysis_subjects": int(np.sum(isolated_mask)),
        "common_analysis_subjects": int(
            np.sum(np.asarray(shared_mask) & isolated_mask)
        ),
    }


def selection(sampled: dict) -> tuple[np.ndarray, np.ndarray]:
    subjects = sampled["subjects"]
    return (
        np.asarray([s["subjective_order_high_to_low"] for s in subjects]),
        np.asarray(
            [
                s["overall_accuracy"] >= 0.5 and s["ranking_class"] != "correct"
                for s in subjects
            ]
        ),
    )


def internal_preferences(margins: np.ndarray, protocol, spec: dict, seed: int) -> tuple:
    geometry = build_complete_graph_geometry(protocol)
    field = (margins[:, ::2] - margins[:, 1::2]) / 2
    potentials = hodge_potentials(field, geometry)
    orders = np.argsort(-potentials, axis=1, kind="stable")
    oriented = margins.reshape(len(margins), len(geometry.pairs), 2)
    signs = geometry.true_sign[None, :, None] * np.asarray([1, -1])
    probability = exact_probability(
        oriented * signs, spec["evaluation"]["liu"]["temperature"]
    ).mean(2)
    wrong = probability <= 0.2
    arrays = {
        "field": field,
        "potentials": potentials,
        "orders": orders,
        "correct_probability": probability,
        "stable_error_density": wrong.mean(1),
        "stable_error_prevalence": wrong.any(1).astype(float),
    }
    bootstrap_seed = spec["statistics"]["seed_offset"] + seed
    summary = {
        "all_subject_order_tau": diversity(
            orders.tolist(), bootstrap_seed, spec["statistics"]["samples"]
        ),
        **{
            key: estimate(
                arrays[key], seed=bootstrap_seed, statistics=spec["statistics"]
            )
            for key in ("stable_error_density", "stable_error_prevalence")
        },
    }
    return summary, arrays
