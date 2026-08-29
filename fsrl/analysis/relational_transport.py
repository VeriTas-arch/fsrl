"""Reusable estimands for relational mechanism transport evaluations."""

from __future__ import annotations

from itertools import combinations

import numpy as np

from fsrl.evaluation.metrics import count_circular_triads
from fsrl.tasks.protocol import RankingProtocol

from .behavioral import kendall_tau_positions
from .hodge import (
    CompleteGraphGeometry,
    gradient_energy_fraction,
    hodge_potentials,
    kendall_tau_scores,
)
from .statistics import (
    bootstrap_counts,
    json_values,
    stable_sigmoid,
    summarize_subjects,
)

REGISTERED_ITEM_COUNT = 8


def _subject_group_mean(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.mean(np.asarray(values, dtype=np.float64)[:, mask], axis=1)


def condition_metrics(
    field: np.ndarray,
    geometry: CompleteGraphGeometry,
    learned_mask: np.ndarray,
    counts: np.ndarray,
    interval: float,
    temperature: float,
) -> dict:
    correct = np.asarray(field, dtype=np.float64) * geometry.true_sign[None]
    decision = (correct > 0.0).astype(np.float64)
    probability = stable_sigmoid(correct / temperature)
    groups = {
        "overall": np.ones(len(geometry.pairs), dtype=bool),
        "learned": learned_mask,
        "nonlearned": ~learned_mask,
    }
    raw = {
        "exact_decision_accuracy": {
            name: _subject_group_mean(decision, mask) for name, mask in groups.items()
        },
        "correct_probability": {
            name: _subject_group_mean(probability, mask)
            for name, mask in groups.items()
        },
    }
    return {
        "summary": {
            metric: {
                name: summarize_subjects(values, counts, interval=interval)
                for name, values in rows.items()
            }
            for metric, rows in raw.items()
        },
        "raw_subject": {
            metric: {name: json_values(values) for name, values in rows.items()}
            for metric, rows in raw.items()
        },
        "correct_signed_field": json_values(correct),
    }


def constructive_metrics(
    intact_field: np.ndarray,
    global_field: np.ndarray,
    geometry: CompleteGraphGeometry,
    counts: np.ndarray,
    interval: float,
) -> dict:
    """Summarize the registered eight-item constructive estimands."""

    intact_gradient = gradient_energy_fraction(intact_field, geometry)
    global_gradient = gradient_energy_fraction(global_field, geometry)
    potentials = hodge_potentials(intact_field, geometry)
    true = np.broadcast_to(geometry.true_potential, potentials.shape)
    hodge_tau = kendall_tau_scores(potentials, true)
    transitivity = []
    for row in intact_field:
        winners = {
            pair: pair[0] if row[index] > 0.0 else pair[1]
            for index, pair in enumerate(geometry.pairs)
        }
        circular = count_circular_triads(winners, len(geometry.true_potential))
        transitivity.append(
            1.0 - circular / len(tuple(combinations(range(REGISTERED_ITEM_COUNT), 3)))
        )
    raw = {
        "intact_gradient_energy_fraction": intact_gradient,
        "a_off_gradient_energy_fraction": global_gradient,
        "intact_transitive_triplet_fraction": np.asarray(transitivity),
        "intact_hodge_order_kendall_tau_to_true": hodge_tau,
    }
    return {
        "summary": {
            name: summarize_subjects(values, counts, interval=interval)
            for name, values in raw.items()
        },
        "raw_subject": {name: json_values(values) for name, values in raw.items()},
    }


def relation_loo_metrics(
    intact: np.ndarray,
    loo: np.ndarray,
    relations: tuple[tuple[int, int], ...],
    geometry: CompleteGraphGeometry,
    counts: np.ndarray,
    interval: float,
) -> dict:
    """Summarize the registered eight-item relation-LOO estimands."""

    influence = intact[None] - loo
    remote = np.empty((len(relations), intact.shape[0]), dtype=np.float64)
    third_party = np.full_like(remote, np.nan)
    intact_potential = hodge_potentials(intact, geometry)
    for relation_index, relation in enumerate(relations):
        endpoints = set(relation)
        remote_mask = np.asarray(
            [not endpoints.intersection(pair) for pair in geometry.pairs], dtype=bool
        )
        remote[relation_index] = np.mean(
            np.abs(influence[relation_index][:, remote_mask]), axis=1
        )
        delta = intact_potential - hodge_potentials(loo[relation_index], geometry)
        denominator = np.sum(delta * delta, axis=1)
        third_items = np.asarray(
            [item for item in range(REGISTERED_ITEM_COUNT) if item not in endpoints],
            dtype=np.int64,
        )
        relational = delta[:, third_items] - np.mean(
            delta[:, third_items], axis=1, keepdims=True
        )
        numerator = np.sum(relational * relational, axis=1)
        third_party[relation_index] = np.divide(
            numerator,
            denominator,
            out=np.full_like(numerator, np.nan),
            where=denominator > 1e-14,
        )
    subject_remote = np.mean(remote, axis=0)
    finite = np.sum(np.isfinite(third_party), axis=0)
    subject_third = np.divide(
        np.nansum(third_party, axis=0),
        finite,
        out=np.full(intact.shape[0], np.nan),
        where=finite > 0,
    )
    return {
        "summary": {
            "remote_absolute": summarize_subjects(
                subject_remote, counts, interval=interval
            ),
            "third_party_relational": summarize_subjects(
                subject_third, counts, interval=interval
            ),
        },
        "raw_subject": {
            "remote_absolute": json_values(subject_remote),
            "third_party_relational": json_values(subject_third),
        },
        "raw_relation_subject": {
            "remote_absolute": json_values(remote),
            "third_party_relational": json_values(third_party),
        },
    }


def individualized_metrics(
    behavior: dict, rng: np.random.Generator, samples: int
) -> dict:
    """Summarize the registered eight-item individualized estimands."""

    eligible = [row for row in behavior["subjects"] if row["overall_accuracy"] >= 0.5]
    analysis = [row for row in eligible if row["ranking_class"] != "correct"]
    stable = np.asarray(
        [row["stable_error_pair_counts"]["80"] > 0 for row in analysis],
        dtype=np.float64,
    )
    if len(stable):
        stable_counts = bootstrap_counts(rng, samples, len(stable))
        stable_summary = summarize_subjects(stable, stable_counts, interval=0.95)
    else:
        stable_summary = summarize_subjects(
            np.asarray([], dtype=np.float64), np.zeros((samples, 0)), interval=0.95
        )
    orders = [row["subjective_order_high_to_low"] for row in analysis]
    if len(orders) >= 2:
        positions = []
        for order in orders:
            row = np.empty(REGISTERED_ITEM_COUNT, dtype=np.int64)
            row[np.asarray(order, dtype=np.int64)] = np.arange(REGISTERED_ITEM_COUNT)
            positions.append(row)
        positions = np.asarray(positions)
        matrix = np.eye(len(positions), dtype=np.float64)
        for first, second in combinations(range(len(positions)), 2):
            value = kendall_tau_positions(positions[first], positions[second])
            matrix[first, second] = value
            matrix[second, first] = value
        point = float(np.mean(matrix[np.triu_indices(len(positions), 1)]))
        tau_counts = bootstrap_counts(rng, samples, len(positions))
        quadratic = np.einsum(
            "bi,ij,bj->b", tau_counts, matrix, tau_counts, optimize=True
        )
        diagonal = np.sum(tau_counts, axis=1)
        draws = (quadratic - diagonal) / (len(positions) * (len(positions) - 1))
        lower, upper = np.quantile(draws, [0.025, 0.975])
        tau = {
            "subjects": len(positions),
            "mean": point,
            "bootstrap": {
                "mean": float(np.mean(draws)),
                "lower": float(lower),
                "upper": float(upper),
            },
        }
    else:
        tau = {
            "subjects": len(orders),
            "mean": None,
            "bootstrap": {"mean": None, "lower": None, "upper": None},
        }
    return {
        "eligible_subjects": len(eligible),
        "eligible_noncorrect_subjects": len(analysis),
        "stable_error_80_prevalence": stable_summary,
        "mean_pairwise_kendall_tau": tau,
    }


def serial_position_endpoint(behavior: dict, protocol: RankingProtocol) -> dict:
    """Summarize the registered eight-item serial-position endpoint estimand."""

    rank = {
        item: position for position, item in enumerate(protocol.true_order_high_to_low)
    }
    totals = np.zeros(REGISTERED_ITEM_COUNT, dtype=np.float64)
    counts = np.zeros(REGISTERED_ITEM_COUNT, dtype=np.float64)
    for row in behavior["pairs"]:
        value = float(row["mean_accuracy_all"])
        for item in row["pair"]:
            totals[rank[item]] += value
            counts[rank[item]] += 1.0
    profile = totals / counts
    interior = float(np.mean(profile[1:7]))
    return {
        "profile_high_to_low": json_values(profile),
        "interior_mean": interior,
        "mean_endpoint_contrast": float(np.mean(profile[[0, 7]]) - interior),
        "minimum_endpoint_advantage": float(min(profile[0], profile[7]) - interior),
    }


def finite_primary(metrics: dict) -> bool:
    values = []
    for condition in metrics["conditions"].values():
        for metric in condition["summary"].values():
            for group in metric.values():
                values.extend(group["bootstrap"].values())
    for row in metrics["constructive"]["summary"].values():
        values.extend(row["bootstrap"].values())
    for row in metrics["global_relation_LOO"]["summary"].values():
        values.extend(row["bootstrap"].values())
    for row in metrics["contrasts"].values():
        values.extend(row["bootstrap"].values())
    return all(value is not None and np.isfinite(value) for value in values)
