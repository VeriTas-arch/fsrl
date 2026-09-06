"""Count-only common observation, and an explicit legacy tie bridge."""

from itertools import combinations
from typing import Any, cast

import numpy as np
from scipy import stats

from fsrl.analysis.behavioral import (
    fit_beta_distribution,
    hodge_rank_positions,
    kendall_tau_positions,
)
from fsrl.experiments.cohort_diagnostic.statistics import cohort_record

CLASSES = ("correct", "self_consistent_incorrect", "self_inconsistent")


def pair_layout(protocol):
    pairs = np.asarray(list(combinations(range(protocol.n_items), 2)))
    positions = np.empty(protocol.n_items, dtype=int)
    positions[np.asarray(protocol.true_order_high_to_low)] = np.arange(protocol.n_items)
    return pairs, positions


def cycle_counts(first_wins: np.ndarray, n_items: int = 8) -> np.ndarray:
    indices = {pair: i for i, pair in enumerate(combinations(range(n_items), 2))}
    cycles = []
    for a, b, c in combinations(range(n_items), 3):
        ab, ac, bc = (
            first_wins[..., indices[pair]] for pair in ((a, b), (a, c), (b, c))
        )
        cycles.append((ab & bc & ~ac) | (~ab & ~bc & ac))
    return np.stack(cycles, axis=-1)


def replicated_cycles(choices: np.ndarray) -> np.ndarray:
    """The same directed strict-majority cycle in the two five-block halves."""
    if choices.shape[1] != 10:
        raise ValueError("the registered cycle assay requires ten query blocks")
    first, last = choices[:, :5].sum(axis=1) > 2, choices[:, 5:].sum(axis=1) > 2
    indices = {pair: i for i, pair in enumerate(combinations(range(8), 2))}
    repeated = []
    for triad in combinations(range(8), 3):
        edges = [indices[pair] for pair in combinations(triad, 2)]
        same = np.all(first[:, edges] == last[:, edges], axis=1)
        ab, ac, bc = first[:, edges].T
        repeated.append(same & ((ab & bc & ~ac) | (~ab & ~bc & ac)))
    return np.stack(repeated, axis=-1).sum(axis=-1)


def observe(choices: np.ndarray, protocol, *, legacy_margins=None) -> dict:
    """All inputs are recorded choices, except the optional named legacy bridge."""
    choices = np.asarray(choices, dtype=bool)
    pairs, positions = pair_layout(protocol)
    if choices.ndim != 3 or choices.shape[2] != len(pairs):
        raise ValueError("choices must have subject, block, unordered-pair axes")
    preference = choices.mean(axis=1)
    true_first = positions[pairs[:, 0]] < positions[pairs[:, 1]]
    accuracy = np.where(true_first, preference, 1 - preference)
    winners = preference > 0.5
    ties = preference == 0.5
    if legacy_margins is None:
        winners = np.where(ties, ~true_first, winners)
    else:
        margins = np.asarray(legacy_margins)
        if margins.shape != preference.shape:
            raise ValueError("legacy canonical margins have wrong shape")
        winners = np.where(ties, margins > 0, winners)
    circular = cycle_counts(winners).sum(axis=1)
    correct = np.all(winners == true_first, axis=1)
    classes = np.where(correct, 0, np.where(circular == 0, 1, 2))
    learned = np.asarray([tuple(pair) in protocol.learned_pairs for pair in pairs])
    distances = np.abs(positions[pairs[:, 0]] - positions[pairs[:, 1]])
    distance_means = np.stack(
        [accuracy[:, distances == d].mean(axis=1) for d in range(1, 8)], axis=1
    )
    slopes = np.polyfit(np.arange(1, 8), distance_means.T, 1)[0]
    ranks, subjects = [], []
    for s in range(len(choices)):
        matrix = np.zeros((8, 8))
        counts = choices[s].sum(axis=0)
        matrix[pairs[:, 0], pairs[:, 1]] = (
            2 * counts - choices.shape[1]
        ) / choices.shape[1]
        rank = hodge_rank_positions(matrix - matrix.T)
        ranks.append(rank)
        subjects.append(
            {
                "subject": s,
                "overall_accuracy": float(accuracy[s].mean()),
                "learned_accuracy": float(accuracy[s, learned].mean()),
                "nonlearned_accuracy": float(accuracy[s, ~learned].mean()),
                "ranking_class": CLASSES[int(classes[s])],
                "symbolic_distance_slope": float(slopes[s]),
                "pair_accuracy": accuracy[s].tolist(),
                "circular_triads": int(circular[s]),
                "majority_ties": int(ties[s].sum()),
                "subjective_order_high_to_low": np.argsort(rank).tolist(),
                "stable_error_pair_counts": {
                    str(t): int(np.sum(1 - accuracy[s] >= t / 100 - 1e-9))
                    for t in (60, 70, 80, 90, 100)
                },
            }
        )
    eligible = accuracy.mean(axis=1) >= 0.5
    analysis = eligible & (classes != 0)
    beta_counts = dict.fromkeys(
        (
            "ordinary_unimodal",
            "high_accuracy",
            "low_accuracy",
            "bimodal",
            "boundary",
            "not_fit",
        ),
        0,
    )
    rows = []
    for i, pair in enumerate(pairs):
        fit = fit_beta_distribution(accuracy[analysis, i])
        beta_counts[fit["class"]] += 1
        rows.append(
            {
                "pair": pair.tolist(),
                "symbolic_distance": int(distances[i]),
                "learned": bool(learned[i]),
                "mean_accuracy_all": float(accuracy[:, i].mean()),
                "beta_fit_analysis": fit,
            }
        )
    tau_values = [
        kendall_tau_positions(ranks[a], ranks[b])
        for a, b in combinations(np.flatnonzero(analysis), 2)
    ]
    test = (
        cast(Any, stats.ttest_1samp(slopes, 0))
        if len(slopes) > 1 and np.std(slopes) > 1e-12
        else None
    )
    stable = {
        str(t): {
            name: float(
                np.mean(
                    [
                        subjects[s]["stable_error_pair_counts"][str(t)] > 0
                        for s in np.flatnonzero(mask)
                    ]
                )
            )
            if mask.any()
            else None
            for name, mask in (("eligible", eligible), ("analysis", analysis))
        }
        for t in (60, 70, 80, 90, 100)
    }
    return {
        "subjects": subjects,
        "pairs": rows,
        "summary": {
            "generated_subjects": len(subjects),
            "eligible_subjects": int(eligible.sum()),
            "analysis_subjects_excluding_correct_rankers": int(analysis.sum()),
            "ranking_class_counts": {
                name: int(np.sum(eligible & (classes == i)))
                for i, name in enumerate(CLASSES)
            },
            "stable_error_subject_prevalence": stable,
            "symbolic_distance_slope": {
                "mean": float(slopes.mean()),
                "t_vs_zero": float(test.statistic) if test is not None else None,
                "p_vs_zero": float(test.pvalue) if test is not None else None,
            },
            "beta_pair_class_counts_analysis": beta_counts,
            "mean_inter_subject_kendall_tau": float(np.mean(tau_values))
            if tau_values
            else None,
        },
    }


def sample_choices(margins: np.ndarray, protocol, seed: int, temperature=0.25) -> tuple:
    """Replay the original query schedule/RNG while retaining blockwise choices."""
    from scipy.special import expit

    from fsrl.tasks.protocol import ordered_pairs

    pairs, _ = pair_layout(protocol)
    lookup = {tuple(pair): index for index, pair in enumerate(pairs)}
    oriented = {pair: i for i, pair in enumerate(ordered_pairs(8))}
    choices = np.empty((len(margins), protocol.query_blocks, len(pairs)), dtype=bool)
    for s in range(len(margins)):
        schedule = protocol.query_schedule(np.random.default_rng(seed + 2 * s))
        rng = np.random.default_rng(seed + 2 * s + 1)
        counts = np.zeros(len(pairs), dtype=int)
        for trial in schedule:
            pair = (trial.left_item, trial.right_item)
            left = rng.random() < expit(margins[s, oriented[pair]] / temperature)
            index = lookup[tuple(sorted(pair))]
            choices[s, counts[index], index] = left if pair[0] < pair[1] else not left
            counts[index] += 1
    canonical = np.stack(
        [
            0.5
            * (margins[:, oriented[tuple(p)]] - margins[:, oriented[tuple(p[::-1])]])
            for p in pairs
        ],
        axis=1,
    )
    return choices, canonical


def record(choices, protocol, references, *, legacy_margins=None):
    return cohort_record(
        observe(choices, protocol, legacy_margins=legacy_margins), references
    )
