"""Registered human-phenotype analysis for frozen Liu query policies."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from itertools import combinations
from pathlib import Path
from typing import Any, cast

import numpy as np
from scipy import stats

from fsrl.evaluation.frozen_fast_weight import (
    FastWeightIntervention,
    FrozenFastWeightEvaluator,
    load_retro_checkpoint,
)
from fsrl.tasks.registered_protocol import (
    DEFAULT_PROTOCOL_PATH,
    RankingProtocol,
    load_ranking_protocol,
)

Pair = tuple[int, int]


def _sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-value)))


def count_circular_triads(winners: dict[Pair, int], n_items: int) -> int:
    cycles = 0
    for first, second, third in combinations(range(n_items), 3):
        wins = {first: 0, second: 0, third: 0}
        for pair in ((first, second), (first, third), (second, third)):
            wins[winners[pair]] += 1
        if set(wins.values()) == {1}:
            cycles += 1
    return cycles


def maximum_circular_triads(n_items: int) -> int:
    if n_items % 2:
        return (n_items**3 - n_items) // 24
    return (n_items**3 - 4 * n_items) // 24


def hodge_rank_positions(preference: np.ndarray) -> np.ndarray:
    n_items = preference.shape[0]
    rows = []
    values = []
    for first, second in combinations(range(n_items), 2):
        row = np.zeros(n_items, dtype=np.float64)
        row[first] = 1.0
        row[second] = -1.0
        rows.append(row)
        values.append(preference[first, second])
    rows.append(np.ones(n_items, dtype=np.float64))
    values.append(0.0)
    scores, *_ = np.linalg.lstsq(np.vstack(rows), np.asarray(values), rcond=None)
    order = np.argsort(-scores)
    positions = np.empty_like(order)
    positions[order] = np.arange(n_items)
    return positions


def kendall_tau_positions(first: np.ndarray, second: np.ndarray) -> float:
    concordant = 0
    discordant = 0
    for left, right in combinations(range(len(first)), 2):
        product = (first[left] - first[right]) * (second[left] - second[right])
        concordant += int(product > 0)
        discordant += int(product < 0)
    total = concordant + discordant
    return 0.0 if total == 0 else (concordant - discordant) / total


def fit_beta_distribution(values: np.ndarray) -> dict:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if len(finite) < 3:
        return {"alpha": None, "beta": None, "class": "not_fit"}
    clipped = np.clip(finite, 1e-3, 1.0 - 1e-3)
    if float(np.var(clipped)) <= 1e-12:
        return {"alpha": None, "beta": None, "class": "not_fit"}
    alpha: float
    beta: float
    try:
        fitted = cast(
            tuple[float, float, float, float],
            stats.beta.fit(clipped, floc=0.0, fscale=1.0),
        )
        alpha = float(fitted[0])
        beta = float(fitted[1])
    except (ValueError, FloatingPointError, RuntimeError):
        mean = float(np.mean(clipped))
        variance = float(np.var(clipped, ddof=1))
        if variance <= 0.0 or not 0.0 < mean < 1.0:
            return {"alpha": None, "beta": None, "class": "not_fit"}
        common = mean * (1.0 - mean) / variance - 1.0
        alpha = mean * common
        beta = (1.0 - mean) * common
    if alpha > 1.0 and beta > 1.0:
        distribution_class = "ordinary_unimodal"
    elif alpha > 1.0 and beta < 1.0:
        distribution_class = "high_accuracy"
    elif alpha < 1.0 and beta > 1.0:
        distribution_class = "low_accuracy"
    elif alpha < 1.0 and beta < 1.0:
        distribution_class = "bimodal"
    else:
        distribution_class = "boundary"
    return {
        "alpha": float(alpha),
        "beta": float(beta),
        "class": distribution_class,
    }


def _mean_subject_column(
    subjects: list[dict], name: str, mask: np.ndarray
) -> float | None:
    values = np.asarray([subject[name] for subject in subjects], dtype=np.float64)[mask]
    return None if len(values) == 0 else float(np.nanmean(values))


def analyze_sampled_query_policy(
    protocol: RankingProtocol,
    subject_logits: tuple[dict[Pair, float], ...],
    *,
    seed: int,
    temperature: float = 1.0,
) -> dict:
    """Sample the registered 280-query protocol from fixed conditional logits."""

    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    pairs = tuple(combinations(range(protocol.n_items), 2))
    pair_to_index = {pair: index for index, pair in enumerate(pairs)}
    learned = protocol.learned_pairs
    true_positions = np.empty(protocol.n_items, dtype=np.int64)
    for position, item in enumerate(protocol.true_order_high_to_low):
        true_positions[item] = position
    subjects = []
    pair_accuracies = []
    subject_ranks = []

    for subject_index, logits in enumerate(subject_logits):
        schedule_rng = np.random.default_rng(seed + 2 * subject_index)
        choice_rng = np.random.default_rng(seed + 2 * subject_index + 1)
        schedule = protocol.query_schedule(schedule_rng)
        correct_counts = np.zeros(len(pairs), dtype=np.float64)
        choose_first_counts = np.zeros(len(pairs), dtype=np.float64)
        total_counts = np.zeros(len(pairs), dtype=np.float64)
        probability_correct = []
        for trial in schedule:
            oriented_pair = (trial.left_item, trial.right_item)
            logit = logits[oriented_pair] / temperature
            probability_left = _sigmoid(logit)
            choose_left = bool(choice_rng.random() < probability_left)
            chosen_item = trial.left_item if choose_left else trial.right_item
            first, second = sorted(oriented_pair)
            pair = (first, second)
            pair_index = pair_to_index[pair]
            total_counts[pair_index] += 1.0
            correct = choose_left == bool(trial.correct_action)
            correct_counts[pair_index] += float(correct)
            choose_first_counts[pair_index] += float(chosen_item == pair[0])
            probability_correct.append(
                probability_left if trial.correct_action else 1.0 - probability_left
            )

        pair_accuracy = correct_counts / total_counts
        pair_accuracies.append(pair_accuracy)
        winners = {}
        preference = np.zeros((protocol.n_items, protocol.n_items), dtype=np.float64)
        ties = 0
        for pair_index, pair in enumerate(pairs):
            first_choices = choose_first_counts[pair_index]
            second_choices = total_counts[pair_index] - first_choices
            preference_value = (first_choices - second_choices) / total_counts[
                pair_index
            ]
            preference[pair] = preference_value
            preference[(pair[1], pair[0])] = -preference_value
            if first_choices > second_choices:
                winners[pair] = pair[0]
            elif second_choices > first_choices:
                winners[pair] = pair[1]
            else:
                ties += 1
                canonical_margin = 0.5 * (logits[pair] - logits[(pair[1], pair[0])])
                winners[pair] = pair[0] if canonical_margin > 0.0 else pair[1]

        circular = count_circular_triads(winners, protocol.n_items)
        rank_positions = hodge_rank_positions(preference)
        subject_ranks.append(rank_positions)
        majority_correct = all(
            true_positions[winners[pair]]
            < true_positions[pair[1] if winners[pair] == pair[0] else pair[0]]
            for pair in pairs
        )
        if majority_correct:
            ranking_class = "correct"
        elif circular == 0:
            ranking_class = "self_consistent_incorrect"
        else:
            ranking_class = "self_inconsistent"

        learned_mask = np.asarray([pair in learned for pair in pairs])
        distance_accuracy = {}
        for distance in range(1, protocol.n_items):
            distance_mask = np.asarray(
                [
                    abs(true_positions[a] - true_positions[b]) == distance
                    for a, b in pairs
                ]
            )
            distance_accuracy[str(distance)] = float(
                np.mean(pair_accuracy[distance_mask])
            )
        stable_errors = {}
        for threshold in (0.6, 0.7, 0.8, 0.9, 1.0):
            count = int(np.sum((1.0 - pair_accuracy) >= threshold - 1e-9))
            stable_errors[str(round(threshold * 100))] = count
        subjects.append(
            {
                "subject": subject_index,
                "overall_accuracy": float(np.mean(pair_accuracy)),
                "learned_accuracy": float(np.mean(pair_accuracy[learned_mask])),
                "nonlearned_accuracy": float(np.mean(pair_accuracy[~learned_mask])),
                "mean_probability_correct": float(np.mean(probability_correct)),
                "distance_accuracy": distance_accuracy,
                "circular_triads": circular,
                "self_consistency_coefficient": float(
                    1.0 - circular / maximum_circular_triads(protocol.n_items)
                ),
                "ranking_class": ranking_class,
                "kendall_tau_subjective_to_true": kendall_tau_positions(
                    rank_positions, true_positions
                ),
                "subjective_order_high_to_low": [
                    int(item) for item in np.argsort(rank_positions)
                ],
                "majority_ties": ties,
                "stable_error_pair_counts": stable_errors,
            }
        )

    pair_accuracies_array = np.asarray(pair_accuracies)
    subject_ranks_array = np.asarray(subject_ranks)
    overall = np.asarray(
        [subject["overall_accuracy"] for subject in subjects], dtype=np.float64
    )
    eligible = overall >= 0.5
    correct_rankers = np.asarray(
        [subject["ranking_class"] == "correct" for subject in subjects]
    )
    analysis = eligible & ~correct_rankers

    pair_rows = []
    beta_counts = {
        name: 0
        for name in (
            "ordinary_unimodal",
            "high_accuracy",
            "low_accuracy",
            "bimodal",
            "boundary",
            "not_fit",
        )
    }
    for pair_index, pair in enumerate(pairs):
        fit = fit_beta_distribution(pair_accuracies_array[analysis, pair_index])
        beta_counts[fit["class"]] += 1
        pair_rows.append(
            {
                "pair": list(pair),
                "symbolic_distance": int(
                    abs(true_positions[pair[0]] - true_positions[pair[1]])
                ),
                "learned": pair in learned,
                "mean_accuracy_all": float(
                    np.mean(pair_accuracies_array[:, pair_index])
                ),
                "mean_accuracy_analysis": (
                    None
                    if not np.any(analysis)
                    else float(np.mean(pair_accuracies_array[analysis, pair_index]))
                ),
                "beta_fit_analysis": fit,
            }
        )

    slopes = []
    for subject in subjects:
        values = np.asarray(
            [subject["distance_accuracy"][str(distance)] for distance in range(1, 8)]
        )
        slopes.append(float(np.polyfit(np.arange(1, 8), values, 1)[0]))
    slope_values = np.asarray(slopes)
    slope_test = (
        stats.ttest_1samp(slope_values, 0.0)
        if len(slope_values) > 1 and float(np.std(slope_values)) > 1e-12
        else None
    )
    typed_slope_test = cast(Any, slope_test)
    slope_statistic = None if slope_test is None else float(typed_slope_test.statistic)
    slope_pvalue = None if slope_test is None else float(typed_slope_test.pvalue)

    inter_subject_tau = []
    analysis_indices = np.flatnonzero(analysis)
    for first_index, second_index in combinations(analysis_indices, 2):
        inter_subject_tau.append(
            kendall_tau_positions(
                subject_ranks_array[first_index], subject_ranks_array[second_index]
            )
        )

    ranking_counts = {
        ranking_class: int(
            sum(
                eligible[index] and subject["ranking_class"] == ranking_class
                for index, subject in enumerate(subjects)
            )
        )
        for ranking_class in (
            "correct",
            "self_consistent_incorrect",
            "self_inconsistent",
        )
    }
    stable_error_prevalence = {}
    for threshold in (60, 70, 80, 90, 100):
        all_values = np.asarray(
            [
                subject["stable_error_pair_counts"][str(threshold)] > 0
                for subject in subjects
            ]
        )
        stable_error_prevalence[str(threshold)] = {
            "eligible": (
                float(np.mean(all_values[eligible])) if np.any(eligible) else None
            ),
            "analysis": (
                float(np.mean(all_values[analysis])) if np.any(analysis) else None
            ),
        }

    summary = {
        "generated_subjects": len(subjects),
        "eligible_subjects": int(np.sum(eligible)),
        "excluded_below_chance": int(np.sum(~eligible)),
        "analysis_subjects_excluding_correct_rankers": int(np.sum(analysis)),
        "overall_accuracy": _mean_subject_column(
            subjects, "overall_accuracy", eligible
        ),
        "learned_accuracy": _mean_subject_column(
            subjects, "learned_accuracy", eligible
        ),
        "nonlearned_accuracy": _mean_subject_column(
            subjects, "nonlearned_accuracy", eligible
        ),
        "mean_probability_correct": _mean_subject_column(
            subjects, "mean_probability_correct", eligible
        ),
        "mean_self_consistency_coefficient": _mean_subject_column(
            subjects, "self_consistency_coefficient", eligible
        ),
        "ranking_class_counts": ranking_counts,
        "stable_error_subject_prevalence": stable_error_prevalence,
        "symbolic_distance_slope": {
            "mean": float(np.mean(slopes)),
            "t_vs_zero": slope_statistic,
            "p_vs_zero": slope_pvalue,
        },
        "beta_pair_class_counts_analysis": beta_counts,
        "mean_inter_subject_kendall_tau": (
            None if not inter_subject_tau else float(np.mean(inter_subject_tau))
        ),
    }
    return {
        "protocol_id": protocol.protocol_id,
        "sampling": {
            "seed": seed,
            "temperature": temperature,
            "test_blocks": protocol.query_blocks,
            "trials_per_subject": protocol.query_trials,
            "query_fast_weights": "frozen",
            "query_hidden_and_eligibility": "reset_each_trial",
        },
        "summary": summary,
        "subjects": subjects,
        "pairs": pair_rows,
    }


def run_behavioral_analysis(
    checkpoint: Path,
    *,
    batch_size: int,
    cue_seed: int,
    support_seed: int,
    subject_encoding_seed: int,
    choice_seed: int,
    temperature: float,
    subject_encoding_mode: str = "stable_omission",
    protocol_path: Path | str = DEFAULT_PROTOCOL_PATH,
) -> dict:
    protocol_path = Path(protocol_path)
    protocol = load_ranking_protocol(protocol_path)
    net, config, checkpoint_info = load_retro_checkpoint(checkpoint, batch_size)
    evaluator = FrozenFastWeightEvaluator(
        net,
        config,
        protocol,
        cue_seed=cue_seed,
        support_seed=support_seed,
        cue_mode="permuted_shared",
        subject_encoding_mode=subject_encoding_mode,
        subject_encoding_seed=subject_encoding_seed,
    )
    fast_weights = evaluator.learn_fast_weights(FastWeightIntervention.INTACT)
    ordered_pairs = tuple(
        oriented
        for first, second in combinations(range(protocol.n_items), 2)
        for oriented in ((first, second), (second, first))
    )
    logits = evaluator.readout_logits(
        fast_weights, tuple(ordered_pairs for _ in range(batch_size))
    )
    result = analyze_sampled_query_policy(
        protocol, logits, seed=choice_seed, temperature=temperature
    )
    result["checkpoint"] = asdict(checkpoint_info)
    result["cue_seed"] = cue_seed
    result["support_seed"] = support_seed
    result["subject_encoding_seed"] = subject_encoding_seed
    result["subject_encoding_mode"] = subject_encoding_mode
    result["protocol_path"] = str(protocol_path.resolve())
    return result


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Sample and analyze the registered 280-trial Liu behavior."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=77)
    parser.add_argument("--cue-seed", type=int, default=1)
    parser.add_argument("--support-seed", type=int, default=100)
    parser.add_argument("--subject-encoding-seed", type=int, default=300)
    parser.add_argument("--choice-seed", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--subject-encoding",
        choices=[
            "stable_bottleneck",
            "stable_omission",
            "presentationwise_omission",
            "blockwise_omission",
            "uniform_no_bottleneck",
        ],
        default="stable_omission",
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    return parser.parse_args(args)


def main(args=None):
    parsed = parse_args(args)
    result = run_behavioral_analysis(
        parsed.checkpoint,
        batch_size=parsed.batch_size,
        cue_seed=parsed.cue_seed,
        support_seed=parsed.support_seed,
        subject_encoding_seed=parsed.subject_encoding_seed,
        choice_seed=parsed.choice_seed,
        temperature=parsed.temperature,
        subject_encoding_mode=parsed.subject_encoding,
        protocol_path=parsed.protocol,
    )
    parsed.output.parent.mkdir(parents=True, exist_ok=True)
    with parsed.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
