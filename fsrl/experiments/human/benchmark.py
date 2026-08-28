"""Recompute the Liu et al. behavioral benchmark from public trial-level data."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

from fsrl.analysis.behavioral import (
    count_circular_triads,
    fit_beta_distribution,
    hodge_rank_positions,
    kendall_tau_positions,
    maximum_circular_triads,
)
from fsrl.infra.file_contracts import dataset_file, load_dataset_manifest
from fsrl.infra.provenance import file_sha256
from fsrl.infra.study_registry import resolve_record
from fsrl.paths import EXTERNAL_DATA_ROOT, REPO_ROOT
from fsrl.tasks.registered_protocol import RankingProtocol, load_ranking_protocol

ROOT = REPO_ROOT
DEFAULT_PROTOCOL_PATH = resolve_record("benchmarks/liu_v2.json")
LIU_DATASET_ROOT = EXTERNAL_DATA_ROOT / "liu2026"
LIU_DATASET = load_dataset_manifest(LIU_DATASET_ROOT / "dataset.toml")
LIU_DATASET_FILES = {
    file_id: dataset_file(LIU_DATASET, file_id)
    for file_id in ("preregistered", "replication", "figure2d", "figure3b")
}
DEFAULT_PREREGISTERED_PATH = (
    LIU_DATASET_ROOT / LIU_DATASET_FILES["preregistered"]["path"]
)
DEFAULT_REPLICATION_PATH = LIU_DATASET_ROOT / LIU_DATASET_FILES["replication"]["path"]
DEFAULT_OUTPUT_PATH = resolve_record("benchmarks/liu_human_exact_v1.json")
DEFAULT_FIGURE2D_PATH = LIU_DATASET_ROOT / LIU_DATASET_FILES["figure2d"]["path"]
DEFAULT_FIGURE3B_PATH = LIU_DATASET_ROOT / LIU_DATASET_FILES["figure3b"]["path"]

SOURCE_FILES = {
    "preregistered": {
        "download_url": LIU_DATASET_FILES["preregistered"]["source_url"],
        "sha256": LIU_DATASET_FILES["preregistered"]["sha256"],
        "participants": LIU_DATASET_FILES["preregistered"]["participants"],
    },
    "replication": {
        "download_url": LIU_DATASET_FILES["replication"]["source_url"],
        "sha256": LIU_DATASET_FILES["replication"]["sha256"],
        "participants": LIU_DATASET_FILES["replication"]["participants"],
    },
    "figure2d": {
        "download_url": LIU_DATASET_FILES["figure2d"]["source_url"],
        "sha256": LIU_DATASET_FILES["figure2d"]["sha256"],
        "pairs": LIU_DATASET_FILES["figure2d"]["pairs"],
    },
    "figure3b": {
        "download_url": LIU_DATASET_FILES["figure3b"]["source_url"],
        "sha256": LIU_DATASET_FILES["figure3b"]["sha256"],
        "participants": LIU_DATASET_FILES["figure3b"]["participants"],
    },
}

REQUIRED_COLUMNS = {
    "id",
    "trial",
    "block",
    "film_choose_index",
    "film_index_1",
    "film_index_2",
    "r_or_w",
}


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _true_positions(protocol: RankingProtocol) -> np.ndarray:
    positions = np.empty(protocol.n_items, dtype=np.int64)
    for position, item in enumerate(protocol.true_order_high_to_low):
        positions[item] = position
    return positions


def load_human_cohort(
    path: Path,
    cohort: str,
    protocol: RankingProtocol,
    *,
    expected_sha256: str,
) -> list[dict]:
    observed_sha256 = file_sha256(path)
    if observed_sha256 != expected_sha256:
        raise ValueError(f"{cohort} source SHA-256 mismatch: {observed_sha256}")
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if set(reader.fieldnames or ()) != REQUIRED_COLUMNS:
            raise ValueError(f"{cohort} source columns do not match the contract")
        for row in reader:
            grouped[int(row["id"])].append(row)

    pairs = tuple(combinations(range(protocol.n_items), 2))
    pair_to_index = {pair: index for index, pair in enumerate(pairs)}
    true_positions = _true_positions(protocol)
    learned_mask = np.asarray([pair in protocol.learned_pairs for pair in pairs])
    subjects = []
    for source_id, rows in sorted(grouped.items()):
        if len(rows) != protocol.query_trials:
            raise ValueError(
                f"{cohort} subject {source_id} has {len(rows)} query trials"
            )
        correct_counts = np.zeros(len(pairs), dtype=np.float64)
        choose_first_counts = np.zeros(len(pairs), dtype=np.float64)
        total_counts = np.zeros(len(pairs), dtype=np.float64)
        observed_trials = set()
        for row in rows:
            trial = int(row["trial"])
            block = int(row["block"])
            first_source = int(row["film_index_1"])
            second_source = int(row["film_index_2"])
            chosen_source = int(row["film_choose_index"])
            correct = int(row["r_or_w"])
            if not (
                1 <= first_source <= protocol.n_items
                and 1 <= second_source <= protocol.n_items
                and chosen_source in {first_source, second_source}
                and correct in {0, 1}
            ):
                raise ValueError(f"invalid trial values for subject {source_id}")
            expected_correct = int(chosen_source == max(first_source, second_source))
            if correct != expected_correct:
                raise ValueError(
                    f"accuracy disagrees with A < ... < H for subject {source_id}"
                )
            first = first_source - 1
            second = second_source - 1
            chosen = chosen_source - 1
            pair = tuple(sorted((first, second)))
            pair_index = pair_to_index[pair]
            total_counts[pair_index] += 1.0
            correct_counts[pair_index] += correct
            choose_first_counts[pair_index] += int(chosen == pair[0])
            observed_trials.add((block, trial))
        if len(observed_trials) != protocol.query_trials:
            raise ValueError(f"duplicate trial identifiers for subject {source_id}")
        if set(total_counts) != {float(protocol.query_blocks)}:
            raise ValueError(
                f"subject {source_id} does not contain every pair in every block"
            )

        pair_accuracy = correct_counts / total_counts
        preference = np.zeros((protocol.n_items, protocol.n_items), dtype=np.float64)
        winners = {}
        majority_ties = 0
        for pair_index, pair in enumerate(pairs):
            first_choices = choose_first_counts[pair_index]
            preference_value = 2.0 * first_choices / total_counts[pair_index] - 1.0
            preference[pair] = preference_value
            preference[(pair[1], pair[0])] = -preference_value
            true_winner = (
                pair[0]
                if true_positions[pair[0]] < true_positions[pair[1]]
                else pair[1]
            )
            other = pair[1] if true_winner == pair[0] else pair[0]
            if pair_accuracy[pair_index] > 0.5:
                winners[pair] = true_winner
            else:
                winners[pair] = other
                majority_ties += int(pair_accuracy[pair_index] == 0.5)

        circular = count_circular_triads(winners, protocol.n_items)
        positions = hodge_rank_positions(preference)
        majority_correct = bool(np.all(pair_accuracy > 0.5))
        if majority_correct:
            ranking_class = "correct"
        elif circular == 0:
            ranking_class = "self_consistent_incorrect"
        else:
            ranking_class = "self_inconsistent"
        distance_accuracy = {}
        for distance in range(1, protocol.n_items):
            mask = np.asarray(
                [
                    abs(true_positions[first] - true_positions[second]) == distance
                    for first, second in pairs
                ]
            )
            distance_accuracy[str(distance)] = float(np.mean(pair_accuracy[mask]))
        stable_error_counts = {
            str(threshold): int(
                np.sum((1.0 - pair_accuracy) >= threshold / 100.0 - 1e-9)
            )
            for threshold in (60, 70, 80, 90, 100)
        }
        subjects.append(
            {
                "cohort": cohort,
                "source_id": source_id,
                "overall_accuracy": float(np.mean(pair_accuracy)),
                "learned_accuracy": float(np.mean(pair_accuracy[learned_mask])),
                "nonlearned_accuracy": float(np.mean(pair_accuracy[~learned_mask])),
                "pair_accuracy": pair_accuracy.tolist(),
                "ranking_class": ranking_class,
                "circular_triads": circular,
                "majority_ties": majority_ties,
                "self_consistency_coefficient": float(
                    1.0 - circular / maximum_circular_triads(protocol.n_items)
                ),
                "subjective_order_high_to_low": [
                    int(item) for item in np.argsort(positions)
                ],
                "distance_accuracy": distance_accuracy,
                "symbolic_distance_slope": float(
                    np.polyfit(
                        np.arange(1, protocol.n_items),
                        np.asarray(list(distance_accuracy.values())),
                        1,
                    )[0]
                ),
                "stable_error_pair_counts": stable_error_counts,
            }
        )
    return subjects


def fit_source_beta_distribution(values: np.ndarray, *, clip: float = 0.01) -> dict:
    """Recompute the Figure 2d MLE with an explicit endpoint convention."""

    clipped = np.clip(np.asarray(values, dtype=np.float64), clip, 1.0 - clip)
    alpha, beta, _location, _scale = stats.beta.fit(clipped, floc=0.0, fscale=1.0)
    if alpha > 1.0 and beta < 1.0:
        distribution_class = "high_accuracy"
    elif alpha < 1.0 and beta < 1.0:
        distribution_class = "bimodal"
    elif alpha > 1.0 and beta > 1.0:
        distribution_class = "ordinary_unimodal"
    elif alpha < 1.0 and beta > 1.0:
        distribution_class = "low_accuracy"
    else:
        distribution_class = "boundary"
    return {
        "alpha": float(alpha),
        "beta": float(beta),
        "class": distribution_class,
        "endpoint_clip": clip,
    }


def load_published_figure_checks(
    figure2d_path: Path, figure3b_path: Path, protocol: RankingProtocol
) -> dict:
    expected_paths = {
        figure2d_path: SOURCE_FILES["figure2d"]["sha256"],
        figure3b_path: SOURCE_FILES["figure3b"]["sha256"],
    }
    for path, expected_hash in expected_paths.items():
        if file_sha256(path) != expected_hash:
            raise ValueError(f"published figure source SHA-256 mismatch: {path}")

    distribution_name = {
        "highAccu": "high_accuracy",
        "bimodal": "bimodal",
        "unimodal": "ordinary_unimodal",
        "lowAccu": "low_accuracy",
    }
    beta_rows = []
    with figure2d_path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            first = int(row["Item1"]) - 1
            second = int(row["Item2"]) - 1
            beta_rows.append(
                {
                    "pair": [
                        protocol.item_labels[first],
                        protocol.item_labels[second],
                    ],
                    "alpha": float(row["alpha"]),
                    "beta": float(row["beta"]),
                    "class": distribution_name[row["distribution_type"]],
                }
            )
    expected_pairs = [
        [protocol.item_labels[first], protocol.item_labels[second]]
        for first, second in combinations(range(protocol.n_items), 2)
    ]
    if [row["pair"] for row in beta_rows] != expected_pairs:
        raise ValueError("Figure 2d pair order does not match the protocol")

    group_name = {
        "Correct rank": "correct",
        "Self_consistent": "self_consistent_incorrect",
        "Self_inconsistent": "self_inconsistent",
    }
    with figure3b_path.open(newline="", encoding="utf-8-sig") as handle:
        rank_classes = [group_name[row["Group"]] for row in csv.DictReader(handle)]
    if len(rank_classes) != 77:
        raise ValueError("Figure 3b does not contain the registered 77 participants")

    beta_counts = {
        name: sum(row["class"] == name for row in beta_rows)
        for name in (
            "high_accuracy",
            "bimodal",
            "ordinary_unimodal",
            "low_accuracy",
        )
    }
    rank_counts = {
        name: rank_classes.count(name)
        for name in (
            "correct",
            "self_consistent_incorrect",
            "self_inconsistent",
        )
    }
    return {
        "ranking_classes_by_combined_id": rank_classes,
        "ranking_class_counts": rank_counts,
        "beta_pair_class_counts": beta_counts,
        "beta_pairs": beta_rows,
    }


def summarize_human_subjects(protocol: RankingProtocol, subjects: list[dict]) -> dict:
    pairs = tuple(combinations(range(protocol.n_items), 2))
    pair_accuracy = np.asarray(
        [subject["pair_accuracy"] for subject in subjects], dtype=np.float64
    )
    overall = np.asarray(
        [subject["overall_accuracy"] for subject in subjects], dtype=np.float64
    )
    eligible = overall >= 0.5
    correct_rankers = np.asarray(
        [subject["ranking_class"] == "correct" for subject in subjects]
    )
    analysis = eligible & ~correct_rankers
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
    source_beta_counts = {
        name: 0
        for name in (
            "ordinary_unimodal",
            "high_accuracy",
            "low_accuracy",
            "bimodal",
            "boundary",
        )
    }
    pair_rows = []
    true_positions = _true_positions(protocol)
    for pair_index, pair in enumerate(pairs):
        fit = fit_beta_distribution(pair_accuracy[analysis, pair_index])
        beta_counts[fit["class"]] += 1
        source_fit = fit_source_beta_distribution(pair_accuracy[analysis, pair_index])
        source_beta_counts[source_fit["class"]] += 1
        pair_rows.append(
            {
                "pair": [protocol.item_labels[item] for item in pair],
                "learned": pair in protocol.learned_pairs,
                "symbolic_distance": int(
                    abs(true_positions[pair[0]] - true_positions[pair[1]])
                ),
                "mean_accuracy": float(np.mean(pair_accuracy[eligible, pair_index])),
                "beta_fit_analysis": fit,
                "beta_fit_uniform_source_reconstruction": source_fit,
            }
        )
    ranking_counts = {
        name: int(
            sum(
                eligible[index] and subject["ranking_class"] == name
                for index, subject in enumerate(subjects)
            )
        )
        for name in ("correct", "self_consistent_incorrect", "self_inconsistent")
    }
    stable = {}
    for threshold in (60, 70, 80, 90, 100):
        has_error = np.asarray(
            [
                subject["stable_error_pair_counts"][str(threshold)] > 0
                for subject in subjects
            ]
        )
        stable[str(threshold)] = {
            "eligible_count": int(np.sum(has_error & eligible)),
            "eligible_proportion": float(np.mean(has_error[eligible])),
            "analysis_count": int(np.sum(has_error & analysis)),
            "analysis_proportion": float(np.mean(has_error[analysis])),
        }
    ranks = [
        np.asarray(subject["subjective_order_high_to_low"], dtype=np.int64)
        for subject in subjects
    ]
    rank_positions = []
    for order in ranks:
        positions = np.empty(protocol.n_items, dtype=np.int64)
        positions[order] = np.arange(protocol.n_items)
        rank_positions.append(positions)
    inter_subject_tau = [
        kendall_tau_positions(rank_positions[first], rank_positions[second])
        for first, second in combinations(np.flatnonzero(analysis), 2)
    ]

    def mean(name: str, mask: np.ndarray = eligible) -> float:
        return float(np.mean([subjects[index][name] for index in np.flatnonzero(mask)]))

    return {
        "generated_subjects": len(subjects),
        "eligible_subjects": int(np.sum(eligible)),
        "excluded_below_chance": int(np.sum(~eligible)),
        "analysis_subjects_excluding_correct_rankers": int(np.sum(analysis)),
        "overall_accuracy": mean("overall_accuracy"),
        "learned_accuracy": mean("learned_accuracy"),
        "nonlearned_accuracy": mean("nonlearned_accuracy"),
        "symbolic_distance_slope": {
            "mean": mean("symbolic_distance_slope"),
            "t_vs_zero": float(
                stats.ttest_1samp(
                    [
                        subject["symbolic_distance_slope"]
                        for subject in subjects
                        if subject["overall_accuracy"] >= 0.5
                    ],
                    0.0,
                ).statistic
            ),
        },
        "ranking_class_counts": ranking_counts,
        "mean_self_consistency_coefficient": mean("self_consistency_coefficient"),
        "stable_error_subject_prevalence": stable,
        "beta_pair_class_counts_analysis": beta_counts,
        "beta_pair_class_counts_uniform_source_reconstruction": source_beta_counts,
        "mean_inter_subject_kendall_tau": float(np.mean(inter_subject_tau)),
        "pairs": pair_rows,
    }


def bootstrap_human_summary(subjects: list[dict], *, seed: int, samples: int) -> dict:
    if samples < 1:
        raise ValueError("bootstrap samples must be positive")
    rng = np.random.default_rng(seed)
    metrics = {
        "overall_accuracy": [],
        "learned_accuracy": [],
        "nonlearned_accuracy": [],
        "symbolic_distance_slope": [],
        "correct_ranker_proportion": [],
        "self_consistent_incorrect_proportion": [],
        "self_inconsistent_proportion": [],
        "stable_error_80_analysis_proportion": [],
        "stable_error_100_analysis_proportion": [],
    }
    n_subjects = len(subjects)
    for _ in range(samples):
        indices = rng.integers(0, n_subjects, size=n_subjects)
        sample = [subjects[index] for index in indices]
        eligible = [subject for subject in sample if subject["overall_accuracy"] >= 0.5]
        analysis = [
            subject for subject in eligible if subject["ranking_class"] != "correct"
        ]
        for name in (
            "overall_accuracy",
            "learned_accuracy",
            "nonlearned_accuracy",
            "symbolic_distance_slope",
        ):
            metrics[name].append(
                float(np.mean([subject[name] for subject in eligible]))
            )
        for ranking_class in (
            "correct",
            "self_consistent_incorrect",
            "self_inconsistent",
        ):
            metric_name = (
                "correct_ranker_proportion"
                if ranking_class == "correct"
                else f"{ranking_class}_proportion"
            )
            metrics[metric_name].append(
                sum(subject["ranking_class"] == ranking_class for subject in eligible)
                / len(eligible)
            )
        for threshold in (80, 100):
            metrics[f"stable_error_{threshold}_analysis_proportion"].append(
                sum(
                    subject["stable_error_pair_counts"][str(threshold)] > 0
                    for subject in analysis
                )
                / len(analysis)
            )
    return {
        "seed": seed,
        "samples": samples,
        "method": "participant-level nonparametric percentile bootstrap",
        "interval": 0.95,
        "metrics": {
            name: {
                "mean": float(np.mean(values)),
                "standard_deviation": float(np.std(values, ddof=1)),
                "lower": float(np.quantile(values, 0.025)),
                "upper": float(np.quantile(values, 0.975)),
            }
            for name, values in metrics.items()
        },
    }


def validate_paper_internal_checks(
    summary: dict, subjects: list[dict], published: dict
) -> dict:
    expected = {
        "eligible_subjects": 77,
        "analysis_subjects_excluding_correct_rankers": 69,
        "ranking_class_counts": {
            "correct": 8,
            "self_consistent_incorrect": 64,
            "self_inconsistent": 5,
        },
        "stable_80_analysis_count": 63,
        "stable_100_analysis_count": 54,
    }
    observed = {
        "eligible_subjects": summary["eligible_subjects"],
        "analysis_subjects_excluding_correct_rankers": summary[
            "analysis_subjects_excluding_correct_rankers"
        ],
        "ranking_class_counts": summary["ranking_class_counts"],
        "stable_80_analysis_count": summary["stable_error_subject_prevalence"]["80"][
            "analysis_count"
        ],
        "stable_100_analysis_count": summary["stable_error_subject_prevalence"]["100"][
            "analysis_count"
        ],
    }
    if observed != expected:
        raise RuntimeError(
            f"source reanalysis does not reproduce paper checks: {observed}"
        )
    observed_rank_classes = [
        subject["ranking_class_trial_majority"] for subject in subjects
    ]
    rank_exceptions = [
        {
            "combined_id": index + 1,
            "trial_majority_reconstruction": observed,
            "published": released,
            "majority_ties": subjects[index]["majority_ties"],
        }
        for index, (observed, released) in enumerate(
            zip(
                observed_rank_classes,
                published["ranking_classes_by_combined_id"],
                strict=True,
            )
        )
        if observed != released
    ]
    if [row["combined_id"] for row in rank_exceptions] != [30, 38]:
        raise RuntimeError(
            "Figure 3b reconstruction changed outside the registered participants"
        )
    expected_beta_counts = {
        "high_accuracy": 13,
        "bimodal": 15,
        "ordinary_unimodal": 0,
        "low_accuracy": 0,
    }
    if published["beta_pair_class_counts"] != expected_beta_counts:
        raise RuntimeError("Figure 2d does not reproduce the paper beta counts")

    uniform = summary["pairs"]
    exceptions = []
    for recomputed, released in zip(uniform, published["beta_pairs"], strict=True):
        fit = recomputed["beta_fit_uniform_source_reconstruction"]
        if not (
            np.isclose(fit["alpha"], released["alpha"], rtol=0.0, atol=1e-10)
            and np.isclose(fit["beta"], released["beta"], rtol=0.0, atol=1e-10)
        ):
            exceptions.append(
                {
                    "pair": released["pair"],
                    "uniform_clip_0.01": fit,
                    "published": released,
                }
            )
    if [row["pair"] for row in exceptions] != [["B", "H"]]:
        raise RuntimeError(
            "Figure 2d endpoint reconstruction changed outside the registered B-H exception"
        )
    return {
        "figure3b_trial_reconstruction": {
            "matching_participants": 75,
            "exception_participants": rank_exceptions,
            "interpretation": (
                "The paper does not specify majority-tie handling. Published "
                "Figure 3b classes are primary; the transparent majority-rule "
                "reconstruction differs only for IDs 30 and 38."
            ),
        },
        "figure2d_uniform_endpoint_reconstruction": {
            "matching_pairs": 27,
            "exception_pairs": exceptions,
            "interpretation": (
                "The released B-H parameters equal a 0.001/0.999 endpoint clip; "
                "the other 27 equal 0.01/0.99. Published classes remain the "
                "paper-reproduction target."
            ),
        },
    }


def build_human_benchmark(
    preregistered_path: Path,
    replication_path: Path,
    protocol_path: Path,
    *,
    bootstrap_seed: int,
    bootstrap_samples: int,
    figure2d_path: Path = DEFAULT_FIGURE2D_PATH,
    figure3b_path: Path = DEFAULT_FIGURE3B_PATH,
) -> dict:
    protocol = load_ranking_protocol(protocol_path)
    preregistered = load_human_cohort(
        preregistered_path,
        "preregistered",
        protocol,
        expected_sha256=SOURCE_FILES["preregistered"]["sha256"],
    )
    replication = load_human_cohort(
        replication_path,
        "replication",
        protocol,
        expected_sha256=SOURCE_FILES["replication"]["sha256"],
    )
    if len(preregistered) != SOURCE_FILES["preregistered"]["participants"]:
        raise RuntimeError("preregistered cohort participant count changed")
    if len(replication) != SOURCE_FILES["replication"]["participants"]:
        raise RuntimeError("replication cohort participant count changed")
    published = load_published_figure_checks(figure2d_path, figure3b_path, protocol)
    combined = preregistered + replication
    for subject, published_class in zip(
        combined,
        published["ranking_classes_by_combined_id"],
        strict=True,
    ):
        subject["ranking_class_trial_majority"] = subject["ranking_class"]
        subject["ranking_class"] = published_class
    combined_summary = summarize_human_subjects(protocol, combined)
    reproduction = validate_paper_internal_checks(combined_summary, combined, published)
    combined_summary["published_figure_checks"] = {
        "ranking_class_counts": published["ranking_class_counts"],
        "beta_pair_class_counts": published["beta_pair_class_counts"],
    }
    return {
        "schema_version": 1,
        "benchmark_id": "liu-human-exact-combined-v1",
        "protocol_id": protocol.protocol_id,
        "status": "source_recomputed_and_paper_checks_reproduced",
        "paper": {
            "title": "Human brains construct individualized global rankings from identical few-shot learning input",
            "doi": "10.1371/journal.pbio.3003756",
            "osf_project": "https://osf.io/gya95/",
        },
        "source_files": {
            name: {
                **metadata,
                "local_path": portable_path(
                    {
                        "preregistered": preregistered_path,
                        "replication": replication_path,
                        "figure2d": figure2d_path,
                        "figure3b": figure3b_path,
                    }[name]
                ),
            }
            for name, metadata in SOURCE_FILES.items()
        },
        "analysis_contract": {
            "below_chance_exclusion": "overall_accuracy < 0.5",
            "correct_pair": "pair_accuracy > 0.5",
            "correct_ranker": "all 28 pair accuracies > 0.5",
            "ranking_classification": (
                "published Figure 3b classes; trial-majority reconstruction retained "
                "as a diagnostic because tie handling is under-specified"
            ),
            "pair_distribution_cohort": "eligible subjects excluding correct rankers",
            "registered_model_beta_endpoint_clip": 0.001,
            "paper_figure_beta_endpoint_reconstruction": (
                "0.01 for 27 pairs; B-H released parameters match 0.001"
            ),
            "stable_error_thresholds": [0.6, 0.7, 0.8, 0.9, 1.0],
            "self_consistency_denominator": 20,
            "ranking_reconstruction": "Hodge least squares from mean pair preference",
        },
        "source_reproduction": reproduction,
        "combined": combined_summary,
        "cohorts": {
            "preregistered": summarize_human_subjects(protocol, preregistered),
            "replication": summarize_human_subjects(protocol, replication),
        },
        "bootstrap": bootstrap_human_summary(
            combined, seed=bootstrap_seed, samples=bootstrap_samples
        ),
    }


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Recompute the exact Liu human benchmark from OSF trial data."
    )
    parser.add_argument(
        "--preregistered", type=Path, default=DEFAULT_PREREGISTERED_PATH
    )
    parser.add_argument("--replication", type=Path, default=DEFAULT_REPLICATION_PATH)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--bootstrap-seed", type=int, default=20260823)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    return parser.parse_args(args)


def main(args=None):
    parsed = parse_args(args)
    result = build_human_benchmark(
        parsed.preregistered,
        parsed.replication,
        parsed.protocol,
        bootstrap_seed=parsed.bootstrap_seed,
        bootstrap_samples=parsed.bootstrap_samples,
    )
    parsed.output.parent.mkdir(parents=True, exist_ok=True)
    with parsed.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
