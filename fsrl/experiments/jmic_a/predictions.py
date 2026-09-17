"""Task identities, readout predictions, and compact morphology summaries."""

from __future__ import annotations

from itertools import combinations
from typing import Any, Literal, TypedDict, cast

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from scipy.special import expit

from fsrl.analysis.behavioral import fit_beta_distribution
from fsrl.infra.provenance import load_json

from .model import (
    Posterior,
    amplitude_matched_equal_spacing,
    bivariate_sigmoid_moment,
    incidence,
    normal_sigmoid_moment,
    normal_threshold_probability,
    pair_matrix,
    posterior_7d,
    posterior_pair_distribution,
)
from .protocol import TASK_INPUT

ROLES = tuple("ABCDEFGH")
ROLE_INDEX = {role: index for index, role in enumerate(ROLES)}
PAIRS, PAIR_DESIGN = pair_matrix()
PAIR_NAMES = tuple(f"{ROLES[a]}-{ROLES[b]}" for a, b in PAIRS)
PAIR_INDEX = {name: index for index, name in enumerate(PAIR_NAMES)}


class TaskCondition(TypedDict):
    levels: NDArray[np.float64]
    observations: NDArray[np.float64]
    order_low_to_high: list[str]


class TaskInputs(TypedDict):
    design: NDArray[np.float64]
    A: TaskCondition
    B: TaskCondition


def task_conditions() -> tuple[dict[str, Any], TaskInputs]:
    protocol = load_json(TASK_INPUT)
    frozen = protocol["inherited_frozen_contract"]
    expected = {
        "support_relation_order": [
            "F>A",
            "C>B",
            "E>B",
            "G>C",
            "F>D",
            "G>D",
            "H>E",
            "H>A",
        ],
        "assignment_A_gaps": [5, 1, 3, 4, 2, 3, 3, 7],
        "assignment_B_gaps": [2, 3, 1, 3, 5, 7, 4, 3],
    }
    for key, value in expected.items():
        if frozen[key] != value:
            raise RuntimeError(f"frozen magnitude-placement input differs: {key}")
    edges = tuple(
        (ROLE_INDEX[higher], ROLE_INDEX[lower])
        for higher, lower in (
            relation.split(">") for relation in frozen["support_relation_order"]
        )
    )
    design = incidence(edges)
    conditions: dict[str, TaskCondition] = {}
    for name in ("A", "B"):
        order = frozen[f"assignment_{name}_low_to_high"]
        levels = np.empty(8, dtype=np.float64)
        for position, role in enumerate(order):
            levels[ROLE_INDEX[role]] = position / 7.0
        levels -= np.mean(levels)
        gaps = np.asarray(frozen[f"assignment_{name}_gaps"], dtype=np.float64) / 7.0
        if not np.allclose(design @ levels, gaps, atol=1e-15, rtol=0.0):
            raise RuntimeError(f"condition {name} displayed gaps are inconsistent")
        conditions[name] = {
            "levels": levels,
            "observations": gaps,
            "order_low_to_high": order,
        }
    return protocol, TaskInputs(
        design=design,
        A=conditions["A"],
        B=conditions["B"],
    )


def condition_posterior(
    name: Literal["A", "B"], sigma: float, tau2: float
) -> Posterior:
    _, task = task_conditions()
    return posterior_7d(
        task["design"], task[name]["observations"], sigma=sigma, tau2=tau2
    )


def analytic_readout(posterior: Posterior, theta: float, nodes: int) -> dict:
    _, means, covariance = posterior_pair_distribution(posterior)
    variance = np.diag(covariance)
    marginal = normal_sigmoid_moment(means, variance, theta, nodes=nodes)
    joint = bivariate_sigmoid_moment(means, covariance, theta, nodes=nodes)
    second = np.diag(joint)
    return {
        "pair_mean": means,
        "pair_covariance": covariance,
        "posterior_mean_probability": expit(means / theta),
        "marginal_probability": marginal,
        "persistent_second_moment": second,
        "persistent_probability_covariance": joint - np.outer(marginal, marginal),
        "latent_reversal_probability": normal_threshold_probability(
            means, variance, 0.0
        ),
        "first_item_strong_loss_probability": normal_threshold_probability(
            means, variance, -theta * np.log(4.0)
        ),
        "persistent_all_first_loss_probability": normal_sigmoid_moment(
            -means, variance, theta, power=10, nodes=nodes
        ),
        "trialwise_all_first_loss_probability": (1.0 - marginal) ** 10,
    }


def sampled_readouts(
    posterior: Posterior,
    theta: float,
    *,
    subjects: int,
    equal_spacing_draws: int,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    z = rng.multivariate_normal(
        posterior.mean_z, posterior.covariance_z, size=equal_spacing_draws
    )
    values = z @ posterior.basis.T
    continuous_probability = expit((values @ PAIR_DESIGN.T) / theta)
    equal_values = amplitude_matched_equal_spacing(values)
    equal_probability = expit((equal_values @ PAIR_DESIGN.T) / theta)
    if subjects > equal_spacing_draws:
        raise ValueError("prediction subjects exceed fixed draws")
    return {
        "persistent_continuous": continuous_probability[:subjects],
        "persistent_equal_spacing": equal_probability[:subjects],
        "equal_spacing_mean": np.mean(equal_probability, axis=0),
        "equal_spacing_second": np.mean(equal_probability**2, axis=0),
        "equal_spacing_covariance": np.cov(equal_probability, rowvar=False, ddof=0),
        "equal_spacing_mean_mcse": np.std(equal_probability, axis=0, ddof=1)
        / np.sqrt(equal_spacing_draws),
    }


def _mean_pairwise_tau(orders: np.ndarray) -> float | None:
    subjects = len(orders)
    if subjects < 2:
        return None
    denominator = subjects * (subjects - 1) / 2
    values = []
    for first, second in PAIRS:
        first_before = orders[:, first] < orders[:, second]
        count = int(np.sum(first_before))
        concordant = (
            count * (count - 1) / 2 + (subjects - count) * (subjects - count - 1) / 2
        )
        discordant = count * (subjects - count)
        values.append((concordant - discordant) / denominator)
    return float(np.mean(values))


def morphology_summary(
    probability_first: np.ndarray,
    true_levels: np.ndarray,
    learned_pair_names: list[str],
    *,
    seed: int,
    repetitions: int = 10,
) -> dict:
    probability_first = np.asarray(probability_first, dtype=np.float64)
    rng = np.random.default_rng(seed)
    first_counts = rng.binomial(repetitions, probability_first)
    true_first = np.asarray(
        [true_levels[first] > true_levels[second] for first, second in PAIRS]
    )
    correct_probability = np.where(
        true_first[None], probability_first, 1.0 - probability_first
    )
    correct_counts = np.where(
        true_first[None], first_counts, repetitions - first_counts
    )
    accuracy = correct_counts / repetitions
    overall = np.mean(accuracy, axis=1)
    learned = np.asarray([name in learned_pair_names for name in PAIR_NAMES])

    first_wins = first_counts > repetitions / 2
    ties = first_counts == repetitions / 2
    first_wins[ties] = probability_first[ties] > 0.5
    winners = np.where(
        first_wins[..., None], PAIR_DESIGN[None] > 0, PAIR_DESIGN[None] < 0
    )
    circular = np.zeros(len(probability_first), dtype=np.int64)
    for a, b, c in combinations(range(8), 3):
        ab = winners[:, PAIRS.index((a, b)), a]
        ac = winners[:, PAIRS.index((a, c)), a]
        bc = winners[:, PAIRS.index((b, c)), b]
        circular += (ab & ~ac & bc) | (~ab & ac & ~bc)
    majority_correct = np.all(first_wins == true_first[None], axis=1)
    consistent = circular == 0
    eligible = overall >= 0.5
    analysis = eligible & ~majority_correct

    preference = (2.0 * first_counts / repetitions) - 1.0
    item_scores = preference @ PAIR_DESIGN
    order = np.argsort(np.argsort(-item_scores, axis=1), axis=1)
    tau = _mean_pairwise_tau(order[analysis])

    latent_classes = []
    sampled_classes = []
    for pair_index in range(len(PAIRS)):
        latent_classes.append(
            fit_beta_distribution(correct_probability[analysis, pair_index])["class"]
        )
        sampled_classes.append(
            fit_beta_distribution(accuracy[analysis, pair_index])["class"]
        )
    class_names = (
        "ordinary_unimodal",
        "high_accuracy",
        "low_accuracy",
        "bimodal",
        "boundary",
        "not_fit",
    )
    latent_counts = {name: latent_classes.count(name) for name in class_names}
    sampled_counts = {name: sampled_classes.count(name) for name in class_names}

    positions = np.argsort(np.argsort(true_levels))
    distances = np.asarray(
        [abs(positions[a] - positions[b]) for a, b in PAIRS], dtype=np.int64
    )
    subject_distance = np.column_stack(
        [
            np.mean(accuracy[:, distances == distance], axis=1)
            for distance in range(1, 8)
        ]
    )
    slopes = np.polyfit(np.arange(1, 8), subject_distance.T, 1)[0]
    slope_test = cast(Any, stats.ttest_1samp(slopes, 0.0))
    slope_pvalue = float(slope_test[1])
    item_accuracy = np.column_stack(
        [
            np.mean(accuracy[:, np.abs(PAIR_DESIGN[:, item]) > 0], axis=1)
            for item in range(8)
        ]
    )
    by_position = np.empty((len(probability_first), 8), dtype=np.float64)
    by_position[:, positions] = item_accuracy
    serial = bool(
        np.mean(by_position[:, 0]) > np.mean(by_position[:, 1:-1])
        and np.mean(by_position[:, -1]) > np.mean(by_position[:, 1:-1])
    )

    ranking_counts = {
        "correct": int(np.sum(eligible & majority_correct)),
        "self_consistent_incorrect": int(
            np.sum(eligible & ~majority_correct & consistent)
        ),
        "self_inconsistent": int(np.sum(eligible & ~majority_correct & ~consistent)),
    }
    analysis_count = int(np.sum(analysis))
    stable80 = (
        None
        if not analysis_count
        else float(np.mean(np.any(correct_counts[analysis] <= 2, axis=1)))
    )
    stable100 = (
        None
        if not analysis_count
        else float(np.mean(np.any(correct_counts[analysis] == 0, axis=1)))
    )
    flags = {
        "learned_accuracy": float(np.mean(accuracy[:, learned])) > 0.5,
        "nonlearned_accuracy": float(np.mean(accuracy[:, ~learned])) > 0.5,
        "symbolic_distance_effect": float(np.mean(slopes)) > 0.0
        and slope_pvalue < 0.05,
        "serial_position_effect": serial,
        "difficult_pair_bimodality": latent_counts["bimodal"] >= 15
        and latent_counts["ordinary_unimodal"] == 0
        and latent_counts["low_accuracy"] == 0,
        "stable_within_subject_errors": stable80 is not None and stable80 >= 0.8,
        "self_consistent_vs_inconsistent_errors": ranking_counts[
            "self_consistent_incorrect"
        ]
        > ranking_counts["self_inconsistent"],
        "hodge_reconstructed_subjective_ranking": ranking_counts["correct"]
        < 0.5 * int(np.sum(eligible)),
        "inter_subject_ranking_diversity": tau is not None and tau < 0.8,
    }
    direction = sum(
        np.any(correct_probability[analysis, index] < 0.5)
        and np.any(correct_probability[analysis, index] > 0.5)
        for index in range(len(PAIRS))
    )
    strong = sum(
        np.any(correct_probability[analysis, index] <= 0.2)
        and np.any(correct_probability[analysis, index] >= 0.8)
        for index in range(len(PAIRS))
    )
    bad_pair = any(
        name in {"ordinary_unimodal", "low_accuracy"} for name in sampled_classes
    )
    all_nine = all(flags.values())
    return {
        "subjects": len(probability_first),
        "eligible_subjects": int(np.sum(eligible)),
        "analysis_subjects": analysis_count,
        "competence": {
            "overall_accuracy": float(np.mean(accuracy)),
            "learned_accuracy": float(np.mean(accuracy[:, learned])),
            "nonlearned_accuracy": float(np.mean(accuracy[:, ~learned])),
        },
        "ranking_class_counts": ranking_counts,
        "mean_self_consistency": float(np.mean(1.0 - circular / 16.0)),
        "mean_inter_subject_kendall_tau": tau,
        "stable_error_80": stable80,
        "stable_error_100": stable100,
        "symbolic_distance_slope": {
            "mean": float(np.mean(slopes)),
            "p_vs_zero": slope_pvalue,
        },
        "latent_beta_classes": latent_counts,
        "sampled_beta_classes": sampled_counts,
        "direction_present_pairs": int(direction),
        "strong_two_sided_pairs": int(strong),
        "historical_nine_qualitative": flags,
        "all_nine_qualitative": all_nine,
        "constrained_without_evidence_binding": bool(
            all_nine
            and latent_counts["bimodal"] >= 15
            and sampled_counts["bimodal"] >= 15
            and not bad_pair
        ),
        "compatibility_boundary": (
            "Uses frozen qualitative definitions without human calibration intervals "
            "or an evidence-binding intervention; descriptive only."
        ),
    }


def pair_field(probability_first: np.ndarray) -> dict[str, float]:
    return {
        pair: float(probability_first[index]) for index, pair in enumerate(PAIR_NAMES)
    }


def placement_estimands(
    field_a: np.ndarray, field_b: np.ndarray, protocol: dict
) -> dict[str, float]:
    inherited = protocol["inherited_frozen_contract"]
    positions = {
        name: {
            role: index
            for index, role in enumerate(inherited[f"assignment_{name}_low_to_high"])
        }
        for name in ("A", "B")
    }

    def choice(field: np.ndarray, pair: str, role: str) -> float:
        first, _second = pair.split("-")
        value = float(field[PAIR_INDEX[pair]])
        return value if role == first else 1.0 - value

    flip = []
    for pair in inherited["pair_partition"]["nonlearned_order_flip"]:
        first, second = pair.split("-")
        higher = first if positions["A"][first] > positions["A"][second] else second
        flip.append(choice(field_a, pair, higher) - choice(field_b, pair, higher))

    confidence_x = []
    confidence_y = []
    for pair in inherited["pair_partition"]["nonlearned_same_direction"]:
        first, second = pair.split("-")
        higher = first if positions["A"][first] > positions["A"][second] else second
        confidence_x.append(
            abs(positions["B"][first] - positions["B"][second])
            - abs(positions["A"][first] - positions["A"][second])
        )
        confidence_y.append(
            choice(field_b, pair, higher) - choice(field_a, pair, higher)
        )

    learned_x = []
    learned_y = []
    gaps = {
        name: dict(
            zip(
                inherited["support_relation_order"],
                inherited[f"assignment_{name}_gaps"],
                strict=True,
            )
        )
        for name in ("A", "B")
    }
    for relation in inherited["support_relation_order"]:
        higher, lower = relation.split(">")
        pair = "-".join(sorted((higher, lower), key=ROLE_INDEX.__getitem__))
        learned_x.append(gaps["B"][relation] - gaps["A"][relation])
        learned_y.append(choice(field_b, pair, higher) - choice(field_a, pair, higher))

    def slope(x, y) -> float:
        design = np.column_stack((np.ones(len(x)), np.asarray(x, dtype=np.float64)))
        return float(np.linalg.lstsq(design, np.asarray(y), rcond=None)[0][1])

    return {
        "delta_flip": float(np.mean(flip)),
        "beta_conf": slope(confidence_x, confidence_y),
        "beta_learned": slope(learned_x, learned_y),
    }


__all__ = [
    "PAIRS",
    "PAIR_DESIGN",
    "PAIR_NAMES",
    "ROLES",
    "analytic_readout",
    "condition_posterior",
    "morphology_summary",
    "pair_field",
    "placement_estimands",
    "sampled_readouts",
    "task_conditions",
]
