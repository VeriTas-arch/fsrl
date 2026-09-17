"""Pure estimators and decision rules for the registered attribution."""

from __future__ import annotations

from collections.abc import Mapping
from itertools import combinations

import numpy as np

from fsrl.analysis.hodge import CompleteGraphGeometry
from fsrl.tasks.protocol import RankingProtocol


def sample_pair_accuracies(
    protocol: RankingProtocol,
    subject_logits: tuple[dict[tuple[int, int], float], ...],
    *,
    seed: int,
    temperature: float,
) -> np.ndarray:
    """Replay the frozen ten-choice policy without changing its RNG stream."""

    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    pairs = tuple(combinations(range(protocol.n_items), 2))
    pair_index = {pair: index for index, pair in enumerate(pairs)}
    output = np.zeros((len(subject_logits), len(pairs)), dtype=np.float64)
    for subject, logits in enumerate(subject_logits):
        schedule_rng = np.random.default_rng(seed + 2 * subject)
        choice_rng = np.random.default_rng(seed + 2 * subject + 1)
        correct = np.zeros(len(pairs), dtype=np.float64)
        total = np.zeros(len(pairs), dtype=np.float64)
        for trial in protocol.query_schedule(schedule_rng):
            oriented = (trial.left_item, trial.right_item)
            scaled = np.clip(float(logits[oriented]) / temperature, -700.0, 700.0)
            probability_left = 1.0 / (1.0 + np.exp(-scaled))
            choose_left = bool(choice_rng.random() < probability_left)
            pair = (min(oriented), max(oriented))
            index = pair_index[pair]
            correct[index] += float(choose_left == bool(trial.correct_action))
            total[index] += 1.0
        output[subject] = correct / total
    return output


def hodge_components(
    fields: np.ndarray, geometry: CompleteGraphGeometry
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(fields, dtype=np.float64)
    gradient = values @ geometry.projection.T
    return gradient, values - gradient


def panel_stage(
    direction_present: int,
    strong_two_sided: int,
    latent_bimodal: int,
    sampled_bimodal: int,
    *,
    target: int = 15,
) -> str:
    if direction_present < target:
        return "direction_absent"
    if strong_two_sided < target:
        return "weak_margin"
    if latent_bimodal < target:
        return "latent_shape"
    if sampled_bimodal < target:
        return "finite_sampling"
    return "already_latent_and_sampled"


def study_outcome(stage_counts: Mapping[str, int], *, required: int = 40) -> str:
    winners = [name for name, count in stage_counts.items() if count >= required]
    return winners[0] if len(winners) == 1 else "mixed_or_unidentified"


def interface_checks(
    m2: Mapping[str, np.ndarray], score: Mapping[str, np.ndarray]
) -> dict[str, bool]:
    """Exact archived-array checks; semantics remain an explicit separate gate."""

    m2_pairs = np.asarray(m2["support_pairs"]).transpose(1, 0, 2)
    m2_query = np.broadcast_to(
        np.asarray(m2["query_pairs"])[None], np.asarray(score["query_pairs"]).shape
    )
    codes = np.asarray(m2["item_codes"])
    expected_support = np.concatenate(
        (
            np.take_along_axis(codes, m2_pairs[..., 0, None], axis=1),
            np.take_along_axis(codes, m2_pairs[..., 1, None], axis=1),
        ),
        axis=-1,
    ).transpose(1, 0, 2)
    score_query_pairs = np.asarray(score["query_pairs"])
    expected_query = np.concatenate(
        (
            np.take_along_axis(codes, score_query_pairs[..., 0, None], axis=1),
            np.take_along_axis(codes, score_query_pairs[..., 1, None], axis=1),
        ),
        axis=-1,
    )
    return {
        "subjects_equal": len(codes) == len(score["codes"]),
        "item_codes_equal": np.array_equal(codes, score["codes"]),
        "support_pairs_equal": np.array_equal(m2_pairs, score["support_pairs"]),
        "query_pairs_equal": np.array_equal(m2_query, score["query_pairs"]),
        "support_cues_equal": np.array_equal(expected_support, score["support_cues"]),
        "query_cues_equal": np.array_equal(expected_query, score["query_cues"]),
        "displayed_signed_equal": np.array_equal(
            m2["signed_magnitudes"], score["signed"]
        ),
        "retention_equal": np.array_equal(m2["trial_retention"], score["retention"]),
        "probabilities_equal": np.array_equal(
            m2["probabilities"], score["probabilities"]
        ),
        "realized_evidence_equal": np.array_equal(
            m2["local_evidence"], score["local_evidence"]
        ),
    }


__all__ = [
    "hodge_components",
    "interface_checks",
    "panel_stage",
    "sample_pair_accuracies",
    "study_outcome",
]
