"""Pure construction of frozen evaluator subject-encoding state."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fsrl.core.config import TrainConfig
from fsrl.tasks.protocol import RankingProtocol, SupportTrial
from fsrl.tasks.subject_encoding import (
    SubjectEncodingState,
    sample_subject_encoding_states,
)

Relation = tuple[int, int]
RelationProbabilities = list[dict[Relation, float]]

_SUPPORTED_MODES = {
    "none",
    "stable_bottleneck",
    "stable_omission",
    "presentationwise_omission",
    "blockwise_omission",
    "uniform_no_bottleneck",
}


@dataclass(frozen=True)
class FrozenSubjectEncoding:
    states: tuple[SubjectEncodingState, ...] | None
    relation_gains: tuple[dict[Relation, float], ...] | None
    trial_gains: tuple[tuple[float, ...], ...] | None


def _relation_probabilities(
    states: tuple[SubjectEncodingState, ...],
    protocol: RankingProtocol,
    item_rank: dict[int, int],
) -> RelationProbabilities:
    probabilities: RelationProbabilities = []
    for state in states:
        subject_probabilities: dict[Relation, float] = {}
        for higher, lower in protocol.support_pairs_higher_lower:
            symbolic_distance = item_rank[lower] - item_rank[higher]
            subject_probabilities[(higher, lower)] = state.relation_reliability(
                higher, lower, symbolic_distance
            )
        probabilities.append(subject_probabilities)
    return probabilities


def _relation_gains(
    probabilities: RelationProbabilities,
    mode: str,
    rng: np.random.Generator,
) -> RelationProbabilities:
    if mode == "stable_omission":
        return [
            {
                pair: float(rng.random() < probability)
                for pair, probability in subject_probabilities.items()
            }
            for subject_probabilities in probabilities
        ]
    if mode == "uniform_no_bottleneck":
        uniform_gain = float(
            np.mean(
                [
                    probability
                    for subject_probabilities in probabilities
                    for probability in subject_probabilities.values()
                ]
            )
        )
        return [
            {pair: uniform_gain for pair in subject_probabilities}
            for subject_probabilities in probabilities
        ]
    return probabilities


def _trial_gains(
    probabilities: RelationProbabilities,
    relation_gains: RelationProbabilities,
    support_schedules: tuple[tuple[SupportTrial, ...], ...],
    protocol: RankingProtocol,
    mode: str,
    rng: np.random.Generator,
) -> tuple[tuple[float, ...], ...]:
    trial_gains: list[tuple[float, ...]] = []
    for subject, schedule in enumerate(support_schedules):
        if mode == "presentationwise_omission":
            values = tuple(
                float(
                    rng.random()
                    < probabilities[subject][(trial.higher_item, trial.lower_item)]
                )
                for trial in schedule
            )
        elif mode == "blockwise_omission":
            block_relation_gains = {
                (block, pair): float(rng.random() < probabilities[subject][pair])
                for block in range(protocol.support_blocks)
                for pair in protocol.support_pairs_higher_lower
            }
            values = tuple(
                block_relation_gains[
                    (trial.block_index, (trial.higher_item, trial.lower_item))
                ]
                for trial in schedule
            )
        else:
            values = tuple(
                relation_gains[subject][(trial.higher_item, trial.lower_item)]
                for trial in schedule
            )
        trial_gains.append(values)
    return tuple(trial_gains)


def realized_support_evidence(
    support_schedules: tuple[tuple[SupportTrial, ...], ...],
    trial_gains: tuple[tuple[float, ...], ...] | None,
) -> tuple[tuple[dict[str, int | float], ...], ...]:
    """Describe the exact support evidence delivered to every subject."""

    rows: list[tuple[dict[str, int | float], ...]] = []
    for subject, schedule in enumerate(support_schedules):
        rows.append(
            tuple(
                {
                    "higher_item": trial.higher_item,
                    "lower_item": trial.lower_item,
                    "magnitude": abs(trial.signed_magnitude),
                    "reliability": (
                        1.0
                        if trial_gains is None
                        else trial_gains[subject][trial_index]
                    ),
                    "block_index": trial.block_index,
                }
                for trial_index, trial in enumerate(schedule)
            )
        )
    return tuple(rows)


def build_frozen_subject_encoding(
    config: TrainConfig,
    protocol: RankingProtocol,
    item_rank: dict[int, int],
    support_schedules: tuple[tuple[SupportTrial, ...], ...],
    *,
    mode: str,
    seed: int,
) -> FrozenSubjectEncoding:
    if mode not in _SUPPORTED_MODES:
        raise ValueError(f"unknown subject encoding mode: {mode}")
    if mode == "none":
        return FrozenSubjectEncoding(None, None, None)

    encoding_rng = np.random.default_rng(seed)
    states = sample_subject_encoding_states(
        encoding_rng,
        config.bs,
        protocol.n_items,
    )
    probabilities = _relation_probabilities(states, protocol, item_rank)
    relation_gains = _relation_gains(probabilities, mode, encoding_rng)
    trial_gains = _trial_gains(
        probabilities,
        relation_gains,
        support_schedules,
        protocol,
        mode,
        encoding_rng,
    )
    return FrozenSubjectEncoding(states, tuple(relation_gains), trial_gains)
