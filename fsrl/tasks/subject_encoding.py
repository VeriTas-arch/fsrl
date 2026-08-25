"""Stable subject-level bottleneck for relation evidence encoding."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class SubjectEncodingConfig:
    baseline_logit_mean: float = 1.0
    baseline_logit_sd: float = 0.5
    item_salience_sd: float = 0.35
    distance_slope_sd: float = 0.25
    minimum_reliability: float = 0.1

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class SubjectEncodingState:
    """One latent encoding state, fixed for an entire experiment episode."""

    baseline_logit: float
    item_salience: tuple[float, ...]
    distance_slope: float
    minimum_reliability: float

    def relation_reliability(
        self, first_item: int, second_item: int, symbolic_distance: int
    ) -> float:
        if first_item == second_item:
            raise ValueError("relation reliability requires two distinct items")
        n_items = len(self.item_salience)
        if not 0 <= first_item < n_items or not 0 <= second_item < n_items:
            raise ValueError("item lies outside the subject encoding state")
        if not 1 <= symbolic_distance < n_items:
            raise ValueError("symbolic_distance lies outside the ranking")
        centered_distance = symbolic_distance / float(n_items - 1) - 0.5
        logit = (
            self.baseline_logit
            + self.item_salience[first_item]
            + self.item_salience[second_item]
            + self.distance_slope * centered_distance
        )
        probability = 1.0 / (1.0 + np.exp(-logit))
        return float(
            self.minimum_reliability + (1.0 - self.minimum_reliability) * probability
        )


def sample_subject_encoding_states(
    rng: np.random.Generator,
    n_subjects: int,
    n_items: int,
    config: SubjectEncodingConfig | None = None,
) -> tuple[SubjectEncodingState, ...]:
    config = config or SubjectEncodingConfig()
    if n_subjects < 1:
        raise ValueError("n_subjects must be positive")
    if n_items < 2:
        raise ValueError("n_items must be at least two")
    if not 0.0 <= config.minimum_reliability < 1.0:
        raise ValueError("minimum_reliability must lie in [0, 1)")
    states = []
    for _ in range(n_subjects):
        states.append(
            SubjectEncodingState(
                baseline_logit=float(
                    rng.normal(config.baseline_logit_mean, config.baseline_logit_sd)
                ),
                item_salience=tuple(
                    float(value)
                    for value in rng.normal(0.0, config.item_salience_sd, n_items)
                ),
                distance_slope=float(rng.normal(0.0, config.distance_slope_sd)),
                minimum_reliability=config.minimum_reliability,
            )
        )
    return tuple(states)
