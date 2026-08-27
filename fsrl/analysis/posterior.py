"""Exact posterior over global ranking hypotheses for eight-item tasks."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from itertools import permutations

import numpy as np


@dataclass(frozen=True)
class RelationEvidence:
    higher_item: int
    lower_item: int
    magnitude: float
    reliability: float = 1.0


@dataclass(frozen=True)
class RankingPosteriorState:
    probabilities: np.ndarray
    energy: np.ndarray
    map_index: int


@cache
def _ranking_hypothesis_space(n_items: int) -> tuple[np.ndarray, np.ndarray]:
    orders = np.asarray(list(permutations(range(n_items))), dtype=np.int16)
    positions = np.empty_like(orders)
    positions[np.arange(len(orders))[:, None], orders] = np.arange(
        n_items, dtype=np.int16
    )
    orders.flags.writeable = False
    positions.flags.writeable = False
    return orders, positions


class ExactRankingPosterior:
    """Enumerate all orders and infer a posterior from support evidence only."""

    def __init__(self, n_items: int = 8, *, temperature: float = 0.05) -> None:
        if n_items < 2:
            raise ValueError("n_items must be at least two")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        self.n_items = int(n_items)
        self.temperature = float(temperature)
        self.orders, self.positions = _ranking_hypothesis_space(self.n_items)
        self._predicted_magnitudes: dict[tuple[int, int], np.ndarray] = {}
        self._pair_masks: dict[tuple[int, int], np.ndarray] = {}

    @property
    def n_hypotheses(self) -> int:
        return len(self.orders)

    def fit(
        self,
        evidence: tuple[RelationEvidence, ...] | list[RelationEvidence],
        *,
        prior_log_probabilities: np.ndarray | None = None,
    ) -> RankingPosteriorState:
        energy = np.zeros(self.n_hypotheses, dtype=np.float64)
        for observation in evidence:
            self._validate_evidence(observation)
            pair = (observation.higher_item, observation.lower_item)
            predicted = self._predicted_magnitudes.get(pair)
            if predicted is None:
                predicted = (
                    self.positions[:, observation.lower_item]
                    - self.positions[:, observation.higher_item]
                ) / float(self.n_items - 1)
                predicted.flags.writeable = False
                self._predicted_magnitudes[pair] = predicted
            residual = predicted - observation.magnitude
            energy += observation.reliability * residual * residual

        log_weights = -energy / self.temperature
        if prior_log_probabilities is not None:
            prior = np.asarray(prior_log_probabilities, dtype=np.float64)
            if prior.shape != (self.n_hypotheses,):
                raise ValueError("prior_log_probabilities has the wrong shape")
            if not np.all(np.isfinite(prior)):
                raise ValueError("prior_log_probabilities must be finite")
            log_weights = log_weights + prior
        log_weights -= np.max(log_weights)
        weights = np.exp(log_weights)
        probabilities = weights / np.sum(weights)
        return RankingPosteriorState(
            probabilities=probabilities,
            energy=energy,
            map_index=int(np.argmax(probabilities)),
        )

    def map_order(self, state: RankingPosteriorState) -> tuple[int, ...]:
        self._validate_state(state)
        return tuple(int(item) for item in self.orders[state.map_index])

    def order_index(self, order_high_to_low: tuple[int, ...] | list[int]) -> int:
        """Return the lexicographic hypothesis index without building a lookup."""

        order = tuple(int(item) for item in order_high_to_low)
        if len(order) != self.n_items or set(order) != set(range(self.n_items)):
            raise ValueError("order must contain every item exactly once")
        remaining = list(range(self.n_items))
        index = 0
        for item in order:
            offset = remaining.index(item)
            index = index * len(remaining) + offset
            remaining.pop(offset)
        return index

    def sample_order(
        self, state: RankingPosteriorState, rng: np.random.Generator
    ) -> tuple[int, ...]:
        self._validate_state(state)
        index = int(rng.choice(self.n_hypotheses, p=state.probabilities))
        return tuple(int(item) for item in self.orders[index])

    def pair_probability(
        self, state: RankingPosteriorState, left_item: int, right_item: int
    ) -> float:
        """Return posterior P(left ranks higher than right)."""

        self._validate_state(state)
        self._validate_pair(left_item, right_item)
        pair = (left_item, right_item)
        left_higher = self._pair_masks.get(pair)
        if left_higher is None:
            left_higher = self.positions[:, left_item] < self.positions[:, right_item]
            left_higher.flags.writeable = False
            self._pair_masks[pair] = left_higher
        return float(np.sum(state.probabilities[left_higher]))

    def committed_pair_probability(
        self,
        order_high_to_low: tuple[int, ...],
        left_item: int,
        right_item: int,
        *,
        beta: float = 12.0,
    ) -> float:
        """Read a query from one committed global order without changing it."""

        self._validate_pair(left_item, right_item)
        if beta < 0.0:
            raise ValueError("beta must be non-negative")
        if set(order_high_to_low) != set(range(self.n_items)):
            raise ValueError("committed order must contain every item exactly once")
        rank = {item: position for position, item in enumerate(order_high_to_low)}
        margin = (rank[right_item] - rank[left_item]) / float(self.n_items - 1)
        return float(1.0 / (1.0 + np.exp(-beta * margin)))

    @staticmethod
    def posterior_entropy(state: RankingPosteriorState) -> float:
        positive = state.probabilities[state.probabilities > 0.0]
        return float(-np.sum(positive * np.log(positive)))

    def _validate_evidence(self, evidence: RelationEvidence) -> None:
        self._validate_pair(evidence.higher_item, evidence.lower_item)
        if not np.isfinite(evidence.magnitude) or not 0.0 < evidence.magnitude <= 1.0:
            raise ValueError("evidence magnitude must lie in (0, 1]")
        if not np.isfinite(evidence.reliability) or evidence.reliability < 0.0:
            raise ValueError("evidence reliability must be finite and non-negative")

    def _validate_pair(self, first: int, second: int) -> None:
        if first == second:
            raise ValueError("a relation must contain two distinct items")
        if not (0 <= first < self.n_items and 0 <= second < self.n_items):
            raise ValueError("relation item lies outside the hypothesis space")

    def _validate_state(self, state: RankingPosteriorState) -> None:
        if state.probabilities.shape != (self.n_hypotheses,):
            raise ValueError("posterior state belongs to another hypothesis space")


def evidence_from_protocol(protocol) -> tuple[RelationEvidence, ...]:
    rank = {
        item: position for position, item in enumerate(protocol.true_order_high_to_low)
    }
    return tuple(
        RelationEvidence(
            higher_item=higher,
            lower_item=lower,
            magnitude=(rank[lower] - rank[higher]) / float(protocol.n_items - 1),
        )
        for higher, lower in protocol.support_pairs_higher_lower
        for _ in range(protocol.support_blocks)
    )
