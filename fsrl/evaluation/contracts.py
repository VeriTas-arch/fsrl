"""Typed intervention and output contracts for frozen evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class FastWeightIntervention(str, Enum):
    INTACT = "intact"
    WRITE_OFF = "write_off"
    ALPHA_ZERO = "alpha_zero"
    RESET = "reset"
    SHUFFLE = "shuffle"


class FrozenEvaluationBackend(str, Enum):
    """Explicit execution backend for frozen causal evaluation."""

    LEGACY_STEPWISE = "legacy_stepwise"
    BATCHED_SEQUENCE = "batched_sequence"


@dataclass(frozen=True)
class ConditionMetrics:
    intervention: str
    overall_accuracy: float
    learned_accuracy: float
    nonlearned_accuracy: float
    mean_probability_correct: float
    mean_abs_fast_weight: float
    mean_circular_triads: float
    mean_transitive_triplet_fraction: float


@dataclass(frozen=True)
class OrderInvarianceMetrics:
    schedules: int
    pairs: int
    max_abs_logit_delta: float
    mean_abs_logit_delta: float
