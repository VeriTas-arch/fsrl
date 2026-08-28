"""Shared query schedules and field reconstruction for frozen evaluators."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from fsrl.evaluation.frozen_fast_weight import FrozenFastWeightEvaluator


class PairGeometry(Protocol):
    """Minimal complete-graph geometry surface required by field readout."""

    @property
    def pairs(self) -> tuple[tuple[int, int], ...]: ...


def ordered_query_schedule(
    geometry: PairGeometry, subjects: int
) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Repeat the geometry's forward/reverse query order for every subject."""

    ordered = tuple(
        oriented
        for first, second in geometry.pairs
        for oriented in ((first, second), (second, first))
    )
    return tuple(ordered for _ in range(subjects))


def readout_margin_fields(
    evaluator: FrozenFastWeightEvaluator,
    fast_weights,
    geometry: PairGeometry,
    *,
    alpha_zero: bool = False,
) -> np.ndarray:
    """Reconstruct one antisymmetric complete-graph margin field per subject."""

    schedules = ordered_query_schedule(geometry, evaluator.config.bs)
    logits = evaluator.readout_logits(fast_weights, schedules, alpha_zero=alpha_zero)
    return np.asarray(
        [
            [0.5 * (row[pair] - row[(pair[1], pair[0])]) for pair in geometry.pairs]
            for row in logits
        ],
        dtype=np.float64,
    )
