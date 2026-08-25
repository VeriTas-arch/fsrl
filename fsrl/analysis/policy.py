"""Pure transformations between policy margins, logits, and probabilities."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def exact_probability(
    correct_signed_margin: np.ndarray, temperature: float
) -> np.ndarray:
    """Map correct-signed margins to probabilities at a fixed temperature."""

    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    scaled = np.asarray(correct_signed_margin, dtype=np.float64) / temperature
    scaled = np.clip(scaled, -700.0, 700.0)
    return 1.0 / (1.0 + np.exp(-scaled))


def bundle_logits(
    bundle: dict, pair_schedules: Sequence[Sequence[tuple[int, int]]]
) -> tuple[dict[tuple[int, int], float], ...]:
    """Convert a dense subject-by-query logit array to keyed subject rows."""

    return tuple(
        {
            pair: float(bundle["logits"][subject, index])
            for index, pair in enumerate(pair_schedules[subject])
        }
        for subject in range(len(pair_schedules))
    )


def margin_fields(bundle: dict, n_items: int) -> np.ndarray:
    """Return antisymmetric margins from adjacent forward/reverse logits."""

    del n_items  # Retained in the public signature used by frozen analyses.
    return 0.5 * (bundle["logits"][:, 0::2] - bundle["logits"][:, 1::2])
