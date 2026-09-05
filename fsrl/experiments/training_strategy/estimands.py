"""Registered oriented-probability endpoints and paired complete-case bootstrap."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from numpy.typing import ArrayLike

from fsrl.analysis.policy import exact_probability
from fsrl.analysis.statistics import bootstrap_counts, summarize_subjects


@lru_cache(maxsize=16)
def _counts(seed: int, samples: int, subjects: int) -> np.ndarray:
    counts = bootstrap_counts(np.random.default_rng(seed), samples, subjects)
    counts.flags.writeable = False
    return counts


def estimate(values: np.ndarray, *, seed: int, statistics: dict) -> dict:
    """Drop undefined endpoint subjects BEFORE drawing the paired bootstrap."""

    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or np.any(np.isinf(values)):
        raise ValueError("an endpoint requires one finite-or-missing value per subject")
    finite = values[np.isfinite(values)]
    # summarize_subjects accepts an empty cohort when no resample is possible.
    counts = (
        _counts(seed, statistics["samples"], len(finite))
        if len(finite)
        else np.empty((statistics["samples"], 0))
    )
    result = summarize_subjects(finite, counts, interval=statistics["interval"])
    result["total_subjects"] = len(values)
    result["excluded_subject_indices"] = np.flatnonzero(~np.isfinite(values)).tolist()
    return result


def paired_estimate(
    first: np.ndarray, second: np.ndarray, *, seed: int, statistics: dict
) -> dict:
    first, second = np.asarray(first), np.asarray(second)
    if first.ndim != 1 or first.shape != second.shape:
        raise ValueError("paired endpoints must have the same subject axis")
    return estimate(first - second, seed=seed, statistics=statistics)


def subject_means(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    selected = np.broadcast_to(np.asarray(mask, dtype=bool), values.shape)
    denominator = selected.sum(axis=1)
    return np.divide(
        np.where(selected, values, 0.0).sum(axis=1),
        denominator,
        out=np.full(values.shape[0], np.nan),
        where=denominator > 0,
    )


def query_endpoints(
    margins: np.ndarray,
    correct_signs: ArrayLike,
    groups: dict[str, np.ndarray],
    *,
    temperature: float,
) -> dict[str, dict[str, np.ndarray]]:
    """Input axes are subject, pair, orientation; average AFTER the sigmoid."""

    margins = np.asarray(margins, dtype=np.float64)
    if margins.ndim != 3 or not np.all(np.isfinite(margins)):
        raise ValueError("margins require finite subject/pair/orientation axes")
    signs = np.broadcast_to(correct_signs, margins.shape)
    if not np.all(np.isin(signs, (-1, 1))):
        raise ValueError("correct signs must be minus or plus one")
    correct = margins * signs
    measures = {
        "probability": exact_probability(correct, temperature).mean(axis=2),
        "exact_decision": ((np.sign(correct) + 1.0) / 2.0).mean(axis=2),
    }
    return {
        measure: {name: subject_means(values, mask) for name, mask in groups.items()}
        for measure, values in measures.items()
    }
