"""Pure participant-bootstrap summaries shared by maintained analyses."""

from __future__ import annotations

import numpy as np


def bootstrap_counts(
    rng: np.random.Generator, samples: int, subjects: int
) -> np.ndarray:
    if samples < 1 or subjects < 1:
        raise ValueError("samples and subjects must be positive")
    return rng.multinomial(
        subjects, np.full(subjects, 1.0 / subjects), size=samples
    ).astype(np.float64)


def bootstrap_samples(values: np.ndarray, counts: np.ndarray) -> np.ndarray:
    rows = np.asarray(values, dtype=np.float64)
    if rows.ndim != 1 or rows.shape[0] != counts.shape[1]:
        raise ValueError("bootstrap values must have one scalar per subject")
    finite = np.isfinite(rows)
    if not np.any(finite):
        return np.asarray([], dtype=np.float64)
    numerator = counts[:, finite] @ rows[finite]
    denominator = np.sum(counts[:, finite], axis=1)
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan),
        where=denominator > 0.0,
    )


def summarize_subjects(
    values: np.ndarray, counts: np.ndarray, *, interval: float
) -> dict:
    if not 0.0 < interval < 1.0:
        raise ValueError("interval must lie in (0, 1)")
    rows = np.asarray(values, dtype=np.float64)
    finite = rows[np.isfinite(rows)]
    samples = bootstrap_samples(rows, counts)
    samples = samples[np.isfinite(samples)]
    if len(finite) == 0:
        return {
            "subjects": 0,
            "mean": None,
            "median": None,
            "lower_quartile": None,
            "upper_quartile": None,
            "bootstrap": {"mean": None, "lower": None, "upper": None},
        }
    tail = (1.0 - interval) / 2.0
    return {
        "subjects": len(finite),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "lower_quartile": float(np.quantile(finite, 0.25)),
        "upper_quartile": float(np.quantile(finite, 0.75)),
        "bootstrap": {
            "mean": float(np.mean(samples)),
            "lower": float(np.quantile(samples, tail)),
            "upper": float(np.quantile(samples, 1.0 - tail)),
        },
    }


def summarize_difference(
    first: np.ndarray,
    second: np.ndarray,
    counts: np.ndarray,
    *,
    interval: float,
) -> dict:
    return summarize_subjects(
        np.asarray(first, dtype=np.float64) - np.asarray(second, dtype=np.float64),
        counts,
        interval=interval,
    )
