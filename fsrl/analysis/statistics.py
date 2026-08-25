"""Pure participant-bootstrap summaries shared by maintained analyses."""

from __future__ import annotations

import numpy as np


def json_values(values: np.ndarray) -> float | None | list:
    """Convert an array into JSON-safe finite floats and nested lists."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0:
        return None if not np.isfinite(array) else float(array)
    return [json_values(row) for row in array]


def stable_sigmoid(values: np.ndarray) -> np.ndarray:
    """Evaluate the logistic function without overflow."""

    clipped = np.clip(np.asarray(values, dtype=np.float64), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def finite_column_mean(values: np.ndarray) -> np.ndarray:
    """Average finite rows independently for each subject column."""

    rows = np.asarray(values, dtype=np.float64)
    finite = np.sum(np.isfinite(rows), axis=0)
    return np.divide(
        np.nansum(rows, axis=0),
        finite,
        out=np.full(rows.shape[1], np.nan, dtype=np.float64),
        where=finite > 0,
    )


def masked_column_mean(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Average selected finite rows independently for each subject column."""

    rows = np.where(mask, np.asarray(values, dtype=np.float64), np.nan)
    return finite_column_mean(rows)


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
