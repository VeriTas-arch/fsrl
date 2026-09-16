"""Pure estimands for the frozen single-P time-role diagnostic."""

from __future__ import annotations

from itertools import combinations

import numpy as np


def incidence(n_items: int = 8) -> tuple[tuple[tuple[int, int], ...], np.ndarray]:
    pairs = tuple(combinations(range(n_items), 2))
    matrix = np.zeros((len(pairs), n_items), dtype=np.float64)
    for row, (left, right) in enumerate(pairs):
        matrix[row, left] = 1.0
        matrix[row, right] = -1.0
    return pairs, matrix


def canonical_field(
    margins: np.ndarray,
    query_pairs: np.ndarray,
    *,
    n_items: int = 8,
) -> np.ndarray:
    """Antisymmetrize ordered margins into canonical i<j fields."""
    values = np.asarray(margins, dtype=np.float64)
    pairs = np.asarray(query_pairs, dtype=np.int64)
    if values.shape != pairs.shape[:-1] or pairs.shape[-1] != 2:
        raise ValueError("margins and query pairs have incompatible shapes")
    canonical_pairs, _ = incidence(n_items)
    output = np.empty((*values.shape[:-1], len(canonical_pairs)), dtype=np.float64)
    # The query axis is the final margin axis.
    for index, (left, right) in enumerate(canonical_pairs):
        forward = (pairs[..., 0] == left) & (pairs[..., 1] == right)
        reverse = (pairs[..., 0] == right) & (pairs[..., 1] == left)
        if not np.all(forward.sum(axis=-1) == 1) or not np.all(
            reverse.sum(axis=-1) == 1
        ):
            raise ValueError("each ordered query orientation must occur exactly once")
        forward_value = np.sum(np.where(forward, values, 0.0), axis=-1)
        reverse_value = np.sum(np.where(reverse, values, 0.0), axis=-1)
        output[..., index] = 0.5 * (forward_value - reverse_value)
    return output


def hodge(fields: np.ndarray, *, n_items: int = 8) -> dict[str, np.ndarray]:
    values = np.asarray(fields, dtype=np.float64)
    pairs, matrix = incidence(n_items)
    if values.shape[-1] != len(pairs):
        raise ValueError("field width differs from complete-graph geometry")
    operator = np.linalg.pinv(matrix)
    potentials = values @ operator.T
    gradient = potentials @ matrix.T
    residual = values - gradient
    total = np.sum(values * values, axis=-1)
    residual_fraction = np.divide(
        np.sum(residual * residual, axis=-1),
        total,
        out=np.zeros_like(total),
        where=total > 0,
    )
    return {
        "potentials": potentials,
        "gradient": gradient,
        "residual": residual,
        "residual_fraction": residual_fraction,
    }


def positive_scale(candidate: np.ndarray, control: np.ndarray) -> float:
    left = np.asarray(candidate, dtype=np.float64).reshape(-1)
    right = np.asarray(control, dtype=np.float64).reshape(-1)
    if left.shape != right.shape or not len(left):
        raise ValueError("scale fields must be paired and nonempty")
    denominator = float(left @ left)
    return 0.0 if denominator == 0.0 else max(0.0, float(left @ right) / denominator)


def derangement(length: int, seed: int) -> np.ndarray:
    if length < 2:
        raise ValueError("derangement requires at least two positions")
    rng = np.random.default_rng(seed)
    identity = np.arange(length)
    for _ in range(10000):
        proposal = rng.permutation(length)
        if np.all(proposal != identity):
            return proposal
    raise RuntimeError("failed to construct deterministic derangement")


def effective_distance(
    baseline: np.ndarray, changed: np.ndarray, alpha: np.ndarray
) -> np.ndarray:
    base = np.asarray(baseline, dtype=np.float64) * np.asarray(alpha, dtype=np.float64)
    delta = (np.asarray(changed, dtype=np.float64) - baseline) * alpha
    numerator = np.linalg.norm(delta, axis=(-2, -1))
    denominator = np.linalg.norm(base, axis=(-2, -1)) + 1e-12
    return numerator / denominator


def fixed_effect_slope(
    susceptibility: np.ndarray, maturity: np.ndarray, prefixes: np.ndarray
) -> float:
    y, x, groups = np.broadcast_arrays(
        np.asarray(susceptibility, dtype=np.float64),
        np.asarray(maturity, dtype=np.float64),
        np.asarray(prefixes),
    )
    y_residual = np.empty_like(y)
    x_residual = np.empty_like(x)
    for group in np.unique(groups):
        selected = groups == group
        y_residual[selected] = y[selected] - np.mean(y[selected])
        x_residual[selected] = x[selected] - np.mean(x[selected])
    denominator = float(np.sum(x_residual * x_residual))
    if denominator == 0.0:
        raise ValueError("maturity has no within-prefix variation")
    return float(np.sum(x_residual * y_residual) / denominator)


__all__ = [
    "canonical_field",
    "derangement",
    "effective_distance",
    "fixed_effect_slope",
    "hodge",
    "incidence",
    "positive_scale",
]
