"""Pure analytic model and probability integrals for JMIC-A."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.special import expit


@dataclass(frozen=True)
class Posterior:
    mean_z: np.ndarray
    covariance_z: np.ndarray
    mean_v: np.ndarray
    covariance_v: np.ndarray
    basis: np.ndarray


def zero_sum_basis(n_items: int = 8) -> np.ndarray:
    basis = np.zeros((n_items, n_items - 1), dtype=np.float64)
    for column in range(n_items - 1):
        count = column + 1
        scale = np.sqrt(count * (count + 1.0))
        basis[:count, column] = 1.0 / scale
        basis[count, column] = -count / scale
    return basis


def incidence(edges: tuple[tuple[int, int], ...], n_items: int = 8) -> np.ndarray:
    matrix = np.zeros((len(edges), n_items), dtype=np.float64)
    for row, (positive, negative) in enumerate(edges):
        matrix[row, positive] = 1.0
        matrix[row, negative] = -1.0
    return matrix


def pair_matrix(n_items: int = 8) -> tuple[tuple[tuple[int, int], ...], np.ndarray]:
    pairs = tuple(combinations(range(n_items), 2))
    return pairs, incidence(pairs, n_items)


def posterior_7d(
    design: np.ndarray, observations: np.ndarray, *, sigma: float, tau2: float
) -> Posterior:
    if sigma <= 0.0 or tau2 <= 0.0:
        raise ValueError("sigma and tau2 must be positive")
    design = np.asarray(design, dtype=np.float64)
    observations = np.asarray(observations, dtype=np.float64)
    basis = zero_sum_basis(design.shape[1])
    reduced = design @ basis
    precision = np.eye(reduced.shape[1]) / tau2 + reduced.T @ reduced / sigma**2
    covariance_z = np.linalg.inv(precision)
    mean_z = covariance_z @ reduced.T @ observations / sigma**2
    mean_v = basis @ mean_z
    covariance_v = basis @ covariance_z @ basis.T
    return Posterior(mean_z, covariance_z, mean_v, covariance_v, basis)


def posterior_8d_constrained(
    design: np.ndarray, observations: np.ndarray, *, sigma: float, tau2: float
) -> tuple[np.ndarray, np.ndarray]:
    design = np.asarray(design, dtype=np.float64)
    observations = np.asarray(observations, dtype=np.float64)
    n_items = design.shape[1]
    precision = np.eye(n_items) / tau2 + design.T @ design / sigma**2
    constraint = np.ones((n_items, 1), dtype=np.float64)
    kkt = np.block(
        [[precision, constraint], [constraint.T, np.zeros((1, 1), dtype=np.float64)]]
    )
    inverse = np.linalg.inv(kkt)
    rhs = np.concatenate((design.T @ observations / sigma**2, [0.0]))
    return (inverse @ rhs)[:n_items], inverse[:n_items, :n_items]


def posterior_pair_distribution(
    posterior: Posterior,
) -> tuple[tuple[tuple[int, int], ...], np.ndarray, np.ndarray]:
    pairs, contrasts = pair_matrix(len(posterior.mean_v))
    means = contrasts @ posterior.mean_v
    covariance = contrasts @ posterior.covariance_v @ contrasts.T
    return pairs, means, covariance


def _hermite(nodes: int) -> tuple[np.ndarray, np.ndarray]:
    points, weights = np.polynomial.hermite.hermgauss(nodes)
    return np.sqrt(2.0) * points, weights / np.sqrt(np.pi)


def normal_sigmoid_moment(
    mean: np.ndarray,
    variance: np.ndarray,
    theta: float,
    *,
    power: int = 1,
    nodes: int = 48,
) -> np.ndarray:
    if theta <= 0.0 or power < 1:
        raise ValueError("theta and power must be positive")
    mean = np.asarray(mean, dtype=np.float64)
    variance = np.asarray(variance, dtype=np.float64)
    points, weights = _hermite(nodes)
    values = expit(
        (mean[..., None] + np.sqrt(np.maximum(variance, 0.0))[..., None] * points)
        / theta
    )
    return np.sum(weights * values**power, axis=-1)


def bivariate_sigmoid_moment(
    means: np.ndarray,
    covariance: np.ndarray,
    theta: float,
    *,
    nodes: int = 48,
) -> np.ndarray:
    means = np.asarray(means, dtype=np.float64)
    covariance = np.asarray(covariance, dtype=np.float64)
    count = len(means)
    output = np.empty((count, count), dtype=np.float64)
    first_moment = normal_sigmoid_moment(means, np.diag(covariance), theta, nodes=nodes)
    second_moment = normal_sigmoid_moment(
        means, np.diag(covariance), theta, power=2, nodes=nodes
    )
    np.fill_diagonal(output, second_moment)
    points, weights = _hermite(nodes)
    grid_weight = weights[:, None] * weights[None, :]
    for first, second in combinations(range(count), 2):
        sub = covariance[np.ix_((first, second), (first, second))]
        root = np.linalg.cholesky(sub + np.eye(2) * 1e-15)
        x = means[[first, second], None, None] + np.einsum(
            "ij,jkl->ikl",
            root,
            np.stack(np.broadcast_arrays(points[:, None], points[None, :]), axis=0),
        )
        product = expit(x[0] / theta) * expit(x[1] / theta)
        value = float(np.sum(grid_weight * product))
        output[first, second] = value
        output[second, first] = value
    if not np.allclose(np.diag(output), second_moment):
        raise RuntimeError("bivariate diagonal differs")
    if not np.isfinite(first_moment).all():
        raise RuntimeError("nonfinite sigmoid moment")
    return output


def normal_threshold_probability(
    mean: np.ndarray, variance: np.ndarray, threshold: float
) -> np.ndarray:
    from scipy.special import ndtr

    scale = np.sqrt(np.maximum(np.asarray(variance, dtype=np.float64), 0.0))
    z = np.divide(
        threshold - np.asarray(mean, dtype=np.float64),
        scale,
        out=np.where(np.asarray(mean) <= threshold, np.inf, -np.inf),
        where=scale > 0.0,
    )
    return ndtr(z)


def amplitude_matched_equal_spacing(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    ranks = np.argsort(np.argsort(values, axis=-1), axis=-1).astype(np.float64)
    ranks -= (values.shape[-1] - 1.0) / 2.0
    source_norm = np.linalg.norm(values, axis=-1, keepdims=True)
    rank_norm = np.linalg.norm(ranks, axis=-1, keepdims=True)
    return ranks * np.divide(source_norm, rank_norm, out=np.zeros_like(source_norm))


__all__ = [
    "Posterior",
    "amplitude_matched_equal_spacing",
    "bivariate_sigmoid_moment",
    "incidence",
    "normal_sigmoid_moment",
    "normal_threshold_probability",
    "pair_matrix",
    "posterior_7d",
    "posterior_8d_constrained",
    "posterior_pair_distribution",
    "zero_sum_basis",
]
