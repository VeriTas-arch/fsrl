"""Endpoint and geometry summaries for the time-role diagnostic."""

from __future__ import annotations

import numpy as np

from .estimands import hodge


def stable_sigmoid(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    output = np.empty_like(x)
    positive = x >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exponential = np.exp(x[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


def geometry(candidate: np.ndarray, control: np.ndarray, tolerance: float) -> dict:
    left, right = (
        np.asarray(candidate, dtype=np.float64),
        np.asarray(control, dtype=np.float64),
    )
    left_h, right_h = hodge(left), hodge(right)
    left_norm = np.linalg.norm(left_h["potentials"], axis=-1)
    right_norm = np.linalg.norm(right_h["potentials"], axis=-1)
    cosine = np.sum(left_h["potentials"] * right_h["potentials"], axis=-1) / (
        left_norm * right_norm + 1e-12
    )
    ties = (np.abs(left) <= tolerance) | (np.abs(right) <= tolerance)
    flips = (np.sign(left) != np.sign(right)) & ~ties
    return {
        "field_cosine_mean": float(
            np.mean(
                np.sum(left * right, axis=-1)
                / (
                    np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1)
                    + 1e-12
                )
            )
        ),
        "potential_cosine_mean": float(np.mean(cosine)),
        "potential_norm_ratio_mean": float(np.mean(left_norm / (right_norm + 1e-12))),
        "sign_flip_fraction": float(np.mean(flips)),
        "tie_fraction": float(np.mean(ties)),
        "candidate_residual_fraction_mean": float(np.mean(left_h["residual_fraction"])),
        "control_residual_fraction_mean": float(np.mean(right_h["residual_fraction"])),
    }


def generic_endpoints(
    margins: np.ndarray,
    targets: np.ndarray,
    learned: np.ndarray,
) -> dict[str, np.ndarray]:
    target_values = np.asarray(targets)
    if target_values.shape != margins.shape:
        target_values = target_values.reshape(-1, margins.shape[0]).T
    signs = 2 * target_values - 1
    probability = stable_sigmoid(signs * margins)
    mask = np.asarray(learned, dtype=bool)
    return {
        "generic_learned": np.sum(probability * mask, axis=1) / np.sum(mask, axis=1),
        "generic_nonlearned": np.sum(probability * ~mask, axis=1)
        / np.sum(~mask, axis=1),
    }


def liu_endpoints(
    margins: np.ndarray,
    targets: np.ndarray,
    query_pairs: np.ndarray,
    support_pairs: np.ndarray,
    retention: np.ndarray,
) -> dict[str, np.ndarray]:
    subjects = margins.shape[0]
    signs = 2 * np.asarray(targets).reshape(-1, subjects).T - 1
    probability = stable_sigmoid(signs * margins / 0.25)
    support = [tuple(sorted(map(int, pair))) for pair in support_pairs[:8, 0]]
    relation_index = {pair: index for index, pair in enumerate(support)}
    indices = np.asarray(
        [relation_index.get(tuple(sorted(map(int, pair))), -1) for pair in query_pairs]
    )
    learned = indices >= 0
    omitted = np.zeros((subjects, len(query_pairs)), dtype=bool)
    for query, relation in enumerate(indices):
        if relation >= 0:
            omitted[:, query] = ~np.asarray(retention[:, relation], dtype=bool)
    omitted_count = np.sum(omitted, axis=1)
    omitted_probability = np.divide(
        np.sum(probability * omitted, axis=1),
        omitted_count,
        out=np.full(subjects, np.nan, dtype=np.float64),
        where=omitted_count > 0,
    )
    return {
        "liu_learned": probability[:, learned].mean(1),
        "liu_nonlearned": probability[:, ~learned].mean(1),
        "liu_omitted": omitted_probability,
    }


def paired_interval(panels: list[np.ndarray], *, seed: int, draws: int = 2000) -> dict:
    rng = np.random.default_rng(seed)
    bootstrap = []
    for values in panels:
        array = np.asarray(values, dtype=np.float64)
        array = array[np.isfinite(array)]
        if not len(array):
            raise ValueError("paired interval has no complete cases in a panel")
        indices = rng.integers(0, len(array), size=(draws, len(array)))
        bootstrap.append(array[indices].mean(1))
    distribution = np.mean(np.stack(bootstrap), axis=0)
    point = float(
        np.mean([np.mean(np.asarray(values)[np.isfinite(values)]) for values in panels])
    )
    return {
        "point": point,
        "interval": {
            "lower": float(np.quantile(distribution, 0.025)),
            "upper": float(np.quantile(distribution, 0.975)),
        },
    }


__all__ = [
    "generic_endpoints",
    "geometry",
    "liu_endpoints",
    "paired_interval",
    "stable_sigmoid",
]
