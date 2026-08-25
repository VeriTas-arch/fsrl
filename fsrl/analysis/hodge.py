"""Pure complete-graph Hodge geometry used across registered analyses."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from ..tasks.protocol import RankingProtocol

POTENTIAL_ZERO_TOLERANCE = 1e-12


@dataclass(frozen=True)
class CompleteGraphGeometry:
    pairs: tuple[tuple[int, int], ...]
    incidence: np.ndarray
    projection: np.ndarray
    score_operator: np.ndarray
    true_sign: np.ndarray
    true_potential: np.ndarray


def build_complete_graph_geometry(protocol: RankingProtocol) -> CompleteGraphGeometry:
    pairs = tuple(combinations(range(protocol.n_items), 2))
    incidence = np.zeros((len(pairs), protocol.n_items), dtype=np.float64)
    for index, (first, second) in enumerate(pairs):
        incidence[index, first] = 1.0
        incidence[index, second] = -1.0
    score_operator = np.linalg.pinv(incidence)
    projection = incidence @ score_operator
    true_positions = np.empty(protocol.n_items, dtype=np.int64)
    for position, item in enumerate(protocol.true_order_high_to_low):
        true_positions[item] = position
    true_sign = np.asarray(
        [
            1.0 if true_positions[first] < true_positions[second] else -1.0
            for first, second in pairs
        ],
        dtype=np.float64,
    )
    return CompleteGraphGeometry(
        pairs=pairs,
        incidence=incidence,
        projection=projection,
        score_operator=score_operator,
        true_sign=true_sign,
        true_potential=normalize_potentials(-true_positions.astype(np.float64)),
    )


def hodge_potentials(fields: np.ndarray, geometry: CompleteGraphGeometry) -> np.ndarray:
    values = np.asarray(fields, dtype=np.float64)
    if values.shape[-1] != len(geometry.pairs):
        raise ValueError("field does not match the complete-graph edge order")
    return values @ geometry.score_operator.T


def gradient_energy_fraction(
    fields: np.ndarray, geometry: CompleteGraphGeometry
) -> np.ndarray:
    values = np.asarray(fields, dtype=np.float64)
    if values.shape[-1] != len(geometry.pairs):
        raise ValueError("field does not match the complete-graph edge order")
    gradient = values @ geometry.projection.T
    gradient_energy = np.sum(gradient * gradient, axis=-1)
    total_energy = np.sum(values * values, axis=-1)
    return np.divide(
        gradient_energy,
        total_energy,
        out=np.full_like(total_energy, np.nan),
        where=total_energy > 0.0,
    )


def vector_gradient_energy_fraction(
    fields: np.ndarray, geometry: CompleteGraphGeometry
) -> np.ndarray:
    values = np.asarray(fields, dtype=np.float64)
    if values.shape[-2] != len(geometry.pairs):
        raise ValueError("vector field does not match the complete-graph edge order")
    gradient = np.einsum("ef,...fd->...ed", geometry.projection, values)
    gradient_energy = np.sum(gradient * gradient, axis=(-2, -1))
    total_energy = np.sum(values * values, axis=(-2, -1))
    return np.divide(
        gradient_energy,
        total_energy,
        out=np.full_like(total_energy, np.nan),
        where=total_energy > 0.0,
    )


def normalize_potentials(potentials: np.ndarray) -> np.ndarray:
    values = np.asarray(potentials, dtype=np.float64)
    centered = values - np.mean(values, axis=-1, keepdims=True)
    norms = np.linalg.norm(centered, axis=-1, keepdims=True)
    return np.divide(
        centered,
        norms,
        out=np.zeros_like(centered),
        where=norms > POTENTIAL_ZERO_TOLERANCE,
    )


def potential_alignment(first: np.ndarray, second: np.ndarray) -> dict[str, np.ndarray]:
    left = normalize_potentials(first)
    right = normalize_potentials(second)
    cosine = np.sum(left * right, axis=-1)
    valid = (np.linalg.norm(left, axis=-1) > POTENTIAL_ZERO_TOLERANCE) & (
        np.linalg.norm(right, axis=-1) > POTENTIAL_ZERO_TOLERANCE
    )
    cosine = np.where(valid, cosine, np.nan)
    return {
        "cosine": cosine,
        "pearson": cosine.copy(),
        "kendall_tau": kendall_tau_scores(left, right),
    }


def kendall_tau_scores(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    left, right = np.broadcast_arrays(
        np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    )
    if left.ndim == 1:
        left = left[None, :]
        right = right[None, :]
        squeeze = True
    else:
        squeeze = False
    flat_left = left.reshape(-1, left.shape[-1])
    flat_right = right.reshape(-1, right.shape[-1])
    values = []
    item_pairs = tuple(combinations(range(left.shape[-1]), 2))
    for first_row, second_row in zip(flat_left, flat_right, strict=True):
        products = np.asarray(
            [
                (first_row[i] - first_row[j]) * (second_row[i] - second_row[j])
                for i, j in item_pairs
            ]
        )
        nonzero = products != 0.0
        values.append(
            np.nan
            if not np.any(nonzero)
            else float(np.mean(np.sign(products[nonzero])))
        )
    result = np.asarray(values).reshape(left.shape[:-1])
    return result[0] if squeeze else result
