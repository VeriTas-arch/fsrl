"""Pure estimands for the frozen P/L cross-talk decomposition."""

from __future__ import annotations

import numpy as np

from fsrl.analysis.statistics import stable_sigmoid


def packed_keys(
    left: np.ndarray,
    right: np.ndarray,
    *,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """Recreate the normalized packed antisymmetric address in float32."""

    left_values = np.asarray(left, dtype=np.float32)
    right_values = np.asarray(right, dtype=np.float32)
    if left_values.shape != right_values.shape or left_values.ndim < 2:
        raise ValueError("left and right cues must have the same batched shape")
    cue_size = left_values.shape[-1]
    if cue_size < 2 or epsilon <= 0.0:
        raise ValueError("packed keys require cue_size >= 2 and epsilon > 0")
    rows, columns = np.triu_indices(cue_size, k=1)
    packed = (
        left_values[..., rows] * right_values[..., columns]
        - right_values[..., rows] * left_values[..., columns]
    ).astype(np.float32, copy=False)
    norm = np.sqrt(
        np.sum(packed * packed, axis=-1, keepdims=True, dtype=np.float32)
    ).astype(np.float32, copy=False)
    denominator = np.maximum(norm, np.float32(epsilon))
    return (packed / denominator).astype(np.float32, copy=False)


def relation_source_contributions(
    delta_evidence: np.ndarray,
    support_keys: np.ndarray,
    source_relation_indices: np.ndarray,
    query_keys: np.ndarray,
    *,
    relation_count: int,
) -> np.ndarray:
    """Return gain-free oriented contributions by source relation and query."""

    evidence = np.asarray(delta_evidence, dtype=np.float32)
    support = np.asarray(support_keys, dtype=np.float32)
    source_indices = np.asarray(source_relation_indices, dtype=np.int64)
    queries = np.asarray(query_keys, dtype=np.float32)
    if evidence.ndim != 2 or support.shape[:2] != evidence.shape:
        raise ValueError("support evidence and keys must share subject/trial axes")
    if support.ndim != 3 or queries.ndim != 3:
        raise ValueError("support and query keys must be rank three")
    if queries.shape[0] != evidence.shape[0] or queries.shape[2] != support.shape[2]:
        raise ValueError("support and query key dimensions differ")
    if source_indices.shape != evidence.shape:
        raise ValueError("one source-relation index is required per support trial")
    if np.any(source_indices < 0) or np.any(source_indices >= relation_count):
        raise ValueError("source relation index is outside the registered range")

    overlaps = np.einsum(
        "itk,iek->ite", support, queries, dtype=np.float32, optimize=False
    )
    trial_contributions = evidence[:, :, None] * overlaps
    result = np.zeros(
        (evidence.shape[0], relation_count, queries.shape[1]), dtype=np.float64
    )
    for trial_index in range(evidence.shape[1]):
        for subject in range(evidence.shape[0]):
            result[subject, source_indices[subject, trial_index]] += (
                trial_contributions[subject, trial_index]
            )
    return result


def retained_subject_mean(values: np.ndarray, retention: np.ndarray) -> np.ndarray:
    """Average relation-by-orientation values within each retained participant."""

    cells = np.asarray(values, dtype=np.float64)
    retained = np.asarray(retention, dtype=bool)
    if cells.ndim != 3 or cells.shape[:2] != retained.shape:
        raise ValueError("values must be subject x relation x orientation")
    mask = np.broadcast_to(retained[:, :, None], cells.shape)
    selected = np.where(mask, cells, np.nan).reshape(cells.shape[0], -1)
    count = np.sum(np.isfinite(selected), axis=1)
    return np.divide(
        np.nansum(selected, axis=1),
        count,
        out=np.full(cells.shape[0], np.nan, dtype=np.float64),
        where=count > 0,
    )


def probability_components(
    baseline_correct_margin: np.ndarray,
    oriented_raw_crosstalk: np.ndarray,
    query_signs: np.ndarray,
    *,
    gain: float,
    temperature: float,
) -> dict[str, np.ndarray]:
    """Compute exact and first-order effects at one fixed operating point."""

    baseline = np.asarray(baseline_correct_margin, dtype=np.float64)
    crosstalk = np.asarray(oriented_raw_crosstalk, dtype=np.float64)
    signs = np.asarray(query_signs, dtype=np.float64)
    if baseline.shape != crosstalk.shape or signs.shape != baseline.shape[1:]:
        raise ValueError("baseline, cross-talk, and orientation shapes differ")
    if gain <= 0.0 or temperature <= 0.0:
        raise ValueError("gain and temperature must be positive")
    correct_perturbation = gain * crosstalk * signs[None, :, :]
    baseline_probability = stable_sigmoid(baseline / temperature)
    sensitivity = baseline_probability * (1.0 - baseline_probability) / temperature
    exact = (
        stable_sigmoid((baseline + correct_perturbation) / temperature)
        - baseline_probability
    )
    first_order = sensitivity * correct_perturbation
    return {
        "correct_perturbation": correct_perturbation,
        "sensitivity": sensitivity,
        "exact_effect": exact,
        "first_order_effect": first_order,
        "nonlinear_remainder": exact - first_order,
    }


def source_concentration(
    source_contributions: np.ndarray,
    query_signs: np.ndarray,
) -> dict[str, np.ndarray]:
    """Describe how many omitted-source contributions form each query effect."""

    contributions = np.asarray(source_contributions, dtype=np.float64)
    signs = np.asarray(query_signs, dtype=np.float64)
    if contributions.ndim != 4 or signs.shape != contributions.shape[1:3]:
        raise ValueError(
            "source contributions must be subject x relation x orient x source"
        )
    correct = contributions * signs[None, :, :, None]
    absolute = np.abs(correct)
    mass = np.sum(absolute, axis=-1)
    descending = np.sort(absolute, axis=-1)[..., ::-1]
    top_one = np.divide(
        descending[..., 0],
        mass,
        out=np.full_like(mass, np.nan),
        where=mass > 0.0,
    )
    top_two = np.divide(
        np.sum(descending[..., :2], axis=-1),
        mass,
        out=np.full_like(mass, np.nan),
        where=mass > 0.0,
    )
    squared = np.sum(correct * correct, axis=-1)
    effective = np.divide(
        mass * mass,
        squared,
        out=np.full_like(mass, np.nan),
        where=squared > 0.0,
    )
    cancellation = np.divide(
        np.abs(np.sum(correct, axis=-1)),
        mass,
        out=np.full_like(mass, np.nan),
        where=mass > 0.0,
    )
    cumulative = np.cumsum(descending, axis=-1)
    target = 0.8 * mass[..., None]
    count_80 = np.argmax(cumulative >= target, axis=-1).astype(np.float64) + 1.0
    count_80[mass == 0.0] = np.nan
    return {
        "absolute_mass": mass,
        "top_one_share": top_one,
        "top_two_share": top_two,
        "effective_source_count": effective,
        "cancellation_ratio": cancellation,
        "source_count_80pct": count_80,
    }
