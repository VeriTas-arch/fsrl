"""Cue-addressed stochastic teaching codes, never accessible to query readout."""

import numpy as np

from fsrl.experiments.minimal_learner.data import ModelBatch


def rounding_parameters(values: np.ndarray, codebook: np.ndarray) -> tuple:
    values = np.asarray(values, dtype=np.float64)
    codebook = np.asarray(codebook, dtype=np.float64)
    if codebook.shape != (4,) or not np.all(np.diff(codebook) > 0):
        raise ValueError("the four code values must be strictly increasing")
    if not np.all(np.isfinite(values)) or np.any(
        (values < codebook[0]) | (values > codebook[-1])
    ):
        raise ValueError("observed metric is outside the fixed code range")
    lower = np.clip(np.searchsorted(codebook, values, side="right") - 1, 0, 2)
    probability = (values - codebook[lower]) / (codebook[lower + 1] - codebook[lower])
    variance = (values - codebook[lower]) * (codebook[lower + 1] - values)
    return lower, probability, variance


def draw_indices(values, uniforms, codebook) -> np.ndarray:
    uniforms = np.asarray(uniforms)
    if np.shape(values) != uniforms.shape or not np.all(
        np.isfinite(uniforms) & (uniforms >= 0) & (uniforms < 1)
    ):
        raise ValueError("one valid uniform draw is required per observed value")
    lower, probability, _ = rounding_parameters(values, codebook)
    return np.asarray(lower + (uniforms < probability), dtype=np.int8)


def canonical_addresses(cues: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Lossless binary-cue keys; lexicographic orientation, not rank order."""
    cues = np.asarray(cues)
    width = cues.shape[-1] // 2
    if cues.shape[-1] != 2 * width or not 0 < width <= 31:
        raise ValueError("binary cue pair cannot be packed into a signed int64")
    if not np.all((cues == -1) | (cues == 1)):
        raise ValueError("the registered cue alphabet is -1/+1")
    weights = np.left_shift(np.int64(1), np.arange(width - 1, -1, -1))
    left = np.sum((cues[..., :width] > 0) * weights, axis=-1)
    right = np.sum((cues[..., width:] > 0) * weights, axis=-1)
    if np.any(left == right):
        raise ValueError("a relation requires two different ordinary cues")
    keys = np.left_shift(np.minimum(left, right), width) | np.maximum(left, right)
    orientation = np.where(left < right, 1, -1).astype(np.int8)
    return keys, orientation


def persistent_indices(values, retention, keys, uniforms, codebook) -> tuple:
    """Each cache contains only observed cue keys and integer code indices."""
    indices = np.full(values.shape, -1, dtype=np.int8)
    entries = np.zeros(values.shape[1], dtype=np.int64)
    for subject in range(values.shape[1]):
        cache: dict[int, int] = {}
        for trial in range(values.shape[0]):
            if retention[trial, subject] == 0:
                continue
            key = int(keys[trial, subject])
            if key not in cache:
                cache[key] = int(
                    draw_indices(
                        values[trial, subject], uniforms[trial, subject], codebook
                    )
                )
            indices[trial, subject] = cache[key]
        entries[subject] = len(cache)
    return indices, entries


def validate_schedule(keys, canonical, retention) -> None:
    """Validate environment inputs; this exact-value check is not cached state."""
    for subject in range(keys.shape[1]):
        _, first, inverse = np.unique(
            keys[:, subject], return_index=True, return_inverse=True
        )
        if not np.array_equal(
            retention[:, subject], retention[first, subject][inverse]
        ):
            raise ValueError("relation admission must be stable across presentations")
        if not np.array_equal(
            canonical[:, subject], canonical[first, subject][inverse]
        ):
            raise ValueError("the registered task repeats a fixed observed relation")


def encode_batch(batch: ModelBatch, condition: str, uniforms, codebook) -> tuple:
    arrays = batch.arrays
    values, retention = arrays["signed"], arrays["retention"]
    if condition not in {"exact", "persistent", "resampled"}:
        raise ValueError("unregistered codec")
    if not np.all((retention == 0) | (retention == 1)):
        raise ValueError("the candidate requires binary stable admission")
    uniforms = np.asarray(uniforms)
    # Validate every draw/value even when the exact or omitted path ignores it.
    draw_indices(values, uniforms, codebook)
    keys, orientation = canonical_addresses(arrays["support_cues"])
    canonical = orientation * values
    validate_schedule(keys, canonical, retention)
    indices = np.full(values.shape, -1, dtype=np.int8)
    entries = np.zeros(values.shape[1], dtype=np.int64)
    internal = np.array(values, copy=True)
    if condition == "persistent":
        indices, entries = persistent_indices(
            canonical, retention, keys, uniforms, codebook
        )
    elif condition == "resampled":
        indices = np.where(
            retention == 1, draw_indices(canonical, uniforms, codebook), -1
        )
    if condition != "exact":
        internal = np.where(
            retention == 1, orientation * np.asarray(codebook)[indices], 0
        )
    encoded = ModelBatch(
        {**arrays, "signed": internal, "local_evidence": np.zeros_like(internal)}
    )
    witness = {
        "uniforms": uniforms,
        "cue_address_keys": keys,
        "orientation": orientation,
        "code_indices": indices,
        "cache_entries": entries,
        "cache_content_bits": 2 * entries,
        "internal_signed": internal,
    }
    return encoded, witness
