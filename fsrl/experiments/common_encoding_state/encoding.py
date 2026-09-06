"""Marginally matched Gaussian-copula draws for stochastic teaching codes."""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr

from fsrl.experiments.minimal_learner.data import ModelBatch
from fsrl.experiments.quantized_learner.encoding import (
    canonical_addresses,
    encode_batch,
)

from .protocol import CONDITIONS


def relation_latents(keys: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Draw one independent standard normal for each subject-relation address."""
    keys = np.asarray(keys)
    if keys.ndim != 2:
        raise ValueError("relation keys must have trial-by-subject shape")
    result = np.empty(keys.shape, dtype=np.float64)
    for subject in range(keys.shape[1]):
        unique, inverse = np.unique(keys[:, subject], return_inverse=True)
        result[:, subject] = rng.standard_normal(len(unique))[inverse]
    return result


def draw_streams(batch: ModelBatch, rng: np.random.Generator) -> dict[str, np.ndarray]:
    shape = batch.arrays["signed"].shape
    keys = canonical_addresses(batch.arrays["support_cues"])[0]
    return {
        "idiosyncratic": rng.standard_normal(shape),
        "relation": relation_latents(keys, rng),
        "episode": rng.standard_normal(shape[1]),
    }


def validate_streams(batch: ModelBatch, streams: dict[str, np.ndarray]) -> None:
    shape = batch.arrays["signed"].shape
    expected = {"idiosyncratic", "relation", "episode"}
    if set(streams) != expected:
        raise ValueError("common encoder requires all and only registered streams")
    if streams["idiosyncratic"].shape != shape or streams["relation"].shape != shape:
        raise ValueError("trial-level latent stream shape differs")
    if streams["episode"].shape != (shape[1],):
        raise ValueError("episode latent stream shape differs")
    if not all(np.all(np.isfinite(value)) for value in streams.values()):
        raise ValueError("latent streams must be finite")
    keys = canonical_addresses(batch.arrays["support_cues"])[0]
    for subject in range(shape[1]):
        for key in np.unique(keys[:, subject]):
            values = streams["relation"][keys[:, subject] == key, subject]
            if not np.all(values == values[0]):
                raise ValueError("relation latent is not stable within its address")


def copula_uniforms(
    batch: ModelBatch,
    condition: str,
    rho: float,
    streams: dict[str, np.ndarray],
) -> np.ndarray:
    if condition not in CONDITIONS:
        raise ValueError("unregistered common-encoding condition")
    if not 0 <= rho < 1:
        raise ValueError("rho must lie in [0, 1)")
    validate_streams(batch, streams)
    epsilon = streams["idiosyncratic"]
    if condition == "independent":
        latent = epsilon
    else:
        shared = (
            streams["relation"]
            if condition == "relation_common"
            else streams["episode"][None, :]
        )
        latent = np.sqrt(rho) * shared + np.sqrt(1 - rho) * epsilon
    uniforms = ndtr(latent)
    if not np.all((uniforms > 0) & (uniforms < 1)):
        raise RuntimeError("finite normal draws must map inside the unit interval")
    return uniforms


def encode_common(
    batch: ModelBatch,
    condition: str,
    rho: float,
    streams: dict[str, np.ndarray],
) -> tuple[ModelBatch, dict]:
    uniforms = copula_uniforms(batch, condition, rho, streams)
    encoded, witness = encode_batch(
        batch, "resampled", uniforms, (-1, -1 / 3, 1 / 3, 1)
    )
    return encoded, {**witness, "condition": condition, "rho": rho, **streams}
