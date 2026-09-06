"""Non-Liu likelihood recovery for the common-state encoder."""

from __future__ import annotations

import math

import numpy as np
from scipy.special import logsumexp, ndtr, ndtri

from fsrl.experiments.minimal_learner.data import ModelBatch
from fsrl.experiments.quantized_learner.encoding import (
    canonical_addresses,
    rounding_parameters,
)

from .encoding import copula_uniforms, draw_streams
from .protocol import CONDITIONS, QUADRATURE_NODES, RHO_GRID


def upper_probabilities(batch: ModelBatch, codebook) -> tuple[np.ndarray, np.ndarray]:
    keys, orientation = canonical_addresses(batch.arrays["support_cues"])
    canonical = orientation * batch.arrays["signed"]
    _, probabilities, _ = rounding_parameters(canonical, np.asarray(codebook))
    return probabilities, keys


def choices(
    batch: ModelBatch,
    condition: str,
    rho: float,
    streams: dict[str, np.ndarray],
    codebook,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    probabilities, keys = upper_probabilities(batch, codebook)
    uniforms = copula_uniforms(batch, condition, rho, streams)
    return uniforms < probabilities, probabilities, keys


def _quadrature(nodes: int = QUADRATURE_NODES) -> tuple[np.ndarray, np.ndarray]:
    points, weights = np.polynomial.hermite.hermgauss(nodes)
    return np.sqrt(2) * points, np.log(weights) - 0.5 * math.log(math.pi)


def independent_log_likelihood(observed: np.ndarray, probability: np.ndarray) -> float:
    active = (probability > 0) & (probability < 1)
    y = observed[active]
    p = probability[active]
    return float(np.sum(np.where(y, np.log(p), np.log1p(-p))))


def _group_log_likelihood(
    observed: np.ndarray,
    probability: np.ndarray,
    masks: list[np.ndarray],
    rho: float,
) -> float:
    points, log_weights = _quadrature()
    result = 0.0
    for mask in masks:
        active = mask & (probability > 0) & (probability < 1)
        if not np.any(active):
            continue
        y = observed[active]
        thresholds = ndtri(probability[active])
        conditional = ndtr(
            (thresholds[None, :] - np.sqrt(rho) * points[:, None]) / np.sqrt(1 - rho)
        )
        conditional = np.clip(
            conditional, np.finfo(float).tiny, 1 - np.finfo(float).eps
        )
        log_probability = np.where(
            y[None, :], np.log(conditional), np.log1p(-conditional)
        )
        integrated = np.asarray(
            logsumexp(log_weights + log_probability.sum(axis=1))
        ).item()
        result += float(integrated)
    return result


def common_log_likelihood(
    observed: np.ndarray,
    probability: np.ndarray,
    keys: np.ndarray,
    condition: str,
    rho: float,
) -> float:
    if condition not in {"relation_common", "episode_common"}:
        raise ValueError("common likelihood requires a registered shared grouping")
    total = 0.0
    for subject in range(observed.shape[1]):
        if condition == "episode_common":
            masks = [np.ones(observed.shape[0], dtype=bool)]
        else:
            masks = [keys[:, subject] == key for key in np.unique(keys[:, subject])]
        total += _group_log_likelihood(
            observed[:, subject], probability[:, subject], masks, rho
        )
    return total


def likelihoods(
    observed: np.ndarray, probability: np.ndarray, keys: np.ndarray, rho: float
) -> dict[str, float]:
    return {
        "independent": independent_log_likelihood(observed, probability),
        **{
            condition: common_log_likelihood(
                observed, probability, keys, condition, rho
            )
            for condition in CONDITIONS[1:]
        },
    }


def recover_one(
    batch: ModelBatch,
    generator: str,
    rho: float,
    rng: np.random.Generator,
    codebook,
) -> dict:
    streams = draw_streams(batch, rng)
    observed, probability, keys = choices(batch, generator, rho, streams, codebook)
    architecture_scores = likelihoods(observed, probability, keys, rho)
    rho_scores = (
        {}
        if generator == "independent"
        else {
            str(candidate): common_log_likelihood(
                observed, probability, keys, generator, candidate
            )
            for candidate in RHO_GRID
        }
    )
    return {
        "architecture": max(architecture_scores, key=architecture_scores.__getitem__),
        "rho": None
        if not rho_scores
        else float(max(rho_scores, key=rho_scores.__getitem__)),
        "architecture_log_likelihood": architecture_scores,
        "rho_log_likelihood": rho_scores,
    }
