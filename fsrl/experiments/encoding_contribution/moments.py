"""Conditional first and second moments of independent unbiased teaching noise."""

from itertools import product

import numpy as np

from fsrl.experiments.adaptive_plasticity.data import relation_slots
from fsrl.experiments.minimal_learner.data import ModelBatch
from fsrl.experiments.quantized_learner.encoding import rounding_parameters
from fsrl.experiments.structural_identification.model import reference
from fsrl.experiments.structural_identification.protocol import CODEBOOK


def conditional_moments(batch, eta, gain):
    a = batch.arrays
    cues = np.asarray(a["support_cues"], dtype=np.float64)
    width = cues.shape[-1] // 2
    features = cues[..., :width] - cues[..., width:]
    mean = np.zeros((features.shape[1], width))
    covariance = np.zeros((features.shape[1], width, width))
    count = relation_slots(a["support_cues"]).max(axis=0) + 1
    _, _, variance = rounding_parameters(a["signed"], np.asarray(CODEBOOK))
    for t, x in enumerate(features):
        rate = eta / (1 + (t // count) * eta)
        k = rate * a["retention"][t] / (1e-8 + np.sum(x * x, axis=-1))
        sx = np.einsum("bij,bj->bi", covariance, x)
        energy = np.einsum("bi,bi->b", sx, x)
        covariance -= k[:, None, None] * (
            x[:, :, None] * sx[:, None, :] + sx[:, :, None] * x[:, None, :]
        )
        covariance += (k * k * (energy + variance[t]))[:, None, None] * (
            x[:, :, None] * x[:, None, :]
        )
        mean += (k * (a["signed"][t] - np.sum(mean * x, axis=-1)))[:, None] * x
    q = np.asarray(a["query_cues"], dtype=np.float64)
    q = q[..., :width] - q[..., width:]
    query_variance = gain**2 * np.einsum("bqi,bij,bqj->bq", q, covariance, q)
    if np.max(np.abs(covariance - covariance.transpose(0, 2, 1))) > 1e-10:
        raise RuntimeError("asymmetric covariance")
    if np.min(query_variance) < -1e-10:
        raise RuntimeError("negative query variance")
    return mean, covariance, np.maximum(query_variance, 0)


def enumerate_moments(batch: ModelBatch, eta, gain):
    """Independent full enumeration of at most four teaching trials, one subject."""
    a = batch.arrays
    if a["signed"].shape[1] != 1 or len(a["signed"]) > 4:
        raise ValueError("enumeration requires a tiny one-subject fixture")
    lower, p, _ = rounding_parameters(a["signed"], np.asarray(CODEBOOK))
    states, weights = [], []
    for bits in product((0, 1), repeat=len(p)):
        upper = np.asarray(bits)[:, None]
        signed = np.asarray(CODEBOOK)[lower + upper]
        weight = np.prod(np.where(upper, p, 1 - p))
        _, state = reference(ModelBatch({**a, "signed": signed}), eta, gain, "decay")
        weights.append(weight)
        states.append(state[0])
    states, weights = np.asarray(states), np.asarray(weights)
    mean = weights @ states
    centered = states - mean
    covariance = np.einsum("b,bi,bj->ij", weights, centered, centered)
    return mean[None], covariance[None]
