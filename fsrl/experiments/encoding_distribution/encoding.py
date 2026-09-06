"""Gaussian teaching and independent dense propagation of teaching impulses."""

import numpy as np
from scipy.special import ndtri

from fsrl.experiments.adaptive_plasticity.data import relation_slots
from fsrl.experiments.minimal_learner.data import ModelBatch
from fsrl.experiments.quantized_learner.encoding import (
    canonical_addresses,
    rounding_parameters,
    validate_schedule,
)
from fsrl.experiments.structural_identification.protocol import CODEBOOK


def gaussian(batch, uniforms):
    a = batch.arrays
    u = np.asarray(uniforms)
    if u.shape != a["signed"].shape or not np.all((u > 0) & (u < 1)):
        raise ValueError("one open-interval uniform per presentation is required")
    keys, orientation = canonical_addresses(a["support_cues"])
    canonical = orientation * a["signed"]
    validate_schedule(keys, canonical, a["retention"])
    codebook = np.asarray(CODEBOOK)
    lower, _, variance = rounding_parameters(canonical, codebook)
    internal = canonical - np.sqrt(variance) * ndtri(u)
    admitted = a["retention"] == 1
    tail = {
        "admitted": int(admitted.sum()),
        "outside_adjacent": int(
            np.sum(
                admitted
                & ((internal < codebook[lower]) | (internal > codebook[lower + 1]))
            )
        ),
        "outside_display": int(np.sum(admitted & (np.abs(internal) > 1))),
    }
    signed = np.where(admitted, orientation * internal, 0)
    return ModelBatch(
        {**a, "signed": signed, "local_evidence": np.zeros_like(signed)}
    ), tail


def impulse_map(batch, eta):
    """Dense B matrices propagate all teaching basis vectors, independently of moments.py."""
    a = batch.arrays
    width = a["support_cues"].shape[-1] // 2
    cues = np.asarray(a["support_cues"], dtype=float)
    x = cues[..., :width] - cues[..., width:]
    trials, subjects = x.shape[:2]
    mapping = np.zeros((subjects, width, trials))
    counts = relation_slots(a["support_cues"]).max(axis=0) + 1
    for t in range(trials):
        rate = eta / (1 + (t // counts) * eta)
        k = rate * a["retention"][t] / (1e-8 + np.sum(x[t] ** 2, axis=1))
        matrix = (
            np.eye(width)[None] - k[:, None, None] * x[t, :, :, None] * x[t, :, None, :]
        )
        mapping = matrix @ mapping
        mapping[:, :, t] += k[:, None] * x[t]
    return mapping


def impulse_moments(batch, eta, gain):
    mapping = impulse_map(batch, eta)
    _, _, variance = rounding_parameters(batch.arrays["signed"], np.asarray(CODEBOOK))
    factor = mapping * np.sqrt(variance.T)[:, None, :]
    mean = np.einsum("bit,tb->bi", mapping, batch.arrays["signed"])
    covariance = factor @ factor.transpose(0, 2, 1)
    q = np.asarray(batch.arrays["query_cues"], dtype=float)
    width = q.shape[-1] // 2
    query_factor = gain * ((q[..., :width] - q[..., width:]) @ factor)
    gamma = query_factor @ query_factor.transpose(0, 2, 1)
    return mapping, mean, covariance, gamma
