"""Exact fixed-encoding derivative and direct/history decomposition."""

import numpy as np


def score_history(x, signed, retention, query, *, eta, gain, epsilon) -> dict:
    """Inputs are subject/trial/feature, subject/trial and subject/query/feature."""
    x, signed, retention, query = (
        np.asarray(value, dtype=np.float64) for value in (x, signed, retention, query)
    )
    subjects, trials, dimensions = x.shape
    a = eta * retention / (epsilon + np.sum(x * x, axis=-1))
    tail = np.broadcast_to(
        np.eye(dimensions), (subjects, dimensions, dimensions)
    ).copy()
    full = np.empty((subjects, trials, query.shape[1]))
    direct = np.empty_like(full)
    for t in range(trials - 1, -1, -1):
        propagated = np.einsum("bij,bj->bi", tail, x[:, t])
        full[:, t] = gain * a[:, t, None] * np.einsum("bqd,bd->bq", query, propagated)
        direct[:, t] = gain * a[:, t, None] * np.einsum("bqd,bd->bq", query, x[:, t])
        tail -= a[:, t, None, None] * propagated[:, :, None] * x[:, t, None, :]
    w = np.zeros((subjects, dimensions))
    for t in range(trials):
        error = signed[:, t] - np.sum(w * x[:, t], axis=-1)
        w += a[:, t, None] * error[:, None] * x[:, t]
    margin = gain * np.einsum("bqd,bd->bq", query, w)
    contributions = signed[:, :, None] * full
    np.testing.assert_allclose(contributions.sum(axis=1), margin, atol=1e-9, rtol=1e-7)
    return {
        "sensitivity": full,
        "direct_sensitivity": direct,
        "history_sensitivity": full - direct,
        "global_margin": margin,
        "w": w,
        "direct_margin": np.sum(signed[:, :, None] * direct, axis=1),
        "history_margin": np.sum(signed[:, :, None] * (full - direct), axis=1),
    }
