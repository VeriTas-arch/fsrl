"""Independent float64 recurrence and exact evidence-influence kernel."""

from __future__ import annotations

import numpy as np


def recurrence_and_kernel(
    support_cues: np.ndarray,
    q: np.ndarray,
    query_cues: np.ndarray,
    *,
    eta: float,
    gamma: float,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cues = np.asarray(support_cues, dtype=np.float64)
    values = np.asarray(q, dtype=np.float64)
    queries = np.asarray(query_cues, dtype=np.float64)
    trials, subjects, width = cues.shape
    if width % 2:
        raise ValueError("cue width must contain two equal item codes")
    cue_size = width // 2
    d = cues[..., :cue_size] - cues[..., cue_size:]
    d_query = queries[..., :cue_size] - queries[..., cue_size:]
    w = np.zeros((subjects, cue_size), dtype=np.float64)
    for trial in range(trials):
        feature = d[trial]
        a = eta / (epsilon + np.sum(feature * feature, axis=-1))
        error = values[trial] - np.sum(w * feature, axis=-1)
        w += (a * error)[:, None] * feature
    margins = gamma * np.einsum("bi,bqi->bq", w, d_query)

    kernel = np.empty((subjects, queries.shape[1], trials), dtype=np.float64)
    identity = np.eye(cue_size, dtype=np.float64)
    for subject in range(subjects):
        influence = np.empty((trials, cue_size), dtype=np.float64)
        suffix = identity.copy()
        for trial in range(trials - 1, -1, -1):
            feature = d[trial, subject]
            a = eta / (epsilon + feature @ feature)
            influence[trial] = suffix @ (a * feature)
            suffix = suffix @ (identity - a * np.outer(feature, feature))
        kernel[subject] = gamma * d_query[subject] @ influence.T
    reconstructed = np.einsum("bqt,tb->bq", kernel, values)
    return margins, kernel, reconstructed


__all__ = ["recurrence_and_kernel"]
