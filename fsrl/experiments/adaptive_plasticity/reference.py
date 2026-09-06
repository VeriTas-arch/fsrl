"""Independent float64 recurrence and exact teaching-value sensitivities."""

from __future__ import annotations

import numpy as np

from fsrl.experiments.minimal_learner.data import ModelBatch

from .data import occurrence_indices, relation_slots


def rollout(
    batch: ModelBatch,
    *,
    eta: float,
    gain: float,
    epsilon: float,
    adaptive: bool,
    scheduler: str = "relation",
    with_sensitivity: bool = False,
) -> dict[str, np.ndarray]:
    arrays = batch.arrays
    cues = np.asarray(arrays["support_cues"], dtype=np.float64)
    signed = np.asarray(arrays["signed"], dtype=np.float64)
    retention = np.asarray(arrays["retention"], dtype=np.float64)
    query = np.asarray(arrays["query_cues"], dtype=np.float64)
    slots = relation_slots(cues)
    trials, subjects = signed.shape
    width = cues.shape[-1] // 2
    max_relations = int(slots.max()) + 1
    relation_count = slots.max(axis=0) + 1
    w = np.zeros((subjects, width), dtype=np.float64)
    efficacy = np.full((subjects, max_relations), eta, dtype=np.float64)
    derivative = np.zeros((subjects, trials, width), dtype=np.float64)
    for trial in range(trials):
        x = cues[trial, :, :width] - cues[trial, :, width:]
        if not adaptive:
            step_eta = np.full(subjects, eta)
        elif scheduler == "relation":
            step_eta = efficacy[np.arange(subjects), slots[trial]]
        elif scheduler == "global":
            step_eta = eta / (1 + (trial // relation_count) * eta)
        else:
            raise ValueError("scheduler must be relation or global")
        alpha = step_eta * retention[trial] / (epsilon + np.square(x).sum(axis=1))
        if with_sensitivity:
            projection = np.einsum("sd,std->st", x, derivative)
            derivative -= alpha[:, None, None] * projection[:, :, None] * x[:, None]
            derivative[:, trial] += alpha[:, None] * x
        error = signed[trial] - np.sum(w * x, axis=1)
        w += (alpha * error)[:, None] * x
        if adaptive and scheduler == "relation":
            current = step_eta
            efficacy[np.arange(subjects), slots[trial]] += retention[trial] * (
                current / (1 + current) - current
            )
    q = query[..., :width] - query[..., width:]
    margins = gain * np.einsum("sd,sqd->sq", w, q)
    result = {"margins": margins, "w": w, "efficacy": efficacy}
    if with_sensitivity:
        result["sensitivity"] = gain * np.einsum("std,sqd->tsq", derivative, q)
    return result


def occurrence_sensitivity(
    batch: ModelBatch,
    sensitivity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    occurrence = occurrence_indices(batch.arrays["support_cues"])
    retained = np.asarray(batch.arrays["retention"]) == 1
    per_subject = np.empty((retained.shape[1], 4), dtype=np.float64)
    for subject in range(retained.shape[1]):
        for index in range(4):
            selected = retained[:, subject] & (occurrence[:, subject] == index)
            values = sensitivity[selected, subject]
            if not len(values):
                raise RuntimeError("sensitivity fixture lacks an admitted occurrence")
            per_subject[subject, index] = np.sqrt(np.mean(np.square(values)))
    denominator = per_subject.sum(axis=1)
    if np.any(denominator <= 0):
        raise RuntimeError("zero occurrence sensitivity")
    return per_subject, per_subject[:, 3] / denominator
