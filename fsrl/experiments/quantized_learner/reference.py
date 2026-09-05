"""Independent float64 recurrence and exact conditional code mixtures."""

from itertools import product

import numpy as np

from fsrl.experiments.minimal_learner.history import score_history

from .encoding import canonical_addresses, rounding_parameters


def joint_choice_log_probability(
    margins, weights, left_counts, right_counts, *, temperature
):
    """Probability of a choice sequence, mixing shared state AFTER its queries.

    Counts summarize a particular sequence; no binomial multiplicity is added.
    This supports both exact code mixtures and sampled hidden-admission mixtures.
    """
    weights = np.asarray(weights, dtype=np.float64)
    if np.any(weights < 0) or not np.isclose(weights.sum(), 1) or temperature <= 0:
        raise ValueError(
            "a normalized state mixture and positive temperature are required"
        )
    logits = np.asarray(margins, dtype=np.float64) / temperature
    conditional = -np.sum(
        np.logaddexp(0, -logits) * left_counts + np.logaddexp(0, logits) * right_counts,
        axis=-1,
    )
    positive = weights > 0
    return float(np.logaddexp.reduce(np.log(weights[positive]) + conditional[positive]))


def rollout(arrays: dict, *, eta: float, gain: float, epsilon: float) -> dict:
    cues = np.asarray(arrays["support_cues"], dtype=np.float64)
    query = np.asarray(arrays["query_cues"], dtype=np.float64)
    width = cues.shape[-1] // 2
    w = np.zeros((cues.shape[1], width))
    trajectory = [w.copy()]
    for trial, pair in enumerate(cues):
        x = pair[:, :width] - pair[:, width:]
        a = eta * arrays["retention"][trial] / (epsilon + np.sum(x * x, axis=-1))
        prediction = np.einsum("bf,bf->b", w, x)
        w = w + (a * (arrays["signed"][trial] - prediction))[:, None] * x
        trajectory.append(w.copy())
    margin = gain * np.einsum("bf,bqf->bq", w, query[..., :width] - query[..., width:])
    return {"w": w, "trajectory": np.asarray(trajectory), "margins": margin}


def relation_support(arrays: dict, subject: int, codebook) -> dict:
    keys, orientation = canonical_addresses(arrays["support_cues"][:, subject])
    retention = arrays["retention"][:, subject]
    admitted = np.flatnonzero(retention == 1)
    first: dict[int, int] = {}
    for trial in admitted:
        first.setdefault(int(keys[trial]), int(trial))
    trials = list(first.values())
    values = orientation[trials] * arrays["signed"][trials, subject]
    lower, probability, variance = rounding_parameters(values, codebook)
    projection = np.zeros((len(keys), len(first)))
    for column, (key, trial) in enumerate(first.items()):
        same = keys == key
        if not np.all(retention[same] == 1):
            raise ValueError("relation admission must be stable")
        # Task-integrity check, not a state read available to the learner.
        observed = orientation[same] * arrays["signed"][same, subject]
        if not np.all(
            observed == orientation[trial] * arrays["signed"][trial, subject]
        ):
            raise ValueError("the fixed-value relation changed across repetitions")
        projection[same, column] = orientation[same]
    return {
        "projection": projection,
        "lower": lower,
        "probability": probability,
        "variance": variance,
    }


def code_combinations(lower, probability, codebook) -> tuple:
    options = [
        tuple(
            (codebook[index + side], weight)
            for side, weight in ((0, 1 - p), (1, p))
            if weight > 0
        )
        for index, p in zip(lower, probability, strict=True)
    ]
    combinations = list(product(*options))
    codes = np.asarray([[entry[0] for entry in row] for row in combinations])
    weights = np.asarray([np.prod([entry[1] for entry in row]) for row in combinations])
    return codes, weights


def enumerate_persistent(
    arrays: dict, subject: int, codebook, *, eta, gain, epsilon
) -> dict:
    relations = relation_support(arrays, subject, codebook)
    codes, weights = code_combinations(
        relations["lower"], relations["probability"], codebook
    )
    cues = arrays["support_cues"][:, subject].astype(np.float64)
    query = arrays["query_cues"][subject].astype(np.float64)
    width = cues.shape[-1] // 2
    history = score_history(
        (cues[:, :width] - cues[:, width:])[None],
        arrays["signed"][:, subject][None],
        arrays["retention"][:, subject][None],
        (query[:, :width] - query[:, width:])[None],
        eta=eta,
        gain=gain,
        epsilon=epsilon,
    )
    sensitivity = history["sensitivity"][0]
    effect = relations["projection"].T @ sensitivity
    margins = codes @ effect
    mean = weights @ margins
    centered = margins - mean
    covariance = centered.T @ (weights[:, None] * centered)
    predicted = effect.T @ (relations["variance"][:, None] * effect)
    trial_variance = np.square(relations["projection"]) @ relations["variance"]
    return {
        "codes": codes,
        "weights": weights,
        "margins": margins,
        "mean": mean,
        "exact_margin": history["global_margin"][0],
        "covariance": covariance,
        "analytic_covariance": predicted,
        "resampled_covariance": sensitivity.T @ (trial_variance[:, None] * sensitivity),
        "projection": relations["projection"],
    }
