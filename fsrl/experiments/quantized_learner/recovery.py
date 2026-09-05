"""Joint-choice recovery mathematics, with nuisance states hidden from decoding.

No execution or record writes on import. The prospective runner must source-lock
before using these functions for the registered recovery screen.
"""

from itertools import product

import numpy as np
from scipy.special import expit, logsumexp
from scipy.stats import norm

from fsrl.experiments.minimal_learner.data import ModelBatch

from .encoding import encode_batch
from .recovery_inputs import ObservedDesign, nuisance_pool
from .reference import rollout

CONDITIONS = ("exact", "persistent", "resampled")


def generate_choices(design, rng, settings, codebook, *, temperature, epsilon):
    """One independently drawn subject state per family/parameter setting."""
    grid = list(product(settings["eta_grid"], settings["gain_grid"]))
    size = len(CONDITIONS) * len(grid)
    batch, witness = nuisance_pool(design, rng, size)
    margins = np.empty((size, len(design.query_cues)))
    weights = np.empty((size, design.query_cues.shape[-1] // 2))
    internal = np.empty_like(batch.arrays["signed"])
    indices = np.empty(internal.shape, dtype=np.int8)
    for family, condition in enumerate(CONDITIONS):
        encoded, encoding = encode_batch(
            batch, condition, witness["encoding_uniforms"], codebook
        )
        for parameter, (eta, gain) in enumerate(grid):
            subject = family * len(grid) + parameter
            single = {
                key: value[subject : subject + 1]
                if key == "query_cues"
                else value[:, subject : subject + 1]
                for key, value in encoded.arrays.items()
            }
            result = rollout(single, eta=eta, gain=gain, epsilon=epsilon)
            margins[subject] = result["margins"][0]
            weights[subject] = result["w"][0]
            internal[:, subject] = encoded.arrays["signed"][:, subject]
            indices[:, subject] = encoding["code_indices"][:, subject]
    uniforms = rng.random((size, settings["choice_repetitions"], margins.shape[-1]))
    choices = uniforms < expit(margins[:, None] / temperature)
    return {
        "left_counts": choices.sum(axis=1),
        "choices": choices,
        "choice_uniforms": uniforms,
        "margins": margins,
        "w": weights,
        "internal_signed": internal,
        "code_indices": indices,
        "retention": batch.arrays["retention"],
        **witness,
    }


def integrated_log_likelihood(
    batch: ModelBatch,
    uniforms,
    counts,
    settings,
    codebook,
    *,
    temperature,
    epsilon,
) -> np.ndarray:
    """Per-subject likelihood: [budget, generating setting, family, parameter].

    The batch is an INDEPENDENT prior pool, not a generating-state batch. Every
    choice by a subject shares one pool state; integrate after multiplying all
    query probabilities. Parameters remain shared across subjects (see summary).
    """
    counts = np.asarray(counts)
    repetitions = settings["choice_repetitions"]
    if (
        counts.ndim != 2
        or counts.shape[1] != batch.arrays["query_cues"].shape[1]
        or not np.all(np.isfinite(counts) & (counts >= 0) & (counts <= repetitions))
        or not np.all(counts == np.floor(counts))
        or temperature <= 0
    ):
        raise ValueError("invalid observed choice counts or temperature")
    budgets = (settings["nuisance_draws"], settings["integration_check_draws"])
    if not 0 < budgets[0] <= budgets[1] == batch.arrays["signed"].shape[1]:
        raise ValueError("both integration budgets must share one full prior pool")
    n_gain = len(settings["gain_grid"])
    result = np.empty(
        (2, len(counts), len(CONDITIONS), len(settings["eta_grid"]) * n_gain)
    )
    for family, condition in enumerate(CONDITIONS):
        encoded, _ = encode_batch(batch, condition, uniforms, codebook)
        for e, eta in enumerate(settings["eta_grid"]):
            margins = rollout(encoded.arrays, eta=eta, gain=1, epsilon=epsilon)[
                "margins"
            ]
            for g, gain in enumerate(settings["gain_grid"]):
                logits = gain * margins / temperature
                # State by generating-setting matrix, not query-averaged policy.
                conditional = (
                    -np.logaddexp(0, -logits) @ counts.T
                    - np.logaddexp(0, logits) @ (repetitions - counts).T
                )
                for b, budget in enumerate(budgets):
                    result[b, :, family, e * n_gain + g] = logsumexp(
                        conditional[:budget], axis=0
                    ) - np.log(budget)
    return result


def decode_choices(
    design: ObservedDesign,
    counts,
    integration_rng,
    settings,
    codebook,
    *,
    temperature,
    epsilon,
):
    """Public decoder boundary accepts observations only, never generation data."""
    batch, witness = nuisance_pool(
        design, integration_rng, settings["integration_check_draws"]
    )
    values = integrated_log_likelihood(
        batch,
        witness["encoding_uniforms"],
        counts,
        settings,
        codebook,
        temperature=temperature,
        epsilon=epsilon,
    )
    fingerprints = {
        "nuisance_batch_sha256": batch.fingerprint(),
        "nuisance_draws_sha256": ModelBatch(witness).fingerprint(),
    }
    return values, fingerprints


def recovery_summary(per_episode: np.ndarray) -> dict:
    """Marginalize a single shared parameter after summing episode log scores.

    Axes: episode, budget, generating setting, candidate family, parameter.
    Exact score ties do not identify a family and count as recovery failures.
    """
    values = np.asarray(per_episode)
    if values.ndim != 5 or values.shape[1] != 2 or values.shape[3] != 3:
        raise ValueError("recovery requires both budgets and all three families")
    n_parameter = values.shape[-1]
    if values.shape[2] != 3 * n_parameter or not np.all(np.isfinite(values)):
        raise ValueError("recovery requires every generating setting and finite scores")
    joint = values.sum(axis=0)
    family_scores = logsumexp(joint, axis=-1) - np.log(n_parameter)
    winners = np.argmax(family_scores, axis=-1)
    unique = (family_scores == family_scores.max(axis=-1, keepdims=True)).sum(-1) == 1
    winners = np.where(unique, winners, -1)
    truth = np.repeat(np.arange(3), n_parameter)
    confusion = np.zeros((2, 3, 3), dtype=np.int64)
    for budget in range(2):
        for actual, predicted in zip(truth, winners[budget], strict=True):
            if predicted >= 0:
                confusion[budget, actual, predicted] += 1
    diagonal = np.diagonal(confusion, axis1=1, axis2=2)
    p = diagonal / n_parameter
    z = norm.ppf(0.975)
    center = p + z * z / (2 * n_parameter)
    radius = z * np.sqrt(p * (1 - p) / n_parameter + z * z / (4 * n_parameter**2))
    lower = (center - radius) / (1 + z * z / n_parameter)
    stable = bool(np.all(winners[0] == winners[1]) and np.all(unique))
    passed = bool(np.all(lower[0] > 1 / 3) and stable)
    return {
        "joint_parameter_log_scores": joint,
        "family_log_scores": family_scores,
        "family_winners": winners,
        "conditional_parameter_winners": np.argmax(joint, axis=-1),
        "confusion": confusion,
        "unidentified_counts": np.asarray(
            [[np.sum((truth == f) & (row < 0)) for f in range(3)] for row in winners]
        ),
        "diagonal_fraction": p,
        "diagonal_wilson_lower": lower,
        "diagonal_wilson_upper": (center + radius) / (1 + z * z / n_parameter),
        "budget_winners_stable": stable,
        "outcome": "distinguishable_on_registered_screen"
        if passed
        else "specificity_unresolved",
    }
