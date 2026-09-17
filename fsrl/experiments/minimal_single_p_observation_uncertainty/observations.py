"""The one registered support-observation intervention."""

from __future__ import annotations

import numpy as np

from fsrl.experiments.training_strategy.batches import EpisodeBatch

CONDITIONS = ("clean", "folded", "noisy")


def epsilon_for(cpu: EpisodeBatch, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(
        cpu.arrays["local_evidence"].shape
    )


def encode(
    cpu: EpisodeBatch,
    condition: str,
    sigma: float,
    epsilon: np.ndarray,
) -> EpisodeBatch:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown observation condition: {condition}")
    arrays = {name: value.copy() for name, value in cpu.arrays.items()}
    base = arrays["local_evidence"]
    if epsilon.shape != base.shape or not np.isfinite(epsilon).all():
        raise ValueError("observation epsilon shape or values differ")
    if condition == "clean" or sigma == 0:
        return EpisodeBatch(arrays)
    proposal = base.astype(np.float64) + sigma * epsilon
    if condition == "folded":
        proposal = np.sign(arrays["signed_magnitudes"]) * np.abs(proposal)
    q = proposal.astype(np.float32)
    arrays["local_evidence"] = q
    arrays["support_inputs"][:, 0, :, 37] = q
    arrays["support_inputs"][:, 0, :, 34] = arrays["trial_retention"] * q
    return EpisodeBatch(arrays)


def observation_checks(
    clean: EpisodeBatch,
    folded: EpisodeBatch,
    noisy: EpisodeBatch,
) -> dict:
    signed = clean.arrays["signed_magnitudes"]
    folded_q = folded.arrays["local_evidence"]
    noisy_q = noisy.arrays["local_evidence"]
    nonzero = signed != 0
    return {
        "clean_fingerprint": clean.fingerprint(),
        "folded_fingerprint": folded.fingerprint(),
        "noisy_fingerprint": noisy.fingerprint(),
        "nonzero_displayed_relations": bool(nonzero.all()),
        "absolute_amplitudes_equal": bool(
            np.array_equal(np.abs(folded_q), np.abs(noisy_q))
        ),
        "folded_signs_restored": bool(
            np.array_equal(np.sign(folded_q[nonzero]), np.sign(signed[nonzero]))
        ),
        "noisy_sign_errors": int(
            np.count_nonzero(np.sign(noisy_q[nonzero]) != np.sign(signed[nonzero]))
        ),
        "presentations": int(nonzero.sum()),
    }


__all__ = ["CONDITIONS", "encode", "epsilon_for", "observation_checks"]
