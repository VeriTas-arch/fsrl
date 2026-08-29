"""Deterministic cue sampling and relation-retention views."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

import numpy as np

if TYPE_CHECKING:
    from .frozen_fast_weight import FrozenFastWeightEvaluator


def retained_relation_mask(
    evaluator: FrozenFastWeightEvaluator,
    relations: Sequence[tuple[int, int]],
) -> np.ndarray:
    """Return relation-by-subject retention under the evaluator's encoding."""

    if evaluator.subject_relation_gains is None:
        return np.ones((len(relations), evaluator.config.bs), dtype=bool)
    return np.asarray(
        [
            [
                evaluator.subject_relation_gains[subject][relation] > 0.0
                for subject in range(evaluator.config.bs)
            ]
            for relation in relations
        ],
        dtype=bool,
    )


def deterministic_cue_codes(
    n_subjects: int,
    n_items: int,
    cue_size: int,
    seed: int,
    *,
    mode: str = "shared",
) -> np.ndarray:
    """Generate a shared cue set with optional subject-specific item mappings."""

    if cue_size > 20:
        raise ValueError("cue_size > 20 is not supported by exhaustive code generation")
    rng = np.random.default_rng(seed)
    values = np.arange(1 << cue_size, dtype=np.uint32)
    bit_positions = np.arange(cue_size, dtype=np.uint32)
    bits = ((values[:, None] >> bit_positions) & 1).astype(np.int8)
    candidates = (bits * 2 - 1).astype(np.float32)
    for _ in range(100):
        codes: list[np.ndarray] = []
        for candidate_index in rng.permutation(len(candidates)):
            candidate = candidates[int(candidate_index)]
            if all(
                cast(float, np.mean(previous == candidate)) <= 0.66
                for previous in codes
            ):
                codes.append(candidate)
                if len(codes) == n_items:
                    shared = np.stack(codes)
                    if mode == "shared":
                        return np.repeat(shared[None, :, :], n_subjects, axis=0)
                    if mode == "permuted_shared":
                        return np.stack(
                            [
                                shared[rng.permutation(n_items)]
                                for _ in range(n_subjects)
                            ]
                        )
                    raise ValueError(f"unknown cue mode: {mode}")
    raise ValueError(
        f"Could not construct {n_items} sufficiently distinct {cue_size}-bit cues"
    )
