"""Paired observation routing and source removal without changing task cues."""

import numpy as np

from fsrl.evaluation.local_access import apply_blockwise_route
from fsrl.experiments.local_fidelity.evidence_access_pilot import blockwise_derangements
from fsrl.experiments.training_strategy.batches import EpisodeBatch


def remove_relation(cpu: EpisodeBatch, relation: tuple) -> EpisodeBatch:
    arrays = {key: value.copy() for key, value in cpu.arrays.items()}
    mask = np.all(
        np.sort(arrays["support_pairs"], axis=-1) == sorted(relation), axis=-1
    )
    arrays["support_inputs"][:, 0, :, 34][mask] = 0
    arrays["support_inputs"][:, 0, :, 37][mask] = 0
    arrays["local_evidence"][mask] = 0
    return EpisodeBatch(arrays)


def shuffle_evidence(cpu: EpisodeBatch, blocks: int, seed: int) -> tuple:
    arrays = {key: value.copy() for key, value in cpu.arrays.items()}
    trials, subjects = arrays["local_evidence"].shape
    route = blockwise_derangements(subjects, blocks, trials // blocks, seed)
    for channel in (34, 37):
        values = arrays["support_inputs"][:, 0, :, channel].T
        arrays["support_inputs"][:, 0, :, channel] = apply_blockwise_route(
            values, route
        ).T
    arrays["local_evidence"] = arrays["support_inputs"][:, 0, :, 37].copy()
    return EpisodeBatch(arrays), route
