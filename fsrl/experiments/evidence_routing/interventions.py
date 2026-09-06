"""Shuffle each evidence representation with one common address permutation."""

from fsrl.evaluation.local_access import apply_blockwise_route
from fsrl.experiments.local_fidelity.evidence_access_pilot import blockwise_derangements
from fsrl.experiments.training_strategy.batches import EpisodeBatch


def shuffle_evidence(cpu: EpisodeBatch, blocks: int, seed: int) -> tuple:
    arrays = {key: value.copy() for key, value in cpu.arrays.items()}
    trials, subjects = arrays["local_evidence"].shape
    route = blockwise_derangements(subjects, blocks, trials // blocks, seed)
    for channel in (34, 37):
        values = arrays["support_inputs"][:, 0, :, channel].T
        arrays["support_inputs"][:, 0, :, channel] = apply_blockwise_route(
            values, route
        ).T
    arrays["local_evidence"] = apply_blockwise_route(
        arrays["local_evidence"].T, route
    ).T
    return EpisodeBatch(arrays), route
