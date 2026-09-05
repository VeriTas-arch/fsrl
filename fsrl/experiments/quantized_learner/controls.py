"""Cue-bound, distribution-preserving teaching-route control."""

import numpy as np

from fsrl.experiments.minimal_learner.data import ModelBatch

from .encoding import canonical_addresses


def teaching_route(batch: ModelBatch, rng, blocks: int) -> np.ndarray:
    """One relation permutation per subject, held fixed across support blocks.

    Resampled code values may vary by block; the donor address does not. Each
    block therefore preserves its own admitted canonical-value multiset without
    adding an extra change to cross-block code persistence.
    """
    keys, _ = canonical_addresses(batch.arrays["support_cues"])
    trials, subjects = keys.shape
    if trials % blocks:
        raise ValueError("support must contain complete blocks")
    length = trials // blocks
    route = np.broadcast_to(np.arange(trials)[:, None], keys.shape).copy()
    retained = batch.arrays["retention"] == 1
    for subject in range(subjects):
        admitted = np.unique(keys[:, subject][retained[:, subject]])
        donor = dict(zip(admitted, rng.permutation(admitted), strict=True))
        for block in range(blocks):
            indices = np.arange(block * length, (block + 1) * length)
            positions = dict(zip(keys[indices, subject], indices, strict=True))
            if len(positions) != length:
                raise ValueError("one occurrence per relation per block is required")
            for key, source in donor.items():
                target, origin = positions[key], positions[source]
                if not retained[target, subject] or not retained[origin, subject]:
                    raise ValueError("admission must remain stable across blocks")
                route[target, subject] = origin
    return route


def shuffled_teaching(batch: ModelBatch, route: np.ndarray) -> ModelBatch:
    _, orientation = canonical_addresses(batch.arrays["support_cues"])
    canonical = orientation * batch.arrays["signed"]
    signed = orientation * np.take_along_axis(canonical, route, axis=0)
    return ModelBatch({**batch.arrays, "signed": signed})
