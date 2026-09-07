"""Remove only the hint after constructing the unchanged physical observation."""

import numpy as np

from fsrl.experiments.observation_uncertainty.evaluation import observed as original
from fsrl.experiments.training_strategy.batches import EpisodeBatch

HINT = 34


def remove_hint(cpu):
    arrays = dict(cpu.arrays)
    support = arrays["support_inputs"].copy()
    assert support.shape[-1] == 38
    assert not np.any(arrays["query_inputs"][..., HINT])
    support[..., HINT] = 0
    arrays["support_inputs"] = support
    candidate = EpisodeBatch(arrays)
    assert np.array_equal(support[:, 0, :, 37], arrays["local_evidence"])
    return candidate


def observed(cpu, arm, spec, *, keep_hint=False):
    encoded = original(cpu, arm, spec)
    return encoded if keep_hint else remove_hint(encoded)
