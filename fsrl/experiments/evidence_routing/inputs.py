"""Remove only weak global evidence while retaining duplicated strong inputs."""

import numpy as np

from fsrl.experiments.training_strategy.batches import EpisodeBatch


def route_batch(base: EpisodeBatch, condition: str) -> EpisodeBatch:
    arrays = dict(base.arrays)
    inputs = arrays["support_inputs"].copy()
    if condition == "isolated":
        inputs[:, 0, :, 37] = inputs[:, 0, :, 34]
    elif condition != "shared":
        raise ValueError("unregistered evidence route")
    arrays["support_inputs"] = inputs
    return EpisodeBatch(arrays)


def check_pair(base: EpisodeBatch) -> dict:
    shared, isolated = (route_batch(base, arm) for arm in ("shared", "isolated"))
    a, b = shared.arrays, isolated.arrays
    for key in a:
        if key != "support_inputs":
            np.testing.assert_array_equal(a[key], b[key])
    np.testing.assert_array_equal(
        a["support_inputs"][..., :37], b["support_inputs"][..., :37]
    )
    strong = a["support_inputs"][:, 0, :, 34]
    np.testing.assert_array_equal(b["support_inputs"][:, 0, :, 37], strong)
    weak = a["local_evidence"] - strong
    np.testing.assert_array_equal(
        a["support_inputs"][:, 0, :, 37] - b["support_inputs"][:, 0, :, 37], weak
    )
    assert not a["support_inputs"][:, 1:, :, 37].any()
    assert not a["query_inputs"][..., 37].any()
    assert not weak[strong != 0].any()
    return {"passed": True, "weak_entries": int(np.count_nonzero(weak))}
