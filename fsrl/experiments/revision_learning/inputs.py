"""One deterministic stream shared by both learners; no strategy supervision."""

import numpy as np

from fsrl.experiments.duplicate_observation.inputs import duplicate_observation
from fsrl.experiments.memory_structure.inputs import generator, prepare_shared
from fsrl.experiments.observation_uncertainty.inputs import encode
from fsrl.experiments.relational_revision.inputs import make_panel
from fsrl.experiments.training_strategy.batches import EpisodeBatch, sample_episodes


def generic(rng, count, recipe, noise_seed, validation=False):
    episodes = sample_episodes(generator(recipe), rng, count, validation=validation)
    cpu = prepare_shared(episodes)
    cpu = duplicate_observation(
        encode(cpu, "noisy", recipe["observation"]["sigma"], seed=noise_seed)
    )
    a = cpu.arrays
    query = a["query_inputs"].reshape(2, -1, count, 38).transpose(1, 0, 2, 3)
    return {
        "support": a["support_inputs"],
        "query": query,
        "signs": (2 * a["targets"] - 1).reshape(-1, count).T[:, None],
        "prefixes": np.asarray([len(a["support_inputs"])]),
    }


def training_batch(seed, step, spec):
    count = spec["training"]["batch_size"]
    batch_seed = 2000000000 + seed * 10000 + step
    rng = np.random.default_rng(batch_seed)
    if step % 2 == 0:
        panel = generic(rng, count, spec["ranking_recipe"], batch_seed + 100000000)
    else:
        history = int(rng.integers(2))
        condition = str(rng.choice(["stable", "outlier", "revision"]))
        shift = float(rng.choice([-1, 1]) * rng.uniform(0.2, 0.6))
        raw = make_panel(batch_seed, count, history, "chain", condition, shift=shift)
        prefix = int(rng.integers(24, 33))
        signs = raw["old_sign"] if prefix == 24 else raw["final_sign"]
        panel = {
            "support": raw["support"][:prefix],
            "query": raw["query"],
            "signs": signs[:, None],
            "prefixes": np.asarray([prefix]),
        }
    # Uniform query subsampling controls memory; no learned/remote prioritization.
    indices = rng.choice(len(panel["query"]), 8, replace=False)
    panel["query"] = panel["query"][indices]
    panel["signs"] = panel["signs"][:, :, indices]
    return panel


def fingerprint(panel):
    return EpisodeBatch(panel).fingerprint()
