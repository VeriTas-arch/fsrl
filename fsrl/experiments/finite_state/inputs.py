"""Fresh locked input panels reusing the established task and history contracts."""

import copy

import numpy as np

from fsrl.experiments.memory_structure.inputs import generator, liu_inputs
from fsrl.experiments.training_strategy.batches import EpisodeBatch, sample_episodes
from fsrl.experiments.training_strategy.evaluation import write_arrays
from fsrl.experiments.training_strategy.generic_validation import (
    validation_episodes,
    validation_groups,
)
from fsrl.experiments.training_strategy.locks import reference
from fsrl.experiments.write_cost.inputs import history_pair, with_learned

from .protocol import RUNS


def save_input(name, cpu):
    path = RUNS / "inputs" / f"{name}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_arrays(path, cpu.arrays)
    return {"file": reference(path), "fingerprint": cpu.fingerprint()}


def freeze_inputs(spec):
    records = {}
    for split in ("development", "test"):
        config = copy.deepcopy(spec)
        if split == "development":
            config["evaluation"]["generic"].update(
                spec["evaluation"]["generic_development"]
            )
        episodes = validation_episodes(config)
        for length, indices in validation_groups(episodes).items():
            cpu = with_learned(tuple(episodes[i] for i in indices))
            cpu.arrays["episode_indices"] = np.asarray(indices)
            records[f"{split}-{length}"] = save_input(f"{split}-{length}", cpu)
    _, cpu = liu_inputs(spec, 8)
    records["liu-8"] = save_input("liu-8", cpu)
    rng = np.random.default_rng(spec["diagnostic"]["rng_seed"])
    task, histories, index = generator(spec), {}, 0
    while index < spec["diagnostic"]["episodes"]:
        pair = history_pair(sample_episodes(task, rng, 1, validation=True)[0], rng)
        if pair is None:
            continue
        for name, cpu in zip(("novel", "redundant"), pair, strict=True):
            histories.update(
                {f"{index}__{name}__{key}": value for key, value in cpu.arrays.items()}
            )
        index += 1
    records["histories"] = save_input("histories", EpisodeBatch(histories))
    return records


def rounding_seed(panel, replay=0, batch_id=0):
    return 300000000 + panel * 100000 + replay * 10000 + batch_id
