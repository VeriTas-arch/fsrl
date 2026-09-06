"""Locked generic selection/test sets and a paired direct-exposure diagnostic."""

import copy

import numpy as np

from fsrl.experiments.memory_structure.inputs import (
    generator,
    liu_inputs,
    prepare_shared,
)
from fsrl.experiments.training_strategy.batches import EpisodeBatch, sample_episodes
from fsrl.experiments.training_strategy.evaluation import write_arrays
from fsrl.experiments.training_strategy.generic_validation import (
    validation_episodes,
    validation_groups,
)
from fsrl.experiments.training_strategy.locks import reference, verify_reference

from .protocol import RUNS


def with_learned(episodes):
    cpu = prepare_shared(episodes)
    cpu.arrays["learned"] = np.asarray(
        [
            [
                tuple(sorted((q.left_item, q.right_item)))
                in {
                    tuple(sorted((t.left_item, t.right_item))) for t in e.support_trials
                }
                for q in e.query_trials
            ]
            for e in episodes
        ]
    )
    return cpu


def save_input(name, cpu):
    path = RUNS / "inputs" / f"{name}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_arrays(path, cpu.arrays)
    return {"file": reference(path), "fingerprint": cpu.fingerprint()}


def load_input(record):
    with np.load(verify_reference(record["file"]), allow_pickle=False) as raw:
        cpu = EpisodeBatch({key: raw[key] for key in raw.files})
    if cpu.fingerprint() != record["fingerprint"]:
        raise RuntimeError("input fingerprint differs")
    return cpu


def copy_trial(arrays, target, source):
    time = arrays["support_inputs"][target, :, :, 32].copy()
    for name in (
        "support_inputs",
        "local_evidence",
        "signed_magnitudes",
        "retention",
        "probabilities",
        "support_pairs",
    ):
        arrays[name][target] = arrays[name][source]
    arrays["support_inputs"][target, :, :, 32] = time


def internal_bridge(pairs, index):
    """Removing this edge separates two groups with remote queries on each side."""
    first, second = map(int, pairs[index])
    edges = [tuple(map(int, pair)) for i, pair in enumerate(pairs) if i != index]
    reached = {first}
    previous = set()
    while previous != reached:
        previous = set(reached)
        for left, right in edges:
            if left in reached or right in reached:
                reached.update((left, right))
    return second not in reached and 2 <= len(reached) <= 6


def history_pair(episode, rng):
    cpu = with_learned((episode,))
    edges = len(episode.graph_rank_pairs)
    retained = [
        int(i)
        for i in np.flatnonzero(cpu.arrays["retention"][:edges, 0])
        if internal_bridge(cpu.arrays["support_pairs"][:edges, 0], i)
    ]
    if not len(retained):
        return None
    selected = int(rng.choice(retained))
    relation = set(cpu.arrays["support_pairs"][selected, 0])
    second = next(
        i
        for i in range(edges, 2 * edges)
        if set(cpu.arrays["support_pairs"][i, 0]) == relation
    )
    redundant = copy.deepcopy(cpu.arrays)
    # Swap observations in block two but keep the time of each slot unchanged.
    original = copy.deepcopy(redundant)
    copy_trial(redundant, edges, second)
    for name in (
        "support_inputs",
        "local_evidence",
        "signed_magnitudes",
        "retention",
        "probabilities",
        "support_pairs",
    ):
        redundant[name][second] = original[name][edges]
    redundant["support_inputs"][second, :, :, 32] = original["support_inputs"][
        second, :, :, 32
    ]
    if second == edges:
        redundant = original
    copy_trial(redundant, selected, edges)
    novel = copy.deepcopy(redundant)
    donors = [i for i in range(edges) if set(novel["support_pairs"][i, 0]) != relation]
    copy_trial(novel, selected, int(rng.choice(donors)))
    pairs = redundant["query_pairs"][:, 0]
    direct = np.asarray([set(pair) == relation for pair in pairs])
    remote = np.asarray([not set(pair).intersection(relation) for pair in pairs])
    for arrays in (novel, redundant):
        arrays["target_index"] = np.asarray(edges)
        arrays["direct"] = direct
        arrays["remote"] = remote
    return EpisodeBatch(novel), EpisodeBatch(redundant)


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
    for size in (8, 6, 10):
        _, cpu = liu_inputs(spec, size)
        records[f"liu-{size}"] = save_input(f"liu-{size}", cpu)
    rng = np.random.default_rng(spec["diagnostic"]["rng_seed"])
    task = generator(spec)
    index = 0
    histories = {}
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
