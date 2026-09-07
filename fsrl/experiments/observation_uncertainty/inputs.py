"""One observed scalar, shared by both memory routes and all paired controls."""

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
from fsrl.experiments.write_cost.inputs import copy_trial, with_learned

from .protocol import RUNS


def observation_seed(panel, replay=0, batch_id=0):
    return 500000000 + panel * 100000 + replay * 10000 + batch_id


def gains(arrays):
    return arrays.get("trial_retention", arrays["retention"])


def encode(cpu, arm, sigma, *, seed=None, replay=0):
    arrays = {key: value.copy() for key, value in cpu.arrays.items()}
    u = arrays["local_evidence"]
    noise = (
        np.random.default_rng(seed).standard_normal(u.shape)
        if seed is not None
        else arrays[f"encoding_noise_{replay}"]
    )
    if arm == "clean" or sigma == 0:
        return EpisodeBatch(arrays)
    proposal = u.astype(float) + sigma * noise
    if arm == "folded":
        proposal = np.sign(arrays["signed_magnitudes"]) * np.abs(proposal)
    elif arm != "noisy":
        raise ValueError("unknown observation condition")
    q = proposal.astype(np.float32)
    arrays["local_evidence"] = q
    arrays["support_inputs"][:, 0, :, 34] = gains(arrays) * q
    arrays["support_inputs"][:, 0, :, 37] = q
    return EpisodeBatch(arrays)


def attach_noise(cpu, panel, batch_id=0, replays=(0,)):
    for replay in replays:
        cpu.arrays[f"encoding_noise_{replay}"] = np.random.default_rng(
            observation_seed(panel, replay, batch_id)
        ).standard_normal(cpu.arrays["local_evidence"].shape)
    return cpu


def connected_without(pairs, relation):
    reached = {int(relation[0])}
    previous = set()
    while previous != reached:
        previous = set(reached)
        for left, right in pairs:
            if {left, right} == set(relation):
                continue
            if left in reached or right in reached:
                reached.update((int(left), int(right)))
    return int(relation[1]) in reached


def history_pair(episode, rng, sigma, index):
    original = with_learned((episode,)).arrays
    edge_count = len(episode.graph_rank_pairs)
    candidates = [
        i
        for i in range(edge_count)
        if original["retention"][i, 0] == 1
        and np.isclose(abs(original["signed_magnitudes"][i, 0]), 1 / 7)
        and connected_without(
            original["support_pairs"][:edge_count, 0], original["support_pairs"][i, 0]
        )
    ]
    if not candidates:
        return None
    selected = int(rng.choice(candidates))
    relation = original["support_pairs"][selected, 0]
    left, right = map(int, relation)
    if original["signed_magnitudes"][selected, 0] < 0:
        left, right = right, left
    donors = [
        i
        for i in range(edge_count)
        if set(original["support_pairs"][i, 0]) != {left, right}
    ]
    supported = copy.deepcopy(original)
    for slot in range(len(supported["support_inputs"]) - 1):
        if set(supported["support_pairs"][slot, 0]) == {left, right}:
            copy_trial(supported, slot, int(rng.choice(donors)))
    # Donor target comes from the original arrays, never an overwritten slot.
    last = len(supported["support_inputs"]) - 1
    for key in (
        "support_inputs",
        "local_evidence",
        "signed_magnitudes",
        "retention",
        "probabilities",
        "support_pairs",
    ):
        supported[key][last] = original[key][selected]
    supported["support_inputs"][last, :, :, 32] = original["support_inputs"][
        last, :, :, 32
    ]
    codes = supported["item_codes"][0]
    cs = codes.shape[1]
    supported["support_pairs"][last, 0] = left, right
    supported["support_inputs"][last, 0, 0, :cs] = codes[left]
    supported["support_inputs"][last, 0, 0, cs : 2 * cs] = codes[right]
    supported["signed_magnitudes"][last, 0] = abs(
        original["signed_magnitudes"][selected, 0]
    )
    supported["local_evidence"][last, 0] = abs(original["local_evidence"][selected, 0])
    supported["support_inputs"][last, 0, 0, 34] = supported["local_evidence"][last, 0]
    supported["support_inputs"][last, 0, 0, 37] = supported["local_evidence"][last, 0]
    cpu = attach_noise(EpisodeBatch(supported), 3, index)
    supported = encode(cpu, "noisy", sigma).arrays
    q = abs(supported["local_evidence"][last, 0])
    supported["local_evidence"][last, 0] = q
    supported["support_inputs"][last, 0, 0, 34] = q
    supported["support_inputs"][last, 0, 0, 37] = q
    conflicting = copy.deepcopy(supported)
    permutation = np.arange(len(codes))
    permutation[left], permutation[right] = right, left
    conflicting["support_pairs"][:last] = permutation[supported["support_pairs"][:last]]
    for slot in range(last):
        pair = conflicting["support_pairs"][slot, 0]
        conflicting["support_inputs"][slot, 0, 0, :cs] = codes[pair[0]]
        conflicting["support_inputs"][slot, 0, 0, cs : 2 * cs] = codes[pair[1]]
    conflicting["orders"] = permutation[supported["orders"]]
    rank = np.argsort(conflicting["orders"][0])
    queries = supported["query_pairs"][:, 0]
    conflicting["targets"] = (rank[queries[:, 0]] < rank[queries[:, 1]]).astype(
        np.int64
    )
    for arrays in (supported, conflicting):
        arrays["target_index"] = np.asarray(last)
        arrays["direct"] = np.asarray([set(pair) == {left, right} for pair in queries])
        arrays["remote"] = np.asarray(
            [not set(pair).intersection((left, right)) for pair in queries]
        )
    return EpisodeBatch(supported), EpisodeBatch(conflicting)


def save_input(name, cpu):
    path = RUNS / "inputs" / f"{name}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_arrays(path, cpu.arrays)
    return {"file": reference(path), "fingerprint": cpu.fingerprint()}


def freeze_inputs(spec):
    records = {}
    for panel, split in enumerate(("development", "test"), 1):
        config = copy.deepcopy(spec)
        if split == "development":
            config["evaluation"]["generic"].update(
                spec["evaluation"]["generic_development"]
            )
        episodes = validation_episodes(config)
        for length, indices in validation_groups(episodes).items():
            cpu = with_learned(tuple(episodes[i] for i in indices))
            cpu.arrays["episode_indices"] = np.asarray(indices)
            records[f"{split}-{length}"] = save_input(
                f"{split}-{length}", attach_noise(cpu, panel, length)
            )
    _, cpu = liu_inputs(spec, 8)
    records["liu-8"] = save_input("liu-8", attach_noise(cpu, 4, replays=(0, 1, 2)))
    task = generator(spec)
    rng = np.random.default_rng(spec["diagnostic"]["rng_seed"])
    histories = {}
    index = 0
    while index < spec["diagnostic"]["episodes"]:
        pair = history_pair(
            sample_episodes(task, rng, 1, validation=True)[0],
            rng,
            spec["observation"]["sigma"],
            index,
        )
        if pair is None:
            continue
        for label, cpu in zip(("supported", "conflicting"), pair, strict=True):
            histories.update(
                {f"{index}__{label}__{key}": value for key, value in cpu.arrays.items()}
            )
        index += 1
    records["histories"] = save_input("histories", EpisodeBatch(histories))
    return records
