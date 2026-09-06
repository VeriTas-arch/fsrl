"""Prospective physical inputs, with paired repeat and order transformations."""

from dataclasses import replace

import numpy as np

from fsrl.experiments.adaptive_plasticity.data import clustered_batch, relation_slots
from fsrl.experiments.minimal_learner.data import ModelBatch, generic_batch, liu_batch
from fsrl.experiments.minimal_learner.protocol import task_generator
from fsrl.experiments.training_strategy.batches import sample_episodes
from fsrl.experiments.training_strategy.locks import reference, verify_reference
from fsrl.infra.provenance import load_json, write_json_exclusive
from fsrl.tasks.protocol import RankingProtocol

from .protocol import RUN_ROOT, cohort_specification, specification

INPUT_LOCK = RUN_ROOT / "input_lock.json"


def save_arrays(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    return reference(path)


def save_batch(path, batch, uniforms, **extra):
    return save_arrays(
        path,
        **{f"input__{key}": value for key, value in batch.arrays.items()},
        uniforms=uniforms,
        **extra,
    )


def read_batch(record) -> tuple:
    with np.load(verify_reference(record), allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    return ModelBatch(
        {
            key.removeprefix("input__"): value
            for key, value in arrays.items()
            if key.startswith("input__")
        }
    ), {key: value for key, value in arrays.items() if not key.startswith("input__")}


def prefix(batch: ModelBatch, uniforms, repeats: int) -> tuple:
    relations = relation_slots(batch.arrays["support_cues"]).max(axis=0) + 1
    if len(set(relations.tolist())) != 1:
        raise ValueError("repeat fixtures require equal relation counts")
    trials = int(relations[0]) * repeats
    arrays = dict(batch.arrays)
    for key in (
        "support_cues",
        "signed",
        "retention",
        "local_evidence",
        "probabilities",
    ):
        arrays[key] = arrays[key][:trials]
    arrays["support_pairs"] = arrays["support_pairs"][:, :trials]
    return ModelBatch(arrays), uniforms[:trials]


def reverse_groups(batch: ModelBatch, uniforms, repeats=4) -> tuple:
    trials = batch.arrays["signed"].shape[0]
    permutation = np.arange(trials).reshape(-1, repeats)[::-1].reshape(-1)
    arrays = dict(batch.arrays)
    for key in (
        "support_cues",
        "signed",
        "retention",
        "local_evidence",
        "probabilities",
    ):
        arrays[key] = arrays[key][permutation]
    arrays["support_pairs"] = arrays["support_pairs"][:, permutation]
    return ModelBatch(arrays), uniforms[permutation]


def manipulation_batch(rng, count: int) -> ModelBatch:
    """A predeclared chain, with a unique bridge and ordinary random item cues."""
    generator = task_generator()
    graph = tuple(tuple(edge) for edge in specification()["manipulations"]["graph"])
    episodes = []
    for _ in range(count):
        episode = generator.sample(rng)
        order = episode.true_order_high_to_low
        pairs = tuple((order[a], order[b]) for a, b in graph)
        protocol = RankingProtocol(
            "structure-chain", tuple(map(str, range(8))), order, pairs, 8, 1, {}
        )
        admission = {
            pair: float(
                rng.random() < episode.subject_encoding.relation_reliability(*pair, 1)
            )
            for pair in pairs
        }
        trials = tuple(
            replace(t, encoding_reliability=admission[(t.higher_item, t.lower_item)])
            for t in protocol.support_schedule(rng)
        )
        episodes.append(replace(episode, graph_rank_pairs=graph, support_trials=trials))
    return generic_batch(tuple(episodes))


def schedule_variants(batch, uniforms, seed: int):
    variants = {f"balanced_K{k}": prefix(batch, uniforms, k) for k in (2, 4, 8)}
    clustered = clustered_batch(
        variants["balanced_K4"][0],
        variants["balanced_K4"][1],
        np.random.default_rng(seed),
    )
    variants["clustered_K4"] = clustered[:2]
    variants["reverse_clustered_K4"] = reverse_groups(*clustered[:2])
    return variants


def prepare_inputs() -> dict:
    from .execution import validate_source

    validate_source()
    if INPUT_LOCK.exists():
        return load_json(INPUT_LOCK)
    settings = specification()
    dest = RUN_ROOT / "inputs"
    rng = np.random.default_rng(settings["design"]["rng"]["generic"])
    encoder = np.random.default_rng(settings["design"]["rng"]["generic_encoding"])
    groups = []
    for group in range(settings["design"]["generic_episodes"] // 32):
        batch = generic_batch(
            sample_episodes(task_generator(), rng, 32, validation=True)
        )
        groups.append(
            save_batch(
                dest / f"generic-{group}.npz",
                batch,
                encoder.random(batch.arrays["signed"].shape),
            )
        )
    cohorts = []
    for index in range(settings["design"]["cohorts"]):
        spec = cohort_specification(index)
        _, batch = liu_batch(spec)
        uniforms = np.random.default_rng(
            spec["evaluation"]["liu"]["encoding_seed"]
        ).random(batch.arrays["signed"].shape)
        cohorts.append(save_batch(dest / f"liu-{index:03}.npz", batch, uniforms))
    rng = np.random.default_rng(settings["manipulations"]["seed"])
    batch = manipulation_batch(rng, settings["manipulations"]["episodes"])
    uniforms = rng.random(batch.arrays["signed"].shape)
    manipulations = {
        name: save_batch(dest / f"manipulation-{name}.npz", b, u)
        for name, (b, u) in schedule_variants(
            batch, uniforms, settings["manipulations"]["seed"] + 1
        ).items()
    }
    # Recovery uses a separate chain fixture, with no generating latents exposed to decoding.
    rng = np.random.default_rng(settings["recovery"]["seed"])
    batch = manipulation_batch(rng, settings["recovery"]["episodes_per_dataset"])
    uniforms = rng.random(batch.arrays["signed"].shape)
    recovery = {
        name: save_batch(dest / f"recovery-{name}.npz", b, u)
        for name, (b, u) in schedule_variants(
            batch, uniforms, settings["recovery"]["seed"] + 1
        ).items()
        if name in settings["recovery"]["designs"]
    }
    result = {
        "generic": groups,
        "liu": cohorts,
        "manipulations": manipulations,
        "recovery": recovery,
        "candidate_evaluation_performed": False,
        "scope": "Prospectively generated physical inputs; no candidate selection.",
    }
    write_json_exclusive(INPUT_LOCK, result)
    return result


def load_inputs() -> dict:
    result = load_json(INPUT_LOCK)
    records = (
        result["generic"]
        + result["liu"]
        + list(result["manipulations"].values())
        + list(result["recovery"].values())
    )
    for row in records:
        verify_reference(row)
    return result
