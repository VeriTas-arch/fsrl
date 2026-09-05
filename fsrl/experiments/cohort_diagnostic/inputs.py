"""All-cohort input lock; no model rollout while generating instances."""

import numpy as np

from fsrl.experiments.minimal_learner.data import ModelBatch, liu_batch
from fsrl.experiments.training_strategy.evaluation import write_arrays
from fsrl.experiments.training_strategy.locks import reference, verify_reference

from .protocol import RECORDS, cohort_settings, specification


def shard_indices(start: int) -> list[int]:
    settings = specification()["cohorts"]
    size = settings["shard_size"]
    if start not in range(0, settings["count"], size):
        raise ValueError("unregistered cohort shard")
    return list(range(start, start + size))


def save_inputs() -> list[dict]:
    destination = RECORDS / "inputs"
    destination.mkdir(parents=True, exist_ok=False)
    settings = specification()["cohorts"]
    records = []
    for start in range(0, settings["count"], settings["shard_size"]):
        samples = []
        for index in shard_indices(start):
            spec = cohort_settings(index)
            _, batch = liu_batch(spec)
            rng = np.random.default_rng(spec["evaluation"]["liu"]["encoding_seed"])
            samples.append(
                {
                    **{f"input__{key}": value for key, value in batch.arrays.items()},
                    "encoding_uniforms": rng.random(batch.arrays["signed"].shape),
                }
            )
        arrays = {key: np.stack([row[key] for row in samples]) for key in samples[0]}
        arrays["cohort_indices"] = np.asarray(shard_indices(start), dtype=np.int64)
        path = destination / f"cohorts-{start:03d}.npz"
        write_arrays(path, arrays)
        records.append(reference(path))
    return records


def read_arrays(ref: dict) -> dict:
    with np.load(verify_reference(ref), allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def load_cohorts(ref: dict, start: int) -> list[tuple]:
    arrays = read_arrays(ref)
    np.testing.assert_array_equal(arrays["cohort_indices"], shard_indices(start))
    return [
        (
            int(index),
            ModelBatch(
                {
                    key.removeprefix("input__"): values[position]
                    for key, values in arrays.items()
                    if key.startswith("input__")
                }
            ),
            arrays["encoding_uniforms"][position],
        )
        for position, index in enumerate(arrays["cohort_indices"])
    ]
