"""Protected generic and Liu inputs for the fresh replication."""

from __future__ import annotations

import numpy as np

from fsrl.experiments.minimal_learner.data import ModelBatch, generic_batch, liu_batch
from fsrl.experiments.training_strategy.evaluation import write_arrays
from fsrl.experiments.training_strategy.generic_validation import (
    validation_episodes,
    validation_groups,
)
from fsrl.experiments.training_strategy.locks import reference, verify_reference
from fsrl.infra.provenance import write_json_exclusive

from .protocol import (
    COHORT_SHARD_SIZE,
    COHORTS,
    GENERIC_ENCODING_SEED,
    RECORDS,
    cohort_specification,
    resolved_specification,
)

GENERIC_MANIFEST = RECORDS / "benchmarks/generic_inputs.json"
LIU_MANIFEST = RECORDS / "benchmarks/liu_inputs.json"


def _write_batch(path, batch: ModelBatch, uniforms: np.ndarray, **extra) -> dict:
    write_arrays(
        path,
        {
            **{f"input__{key}": value for key, value in batch.arrays.items()},
            "encoding_uniforms": uniforms,
            **extra,
        },
    )
    return {"arrays": reference(path), "batch_sha256": batch.fingerprint()}


def save_generic_inputs() -> dict:
    destination = RECORDS / "inputs"
    destination.mkdir(parents=True, exist_ok=True)
    spec = resolved_specification()
    episodes = validation_episodes(spec)
    groups = validation_groups(episodes)
    encoding_rng = np.random.default_rng(GENERIC_ENCODING_SEED)
    records = {}
    for length, indices in sorted(groups.items()):
        batch = generic_batch(tuple(episodes[index] for index in indices))
        uniforms = encoding_rng.random(batch.arrays["signed"].shape)
        records[str(length)] = _write_batch(
            destination / f"generic-{length}.npz",
            batch,
            uniforms,
            episode_indices=np.asarray(indices, dtype=np.int64),
        )
    result = {
        "experiment_id": "global_decay_replication_v1",
        "episodes": len(episodes),
        "groups": records,
        "model_rollout_performed": False,
        "randomness": {
            "generic_episode_seed": spec["evaluation"]["generic"]["rng_seed"],
            "encoding_seed": GENERIC_ENCODING_SEED,
        },
    }
    write_json_exclusive(GENERIC_MANIFEST, result)
    return result


def read_arrays(record: dict) -> dict[str, np.ndarray]:
    with np.load(verify_reference(record), allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def load_group(record: dict) -> tuple[ModelBatch, dict[str, np.ndarray]]:
    arrays = read_arrays(record["arrays"])
    batch = ModelBatch(
        {
            key.removeprefix("input__"): value
            for key, value in arrays.items()
            if key.startswith("input__")
        }
    )
    if batch.fingerprint() != record["batch_sha256"]:
        raise RuntimeError("saved replication generic batch changed")
    return batch, {
        key: value for key, value in arrays.items() if not key.startswith("input__")
    }


def cohort_indices(start: int) -> list[int]:
    if start not in range(0, COHORTS, COHORT_SHARD_SIZE):
        raise ValueError("unregistered replication cohort shard")
    return list(range(start, start + COHORT_SHARD_SIZE))


def save_liu_inputs() -> dict:
    destination = RECORDS / "inputs"
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    for start in range(0, COHORTS, COHORT_SHARD_SIZE):
        rows = []
        for index in cohort_indices(start):
            spec = cohort_specification(index)
            _, batch = liu_batch(spec)
            seed = spec["evaluation"]["liu"]["encoding_seed"]
            rows.append(
                {
                    **{f"input__{key}": value for key, value in batch.arrays.items()},
                    "encoding_uniforms": np.random.default_rng(seed).random(
                        batch.arrays["signed"].shape
                    ),
                }
            )
        arrays = {key: np.stack([row[key] for row in rows]) for key in rows[0]}
        arrays["cohort_indices"] = np.asarray(cohort_indices(start), dtype=np.int64)
        path = destination / f"liu-cohorts-{start:03d}.npz"
        write_arrays(path, arrays)
        records.append(reference(path))
    result = {
        "experiment_id": "global_decay_replication_v1",
        "cohorts": COHORTS,
        "subjects": cohort_specification(0)["evaluation"]["liu"]["subjects"],
        "shards": records,
        "model_rollout_performed": False,
    }
    write_json_exclusive(LIU_MANIFEST, result)
    return result


def load_cohorts(record: dict, start: int) -> list[tuple[int, ModelBatch, np.ndarray]]:
    arrays = read_arrays(record)
    np.testing.assert_array_equal(arrays["cohort_indices"], cohort_indices(start))
    return [
        (
            int(index),
            ModelBatch(
                {
                    key.removeprefix("input__"): value[position]
                    for key, value in arrays.items()
                    if key.startswith("input__")
                }
            ),
            arrays["encoding_uniforms"][position],
        )
        for position, index in enumerate(arrays["cohort_indices"])
    ]
