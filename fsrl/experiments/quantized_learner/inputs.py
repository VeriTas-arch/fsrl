"""Pretraining evaluation instances and paired random streams, without rollout."""

import numpy as np

from fsrl.experiments.minimal_learner.data import ModelBatch, generic_batch, liu_batch
from fsrl.experiments.training_strategy.evaluation import write_arrays
from fsrl.experiments.training_strategy.generic_validation import (
    validation_episodes,
    validation_groups,
)
from fsrl.experiments.training_strategy.locks import reference, verify_reference
from fsrl.infra.provenance import write_json_exclusive

from .controls import teaching_route
from .protocol import RECORDS

INPUT_MANIFEST = RECORDS / "benchmarks/evaluation_inputs.json"


def save_evaluation_inputs(spec: dict) -> dict:
    settings = spec["evaluation"]["liu"]
    destination = RECORDS / "inputs"
    destination.mkdir(parents=True, exist_ok=True)
    episodes = validation_episodes(spec)
    _, liu = liu_batch(spec)
    batches = {
        "generic": {
            str(length): (
                generic_batch(tuple(episodes[index] for index in indices)),
                np.asarray(indices),
            )
            for length, indices in sorted(validation_groups(episodes).items())
        },
        "liu": {"all": (liu, np.arange(settings["subjects"]))},
    }
    records = {}
    for domain, groups in batches.items():
        # Same frozen seed, reset independently per cohort; no fit-specific draws.
        encoding_rng = np.random.default_rng(settings["encoding_seed"])
        routing_rng = np.random.default_rng(settings["routing_seed"])
        records[domain] = {}
        for name, (batch, indices) in groups.items():
            arrays = {
                **{f"input__{key}": value for key, value in batch.arrays.items()},
                "subject_indices": indices,
                "encoding_uniforms": encoding_rng.random(batch.arrays["signed"].shape),
                "teaching_route": teaching_route(
                    batch, routing_rng, spec["task"]["support_blocks"]
                ),
            }
            path = destination / f"{domain}-{name}.npz"
            write_arrays(path, arrays)
            records[domain][name] = {
                "arrays": reference(path),
                "batch_sha256": batch.fingerprint(),
            }
    result = {
        "cohorts": records,
        "evaluation": spec["evaluation"],
        "randomness": "Encoding and routing seeds reset per cohort; sorted-length generic groups; all fits share saved arrays. Historical choice sampling resets the registered choice seed per fit.",
        "model_rollout_performed": False,
    }
    write_json_exclusive(INPUT_MANIFEST, result)
    return result


def load_group(record: dict) -> tuple[ModelBatch, dict]:
    with np.load(verify_reference(record["arrays"]), allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    batch = ModelBatch(
        {
            key.removeprefix("input__"): value
            for key, value in arrays.items()
            if key.startswith("input__")
        }
    )
    if batch.fingerprint() != record["batch_sha256"]:
        raise RuntimeError("saved evaluation inputs changed")
    return batch, {
        key: value for key, value in arrays.items() if not key.startswith("input__")
    }
