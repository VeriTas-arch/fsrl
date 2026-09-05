"""The full fixed cohort matrix, independent reconstruction and no tuning."""

import numpy as np

from fsrl.analysis.behavioral import analyze_sampled_query_policy
from fsrl.analysis.policy import bundle_logits
from fsrl.experiments.minimal_learner.data import ModelBatch
from fsrl.experiments.minimal_learner.locks import validate_complete
from fsrl.experiments.minimal_learner.training import compiled, runtime
from fsrl.experiments.quantized_learner.analysis import readout
from fsrl.experiments.quantized_learner.encoding import encode_batch
from fsrl.experiments.quantized_learner.evaluation import load_model
from fsrl.experiments.quantized_learner.protocol import resolved_specification
from fsrl.experiments.quantized_learner.reference import rollout
from fsrl.experiments.quantized_learner.verification import reconstruct_codes
from fsrl.experiments.training_strategy.behavior import human_references
from fsrl.experiments.training_strategy.evaluation import json_ready, write_arrays
from fsrl.experiments.training_strategy.locks import reference, verify_reference
from fsrl.infra.provenance import load_json, write_json_exclusive
from fsrl.infra.run_manifest import ProspectiveRun
from fsrl.tasks.protocol import ordered_pairs
from fsrl.tasks.protocol_catalog import load_registered_protocol

from .inputs import load_cohorts, read_arrays, shard_indices
from .locks import LOCK, validate_lock
from .protocol import (
    RUN_ROOT,
    cohort_settings,
    load_parameters,
    parameter_records,
    specification,
)
from .statistics import cohort_record

CODEBOOK = [-1, -1 / 3, 1 / 3, 1]


def analyze(margins: np.ndarray, index: int) -> dict:
    spec = cohort_settings(index)
    settings = spec["evaluation"]["liu"]
    protocol = load_registered_protocol(settings["protocol_id"])
    schedules = (ordered_pairs(protocol.n_items),) * len(margins)
    behavior = analyze_sampled_query_policy(
        protocol,
        bundle_logits({"logits": margins}, schedules),
        seed=settings["choice_seed"],
        temperature=settings["temperature"],
    )
    return cohort_record(behavior, human_references(spec))


def validate_shard(record: dict, input_ref: dict, start: int) -> dict:
    if (
        record["execution_lock"] != reference(LOCK)
        or record["input"] != input_ref
        or record["parameters"] != parameter_records()
    ):
        raise RuntimeError("output shard does not belong to this locked execution")
    if [row["cohort"] for row in record["points"]] != shard_indices(start):
        raise RuntimeError("output shard cohort inventory differs")
    if any(set(row["fits"]) != set(record["parameters"]) for row in record["points"]):
        raise RuntimeError("output shard omits a mandatory fit")
    verify_reference(record["arrays"])
    return record


def evaluated_shard(input_ref: dict, start: int) -> dict:
    directory = RUN_ROOT / "evaluation" / f"cohorts-{start:03d}"
    validate_complete(directory)
    return validate_shard(load_json(directory / "result.json"), input_ref, start)


def evaluate_shard(input_ref, start, models, runners, execution) -> None:
    directory = RUN_ROOT / "evaluation" / f"cohorts-{start:03d}"
    if directory.exists():
        evaluated_shard(input_ref, start)
        return
    with ProspectiveRun.start(
        directory,
        workflow_id="resampled_cohort_diagnostic_v1",
        execution_id=f"cohorts-{start:03d}",
        producer={"module": __name__, "execution_lock": reference(LOCK)},
        resolved_config=execution,
    ):
        points, outputs = [], []
        for index, batch, uniforms in load_cohorts(input_ref, start):
            encoded, _ = encode_batch(batch, "resampled", uniforms, CODEBOOK)
            values = {
                key: readout(model, runners[key], encoded)
                for key, model in models.items()
            }
            points.append(
                {
                    "cohort": index,
                    "fits": {
                        key: analyze(value["margins"], index)
                        for key, value in values.items()
                    },
                }
            )
            outputs.append(values)
        arrays = {
            name: np.stack(
                [np.stack([row[key][name] for key in models]) for row in outputs]
            )
            for name in ("w", "margins")
        }
        arrays["cohort_indices"] = np.asarray(shard_indices(start), dtype=np.int64)
        arrays["fit_seeds"] = np.asarray(list(models), dtype=np.int64)
        path = directory / "outputs.npz"
        write_arrays(path, arrays)
        write_json_exclusive(
            directory / "result.json",
            json_ready(
                {
                    "execution_lock": reference(LOCK),
                    "input": input_ref,
                    "parameters": parameter_records(),
                    "arrays": reference(path),
                    "points": points,
                }
            ),
        )


def evaluate() -> dict:
    lock = validate_lock()
    execution = runtime()
    spec = resolved_specification()
    models = {
        key: load_model(config, spec) for key, config in load_parameters().items()
    }
    runners = {key: compiled(model) for key, model in models.items()}
    size = specification()["cohorts"]["shard_size"]
    for offset, input_ref in enumerate(lock["cohort_shards"]):
        start = offset * size
        evaluate_shard(input_ref, start, models, runners, execution)
        evaluated_shard(input_ref, start)
        print(
            f"Completed cohorts {start}..{start + size - 1} for all three fits",
            flush=True,
        )
    return {
        "completed_cohorts_per_fit": lock["cohort_count"],
        "fits": list(models),
        "training_performed": False,
    }


def verify_shard(record: dict, input_ref: dict, start: int, configs: dict) -> dict:
    validate_shard(record, input_ref, start)
    arrays = read_arrays(record["arrays"])
    np.testing.assert_array_equal(arrays["cohort_indices"], shard_indices(start))
    np.testing.assert_array_equal(arrays["fit_seeds"], [int(key) for key in configs])
    error = 0.0
    for position, (index, batch, uniforms) in enumerate(load_cohorts(input_ref, start)):
        signed, _, _ = reconstruct_codes(batch, uniforms, "resampled", CODEBOOK)
        encoded = ModelBatch({**batch.arrays, "signed": signed})
        for column, (key, config) in enumerate(configs.items()):
            parameters = config["physical_parameters"]
            expected = rollout(
                encoded.arrays,
                eta=parameters["eta"],
                gain=parameters["gamma_G"],
                epsilon=resolved_specification()["model"]["epsilon"],
            )
            for name in ("w", "margins"):
                actual = arrays[name][position, column]
                np.testing.assert_allclose(actual, expected[name], atol=1e-5, rtol=1e-4)
                error = max(error, float(np.max(np.abs(actual - expected[name]))))
            point = json_ready(analyze(arrays["margins"][position, column], index))
            if point != record["points"][position]["fits"][key]:
                raise RuntimeError("original behavior points/flags do not reconstruct")
    return {
        "passed": True,
        "cohorts": len(record["points"]),
        "max_recurrence_error": error,
    }
