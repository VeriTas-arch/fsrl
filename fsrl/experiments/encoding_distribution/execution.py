"""Qualified Gaussian replay with immutable parent E/Q conditions."""

import numpy as np

from fsrl.experiments.encoding_contribution.execution import competence, internal_record
from fsrl.experiments.encoding_contribution.moments import conditional_moments
from fsrl.experiments.encoding_contribution.protocol import parameters
from fsrl.experiments.minimal_learner.data import generic_batch
from fsrl.experiments.minimal_learner.locks import validate_complete
from fsrl.experiments.minimal_learner.protocol import task_generator
from fsrl.experiments.minimal_learner.training import runtime
from fsrl.experiments.structural_identification.evaluation import (
    load_runner,
    predict,
    routing_control,
    scalar_identity,
)
from fsrl.experiments.structural_identification.inputs import save_arrays
from fsrl.experiments.structural_identification.model import reference
from fsrl.experiments.structural_identification.observation import (
    record,
    sample_choices,
)
from fsrl.experiments.structural_identification.protocol import cohort_specification
from fsrl.experiments.training_strategy.batches import sample_episodes
from fsrl.infra.provenance import load_json, write_json_exclusive
from fsrl.infra.run_manifest import ProspectiveRun
from fsrl.tasks.protocol_catalog import load_registered_protocol

from .encoding import gaussian, impulse_moments
from .protocol import (
    RUN_ROOT,
    SEEDS,
    STRUCTURES,
    cell,
    input_batch,
    validate_source,
)


def unit(directory, settings):
    return ProspectiveRun.start(
        directory,
        workflow_id="encoding_distribution_v1",
        execution_id=f"{directory.parent.name}-{directory.name}",
        producer={"module": __name__},
        resolved_config=settings,
    )


def moment_check(base, encoded, eta, gain):
    mean, covariance, _ = conditional_moments(base, eta, gain)
    mapping, independent_mean, independent_cov, gamma = impulse_moments(base, eta, gain)
    q = np.asarray(base.arrays["query_cues"], dtype=float)
    width = q.shape[-1] // 2
    q = q[..., :width] - q[..., width:]
    expected_gamma = gain**2 * q @ covariance @ q.transpose(0, 2, 1)
    state = reference(encoded, eta, gain, "decay")[1]
    independent_state = np.einsum("bit,tb->bi", mapping, encoded.arrays["signed"])
    errors = {}
    for name, a, b in (
        ("mean", mean, independent_mean),
        ("covariance", covariance, independent_cov),
        ("full_query_covariance", expected_gamma, gamma),
        ("realized_state", state, independent_state),
    ):
        np.testing.assert_allclose(a, b, atol=1e-12, rtol=1e-10)
        errors[name] = float(np.max(np.abs(a - b)))
    return errors


def qualify():
    snapshot = runtime()
    rng = np.random.default_rng(8520001)
    base = generic_batch(sample_episodes(task_generator(), rng, 2))
    encoded, tail = gaussian(base, rng.random(base.arrays["signed"].shape))
    results = {}
    for seed in SEEDS:
        for donor in STRUCTURES:
            p = parameters(seed, donor)
            output, state = predict(load_runner(p), encoded)
            expected, expected_state = reference(
                encoded, p["eta0"], p["gamma_G"], "decay"
            )
            np.testing.assert_allclose(output, expected, atol=1e-5, rtol=1e-4)
            np.testing.assert_allclose(state, expected_state, atol=1e-5, rtol=1e-4)
            results[cell(seed, "G", donor)] = {
                "CUDA_margin_error": float(np.max(abs(output - expected))),
                "moments": moment_check(base, encoded, p["eta0"], p["gamma_G"]),
            }
    result = {
        "passed": True,
        "checks": results,
        "synthetic_tail": tail,
        "runtime": snapshot,
    }
    write_json_exclusive(RUN_ROOT / "qualification.json", result)
    return result


def generic():
    runtime()
    validate_source()
    results = {}
    for seed in SEEDS:
        for donor in STRUCTURES:
            name = cell(seed, "G", donor)
            directory = RUN_ROOT / "generic" / name
            if directory.exists():
                validate_complete(directory)
                results[name] = load_json(directory / "result.json")
                continue
            runner = load_runner(parameters(seed, donor))
            with unit(directory, {"cell": name, "parameters": parameters(seed, donor)}):
                data = {key: [] for key in ("margins", "shuffled", "signs", "learned")}
                for group in range(8):
                    base, uniforms = input_batch(f"generic-{group}")
                    encoded, _ = gaussian(base, uniforms)
                    data["margins"].append(predict(runner, encoded)[0])
                    data["shuffled"].append(
                        predict(runner, routing_control(encoded, 1330011 + group))[0]
                    )
                    data["signs"].append(2 * base.arrays["targets"] - 1)
                    data["learned"].append(base.arrays["learned"])
                arrays = {key: np.concatenate(values) for key, values in data.items()}
                results[name] = competence(arrays, seed)
                save_arrays(directory / "outputs.npz", **arrays)
                write_json_exclusive(directory / "result.json", results[name])
            print("generic", name, results[name]["passed"], flush=True)
    return results


def cohort(index, runners):
    base, uniforms = input_batch(f"liu-{index:03}")
    encoded, tail = gaussian(base, uniforms)
    # The common reference is owned by the unchanged structural parent.
    from fsrl.experiments.encoding_contribution.protocol import (
        parent_result as structural_result,
    )

    refs = structural_result()["measurement"]
    protocol = load_registered_protocol("liu_v2")
    choice_seed = cohort_specification(index)["evaluation"]["liu"]["choice_seed"]
    rows, arrays = {}, {}
    for seed in SEEDS:
        for donor in STRUCTURES:
            name = cell(seed, "G", donor)
            margins, weights = predict(runners[(seed, donor)], encoded)
            choices, canonical = sample_choices(margins, protocol, choice_seed)
            rows[name] = {
                "common": record(choices, protocol, refs["references"]),
                "legacy": record(
                    choices,
                    protocol,
                    refs["legacy_reference"],
                    legacy_margins=canonical,
                ),
                "internal": internal_record(margins, weights, base, protocol),
                "scalar_closure_max": scalar_identity(margins),
                "tail": tail,
            }
            arrays.update(
                {
                    f"{name}__margins": margins,
                    f"{name}__w": weights,
                    f"{name}__choices": np.packbits(choices, axis=-1),
                }
            )
    return rows, arrays


def liu():
    runtime()
    validate_source()
    for seed in SEEDS:
        for donor in STRUCTURES:
            validate_complete(RUN_ROOT / "generic" / cell(seed, "G", donor))
    runners = {
        (seed, donor): load_runner(parameters(seed, donor))
        for seed in SEEDS
        for donor in STRUCTURES
    }
    for index in range(100):
        directory = RUN_ROOT / "liu" / f"cohort-{index:03}"
        if directory.exists():
            validate_complete(directory)
            continue
        with unit(directory, {"cohort": index}):
            rows, arrays = cohort(index, runners)
            save_arrays(directory / "outputs.npz", **arrays)
            write_json_exclusive(directory / "result.json", rows)
        if index % 10 == 0:
            print("Liu cohort", index, "complete", flush=True)
    return {"cohorts": 100, "new_cells": 6, "training": 0}
