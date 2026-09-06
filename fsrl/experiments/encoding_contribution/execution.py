"""Qualified fixed-parameter replay, with inherited observations and no training."""

import numpy as np

from fsrl.experiments.minimal_learner.data import ModelBatch, generic_batch
from fsrl.experiments.minimal_learner.locks import validate_complete
from fsrl.experiments.minimal_learner.protocol import task_generator
from fsrl.experiments.minimal_learner.training import runtime
from fsrl.experiments.structural_identification.evaluation import (
    load_runner,
    mean_interval,
    predict,
    routing_control,
)
from fsrl.experiments.structural_identification.inputs import save_arrays
from fsrl.experiments.structural_identification.model import encode, reference
from fsrl.experiments.structural_identification.observation import (
    record,
    sample_choices,
)
from fsrl.experiments.structural_identification.protocol import cohort_specification
from fsrl.experiments.training_strategy.batches import sample_episodes
from fsrl.infra.provenance import load_json, write_json_exclusive
from fsrl.infra.run_manifest import ProspectiveRun
from fsrl.tasks.protocol_catalog import load_registered_protocol

from .moments import conditional_moments, enumerate_moments
from .protocol import (
    RUN_ROOT,
    SEEDS,
    STRUCTURES,
    cell,
    input_batch,
    parameters,
    parent_arrays,
    parent_result,
    specification,
    validate_source,
)


def unit(directory, kind, settings):
    return ProspectiveRun.start(
        directory,
        workflow_id="encoding_contribution_v1",
        execution_id=f"{kind}-{directory.name}",
        producer={"module": __name__},
        resolved_config=settings,
    )


def qualify():
    snapshot = runtime()
    rng = np.random.default_rng(8430001)
    base = generic_batch(sample_episodes(task_generator(), rng, 2))
    uniforms = rng.random(base.arrays["signed"].shape)
    errors = {}
    for seed in SEEDS:
        for donor in STRUCTURES:
            p = parameters(seed, donor)
            runner = load_runner(p)
            for structure in STRUCTURES:
                encoded = encode(base, structure, uniforms)
                expected, _ = reference(encoded, p["eta0"], p["gamma_G"], "decay")
                output, _ = predict(runner, encoded)
                np.testing.assert_allclose(output, expected, atol=1e-5, rtol=1e-4)
                errors[cell(seed, structure, donor)] = float(
                    np.max(abs(output - expected))
                )
            fixture = ModelBatch(
                {
                    key: value[:1] if key == "query_cues" else value[:4, :1]
                    for key, value in base.arrays.items()
                    if key in ("query_cues", "support_cues", "signed", "retention")
                }
            )
            mean, covariance, _ = conditional_moments(fixture, p["eta0"], p["gamma_G"])
            expected_mean, expected_cov = enumerate_moments(
                fixture, p["eta0"], p["gamma_G"]
            )
            np.testing.assert_allclose(mean, expected_mean, atol=1e-12, rtol=1e-10)
            np.testing.assert_allclose(covariance, expected_cov, atol=1e-12, rtol=1e-10)
    result = {
        "passed": True,
        "CUDA_max_errors": errors,
        "moments_exact_enumeration": True,
        "runtime": snapshot,
    }
    write_json_exclusive(RUN_ROOT / "qualification.json", result)
    return result


def generic():
    runtime()
    validate_source()
    results = {}
    for seed in SEEDS:
        for structure in STRUCTURES:
            donor = "M11" if structure == "M10" else "M10"
            name = cell(seed, structure, donor)
            directory = RUN_ROOT / "generic" / name
            if directory.exists():
                validate_complete(directory)
                results[name] = load_json(directory / "result.json")
                continue
            runner = load_runner(parameters(seed, donor))
            with unit(directory, "generic", {"cell": name}):
                data = {key: [] for key in ("margins", "shuffled", "signs", "learned")}
                for group in range(8):
                    base, uniforms = input_batch(f"generic-{group}")
                    encoded = encode(base, structure, uniforms)
                    data["margins"].append(predict(runner, encoded)[0])
                    data["shuffled"].append(
                        predict(runner, routing_control(encoded, 1330011 + group))[0]
                    )
                    data["signs"].append(2 * base.arrays["targets"] - 1)
                    data["learned"].append(base.arrays["learned"])
                arrays = {key: np.concatenate(value) for key, value in data.items()}
                result = competence(arrays, seed)
                save_arrays(directory / "outputs.npz", **arrays)
                write_json_exclusive(directory / "result.json", result)
                results[name] = result
            print("generic", name, result["passed"], flush=True)
    return results


def competence(arrays, seed):
    correct = arrays["margins"] * arrays["signs"] > 0
    result = {
        name: mean_interval(
            (correct * mask).sum(axis=1) / mask.sum(axis=1), 8400000 + seed
        )
        for name, mask in (
            ("learned", arrays["learned"]),
            ("nonlearned", ~arrays["learned"]),
        )
    }
    result["binding"] = mean_interval(
        correct.mean(axis=1) - (arrays["shuffled"] * arrays["signs"] > 0).mean(axis=1),
        8400000 + seed,
    )
    passed = all(result[key]["lower"] > 0.5 for key in ("learned", "nonlearned"))
    return {**result, "passed": passed and result["binding"]["lower"] > 0}


def query_truth(base, protocol):
    pairs = base.arrays["query_pairs"]
    positions = np.argsort(protocol.true_order_high_to_low)
    signs = np.where(positions[pairs[..., 0]] < positions[pairs[..., 1]], 1, -1)
    learned = {tuple(sorted(pair)) for pair in protocol.learned_pairs}
    mask = np.asarray(
        [[tuple(sorted(pair)) not in learned for pair in row] for row in pairs]
    )
    return signs, mask


def internal_record(margins, weights, base, protocol):
    signs, mask = query_truth(base, protocol)
    accuracy = ((margins * signs > 0) * mask).sum(axis=1) / mask.sum(axis=1)
    scores = np.einsum("bki,bi->bk", base.arrays["codes"], weights)
    order = np.argsort(-scores, axis=1, kind="stable")
    return {
        "nonlearned_sign_accuracy": float(accuracy.mean()),
        "latent_correct_order": float(
            np.mean(
                np.all(order == np.asarray(protocol.true_order_high_to_low), axis=1)
            )
        ),
    }


def cohort_cells(index, runners):
    protocol = load_registered_protocol("liu_v2")
    human = parent_result()["measurement"]
    base, uniforms = input_batch(f"liu-{index:03}")
    previous = parent_arrays(f"liu/cohort-{index:03}/outputs.npz")
    choice_seed = cohort_specification(index)["evaluation"]["liu"]["choice_seed"]
    rows, arrays = {}, {}
    encoded = {st: encode(base, st, uniforms) for st in STRUCTURES}
    for seed in SEEDS:
        for donor in STRUCTURES:
            exact_weights = None
            for structure in STRUCTURES:
                name = cell(seed, structure, donor)
                diagonal = structure == donor
                prefix = f"{seed}-{structure}-decay__"
                if diagonal:
                    margins, weights = (
                        previous[prefix + "margins"],
                        previous[prefix + "w"],
                    )
                else:
                    margins, weights = predict(
                        runners[(seed, donor)], encoded[structure]
                    )
                choices, canonical = sample_choices(margins, protocol, choice_seed)
                if diagonal:
                    np.testing.assert_array_equal(
                        np.packbits(choices, axis=-1), previous[prefix + "choices"]
                    )
                else:
                    arrays.update(
                        {
                            f"{name}__margins": margins,
                            f"{name}__w": weights,
                            f"{name}__choices": np.packbits(choices, axis=-1),
                        }
                    )
                rows[name] = {
                    "common": record(choices, protocol, human["references"]),
                    "legacy": record(
                        choices,
                        protocol,
                        human["legacy_reference"],
                        legacy_margins=canonical,
                    ),
                    "internal": internal_record(margins, weights, base, protocol),
                }
                if structure == "M10":
                    exact_weights = weights
            p = parameters(seed, donor)
            mean, covariance, qvar = conditional_moments(base, p["eta0"], p["gamma_G"])
            assert exact_weights is not None
            np.testing.assert_allclose(exact_weights, mean, atol=1e-5, rtol=1e-4)
            moment_name = f"{seed}-theta_{donor}"
            arrays.update(
                {
                    f"{moment_name}__mean": mean,
                    f"{moment_name}__query_variance": qvar,
                    f"{moment_name}__covariance_trace": np.trace(
                        covariance, axis1=1, axis2=2
                    ),
                }
            )
            rows[cell(seed, "M11", donor)]["moments"] = {
                "mean_parity_max": float(np.max(abs(exact_weights - mean))),
                "nonlearned_margin_variance": float(
                    qvar[query_truth(base, protocol)[1]].mean()
                ),
                "covariance_trace_mean": float(
                    np.trace(covariance, axis1=1, axis2=2).mean()
                ),
            }
    return rows, arrays


def cross():
    runtime()
    validate_source()
    runners = {
        (seed, donor): load_runner(parameters(seed, donor))
        for seed in SEEDS
        for donor in STRUCTURES
    }
    for index in range(100):
        directory = RUN_ROOT / "cross" / f"cohort-{index:03}"
        if directory.exists():
            validate_complete(directory)
            continue
        with unit(
            directory, "cross", {"cohort": index, "design": specification()["cross"]}
        ):
            rows, arrays = cohort_cells(index, runners)
            save_arrays(directory / "outputs.npz", **arrays)
            write_json_exclusive(directory / "result.json", rows)
        if index % 10 == 0:
            print("cross cohort", index, "complete", flush=True)
    return {"cohorts": 100, "cells": 12, "new_training": 0}
