"""Portable reconstruction from registered observations, codes and final scalars."""

import numpy as np
import torch

from fsrl.experiments.minimal_learner.data import ModelBatch
from fsrl.experiments.training_strategy.evaluation import json_ready
from fsrl.experiments.training_strategy.locks import verify_reference
from fsrl.infra.provenance import load_json

from .analysis import conditional_codes
from .assessment import assess_groups, raw_arrays
from .controls import shuffled_teaching
from .inputs import load_group
from .protocol import make_model, resolved_specification, specification
from .reference import rollout


def reconstruct_codes(batch, uniforms, condition, codebook) -> tuple:
    """Independent scalar branch enumeration and tuple-cue cache (no bit keys)."""
    values = batch.arrays["signed"].copy()
    indices = np.full(values.shape, -1, dtype=np.int8)
    orientation = np.empty(values.shape, dtype=np.int8)
    for subject in range(values.shape[1]):
        cache = {}
        for trial in range(values.shape[0]):
            cue = batch.arrays["support_cues"][trial, subject]
            left, right = tuple(cue[: len(cue) // 2]), tuple(cue[len(cue) // 2 :])
            sign = 1 if left < right else -1
            orientation[trial, subject] = sign
            if condition == "exact":
                continue
            if batch.arrays["retention"][trial, subject] == 0:
                values[trial, subject] = 0
                continue
            key = (left, right) if sign == 1 else (right, left)
            if condition == "persistent" and key in cache:
                code = cache[key]
            else:
                observed = sign * batch.arrays["signed"][trial, subject]
                lower = min(sum(value <= observed for value in codebook) - 1, 2)
                probability = (observed - codebook[lower]) / (
                    codebook[lower + 1] - codebook[lower]
                )
                code = lower + int(uniforms[trial, subject] < probability)
                if condition == "persistent":
                    cache[key] = code
            indices[trial, subject] = code
            values[trial, subject] = sign * codebook[code]
    return values, indices, orientation


def verify_conditional_codes(result: dict, spec: dict, codebook) -> None:
    batch, _ = load_group(result["cohorts"]["liu"]["all"])
    raw = raw_arrays(result["files"]["liu-all"])
    model = make_model(spec)
    model.load_state_dict(
        {
            key: torch.tensor(value, dtype=torch.float32)
            for key, value in result["raw_parameters"].items()
        }
    )
    mixture = conditional_codes(
        batch,
        codebook,
        model,
        spec["evaluation"]["liu"]["temperature"],
        raw["correct_signs"],
    )
    saved = raw_arrays(result["files"]["conditional_codes"])
    for key, value in mixture.items():
        np.testing.assert_allclose(saved[key], value, atol=1e-9, rtol=1e-7)


def verify_fit(result: dict) -> dict:
    spec = resolved_specification()
    candidate = specification()
    errors = []
    for domain, groups in result["cohorts"].items():
        for name, input_record in groups.items():
            batch, auxiliary = load_group(input_record)
            raw = raw_arrays(result["files"][f"{domain}-{name}"])
            values, indices, orientation = reconstruct_codes(
                batch,
                auxiliary["encoding_uniforms"],
                result["condition"],
                candidate["encoding"]["codebook"],
            )
            for key, expected in (
                ("internal_signed", values),
                ("code_indices", indices),
                ("orientation", orientation),
            ):
                np.testing.assert_array_equal(raw[f"encoding__{key}"], expected)
            encoded = ModelBatch({**batch.arrays, "signed": values})
            shuffled = shuffled_teaching(encoded, auxiliary["teaching_route"])
            np.testing.assert_array_equal(
                raw["shuffled_signed"], shuffled.arrays["signed"]
            )
            off = ModelBatch(
                {
                    **encoded.arrays,
                    "retention": np.zeros_like(encoded.arrays["retention"]),
                }
            )
            for control, data in (
                ("intact", encoded),
                ("shuffled", shuffled),
                ("z_off", off),
                ("fixed_parameter", encoded),
            ):
                parameters = result[
                    "fixed_parameters" if control == "fixed_parameter" else "parameters"
                ]
                expected = rollout(
                    data.arrays,
                    eta=parameters["eta"],
                    gain=parameters["gamma_G"],
                    epsilon=spec["model"]["epsilon"],
                )
                for key in ("w", "margins"):
                    actual = raw[f"outputs__{control}__{key}"]
                    np.testing.assert_allclose(
                        actual, expected[key], atol=1e-5, rtol=1e-4
                    )
                    errors.append(float(np.max(np.abs(actual - expected[key]))))
            np.testing.assert_array_equal(
                raw["strict_correct_order"],
                (raw["correct_signs"] * raw["outputs__intact__margins"] > 0).all(
                    axis=1
                ),
            )
            scores = np.einsum(
                "bf,bif->bi", raw["outputs__intact__w"], batch.arrays["codes"]
            )
            np.testing.assert_array_equal(raw["scores"], scores)
            np.testing.assert_array_equal(
                raw["orders"], np.argsort(-scores, axis=1, kind="stable")
            )
    if result["condition"] == "persistent":
        verify_conditional_codes(result, spec, candidate["encoding"]["codebook"])
    reconstructed, behavior = assess_groups(
        result["files"], result["cohorts"], result["seed"], spec
    )
    for key, value in reconstructed.items():
        if json_ready(value) != result[key]:
            raise RuntimeError(
                f"cohort statistic or decision does not reconstruct: {key}"
            )

    if json_ready(behavior) != load_json(verify_reference(result["sampled_behavior"])):
        raise RuntimeError("sampled choices/subject classifications do not reconstruct")
    return {
        "passed": True,
        "max_recurrence_error": max(errors),
        "original_behavior_rows": len(result["behavior"]["flags"]),
    }
