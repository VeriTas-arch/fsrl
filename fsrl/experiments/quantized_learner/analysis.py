"""Fixed query observables, teaching controls and conditional code mixtures."""

import numpy as np
import torch
from scipy.special import expit

from fsrl.experiments.minimal_learner.data import ModelBatch
from fsrl.experiments.training_strategy.estimands import query_endpoints
from fsrl.infra.provenance import tensor_hashes

from .controls import shuffled_teaching
from .encoding import canonical_addresses, encode_batch
from .reference import enumerate_persistent, rollout


def query_groups(batch) -> dict:
    support, _ = canonical_addresses(batch.arrays["support_cues"])
    query, _ = canonical_addresses(batch.arrays["query_cues"])
    same = query[:, None] == support.T[..., None]
    learned = same.any(axis=1)
    retained = (same & (batch.arrays["retention"].T[..., None] == 1)).any(axis=1)
    return {
        "overall": np.ones_like(learned),
        "learned": learned,
        "nonlearned": ~learned,
        "retained": retained,
        "omitted": learned & ~retained,
    }


def query_signs(batch, protocol=None) -> np.ndarray:
    if protocol is None:
        return 2 * batch.arrays["targets"] - 1
    ranks = np.argsort(protocol.true_order_high_to_low)
    pairs = batch.arrays["query_pairs"]
    return np.sign(ranks[pairs[..., 1]] - ranks[pairs[..., 0]])


def readout(model, runner, batch) -> dict:
    before = tensor_hashes(model)
    with torch.no_grad():
        outputs = runner(*batch.tensors(str(next(model.parameters()).device)))
    if tensor_hashes(model) != before:
        raise RuntimeError("evaluation changed slow parameters")
    result = {
        "margins": outputs[0].cpu().numpy().astype(np.float64),
        "w": outputs[3].cpu().numpy().astype(np.float64),
    }
    expected = rollout(
        batch.arrays,
        eta=model.eta.item(),
        gain=model.global_gain.item(),
        epsilon=model.epsilon,
    )
    for key, value in result.items():
        np.testing.assert_allclose(value, expected[key], atol=1e-5, rtol=1e-4)
    return result


def analyze_batch(
    model,
    runner,
    fixed_model,
    fixed_runner,
    batch,
    auxiliary,
    condition,
    codebook,
    temperature,
    protocol=None,
) -> dict:
    encoded, witness = encode_batch(
        batch, condition, auxiliary["encoding_uniforms"], codebook
    )
    shuffled = shuffled_teaching(encoded, auxiliary["teaching_route"])
    off = ModelBatch(
        {**encoded.arrays, "retention": np.zeros_like(encoded.arrays["retention"])}
    )
    outputs = {
        name: readout(model, runner, data)
        for name, data in (("intact", encoded), ("shuffled", shuffled), ("z_off", off))
    }
    outputs["fixed_parameter"] = readout(fixed_model, fixed_runner, encoded)
    np.testing.assert_array_equal(outputs["z_off"]["w"], 0)
    np.testing.assert_array_equal(outputs["z_off"]["margins"], 0)
    queries = encoded.arrays["query_cues"][:, ::-1].copy()
    reversed_queries = ModelBatch({**encoded.arrays, "query_cues": queries})
    check = readout(model, runner, reversed_queries)
    np.testing.assert_array_equal(check["w"], outputs["intact"]["w"])
    np.testing.assert_allclose(
        check["margins"], outputs["intact"]["margins"][:, ::-1], atol=1e-5, rtol=1e-4
    )
    signs = query_signs(batch, protocol)
    groups = query_groups(batch)
    endpoints = {
        name: query_endpoints(
            row["margins"][..., None], signs[..., None], groups, temperature=temperature
        )
        for name, row in outputs.items()
    }
    scores = np.einsum("bf,bif->bi", outputs["intact"]["w"], batch.arrays["codes"])
    return {
        "outputs": outputs,
        "endpoints": endpoints,
        "groups": groups,
        "encoding": witness,
        "shuffled_signed": shuffled.arrays["signed"],
        "correct_signs": signs,
        "scores": scores,
        "orders": np.argsort(-scores, axis=1, kind="stable"),
        "score_ties": (np.diff(np.sort(scores, axis=1), axis=1) == 0).any(axis=1),
        "strict_correct_order": (signs * outputs["intact"]["margins"] > 0).all(axis=1),
        "correct_signed_margins": signs * outputs["intact"]["margins"],
        "pair_correct_probability": expit(
            signs * outputs["intact"]["margins"] / temperature
        ),
    }


def conditional_codes(batch, codebook, model, temperature, signs) -> dict:
    subjects, queries = signs.shape
    weights = np.zeros((subjects, 128))
    margins = np.zeros((subjects, 128, queries))
    counts = np.zeros(subjects, dtype=np.int64)
    covariance = np.zeros((subjects, queries, queries))
    expected_probability = np.zeros_like(signs, dtype=np.float64)
    correct_order_probability = np.zeros(subjects)
    errors = []
    for subject in range(subjects):
        mixture = enumerate_persistent(
            batch.arrays,
            subject,
            codebook,
            eta=model.eta.item(),
            gain=model.global_gain.item(),
            epsilon=model.epsilon,
        )
        count = len(mixture["weights"])
        if count > 128:
            raise RuntimeError("Liu code mixture exceeds the registered support bound")
        np.testing.assert_allclose(
            mixture["mean"], mixture["exact_margin"], atol=1e-9, rtol=1e-7
        )
        np.testing.assert_allclose(
            mixture["covariance"], mixture["analytic_covariance"], atol=1e-9, rtol=1e-7
        )
        errors.append(
            [
                np.max(np.abs(mixture["mean"] - mixture["exact_margin"])),
                np.max(np.abs(mixture["covariance"] - mixture["analytic_covariance"])),
            ]
        )
        weights[subject, :count] = mixture["weights"]
        margins[subject, :count] = mixture["margins"]
        counts[subject] = count
        covariance[subject] = mixture["covariance"]
        correct = mixture["margins"] * signs[subject]
        expected_probability[subject] = mixture["weights"] @ expit(
            correct / temperature
        )
        correct_order_probability[subject] = mixture["weights"] @ (correct > 0).all(
            axis=1
        )
    return {
        "weights": weights,
        "margins": margins,
        "component_counts": counts,
        "covariance": covariance,
        "identity_errors": np.asarray(errors),
        "expected_correct_probability": expected_probability,
        "strict_correct_order_probability": correct_order_probability,
    }
