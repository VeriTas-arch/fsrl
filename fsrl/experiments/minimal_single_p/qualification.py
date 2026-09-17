"""Pre-outcome qualification for every registered ladder equation."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from fsrl.experiments.clean_single_p.batches import prepare_single_p
from fsrl.experiments.pl_direct_training.execution import PROFILE, configure_execution
from fsrl.experiments.pl_direct_training.task import make_task_generator
from fsrl.experiments.training_strategy.batches import sample_episodes
from fsrl.infra.provenance import tensor_hashes, write_json_exclusive
from fsrl.infra.runtime import compile_module

from .decisions import classify_level, successor
from .evaluation import GENERIC_GEOMETRY, _canonical_fields
from .locks import sources
from .model import MinimalSinglePSequence, level_settings, make_model
from .optimization import forward_batch, make_optimizer, margin_loss, training_step
from .protocol import LEVELS, PROTOCOL_SHA256, QUALIFICATION


def _manual_step(model, inputs, hidden, eligibility, weights):
    batch, size = hidden.shape
    drive = model.cue_projection(inputs[:, :31]) + inputs[:, 31:32] * (
        model.evidence_weight
    )
    effective = model.alpha * weights if model.alpha is not None else weights
    next_hidden = torch.tanh(
        drive.view(batch, size, 1)
        + torch.matmul(model.w + effective, hidden.view(batch, size, 1))
    ).view(batch, size)
    margin = F.linear(next_hidden, model.h2margin.weight, model.h2margin.bias)
    modulation = F.linear(
        next_hidden, model.h2modulation.weight, model.h2modulation.bias
    )
    next_weights = torch.clamp(
        weights + modulation.view(batch, 1, 1) * eligibility, -50.0, 50.0
    )
    increment = torch.tanh(
        torch.bmm(next_hidden.view(batch, size, 1), hidden.view(batch, 1, size))
    )
    next_eligibility = (
        increment
        if model.etaet is None
        else (1.0 - model.etaet) * eligibility + model.etaet * increment
    )
    return margin, modulation, next_hidden, next_eligibility, next_weights


def _manual_sequence(model, inputs, hidden, eligibility, weights, update):
    margin = modulation = None
    for step_inputs in inputs.unbind(0):
        margin, modulation, hidden, eligibility, proposal = _manual_step(
            model, step_inputs, hidden, eligibility, weights
        )
        if update:
            weights = proposal
    assert margin is not None and modulation is not None
    return margin, modulation, hidden, eligibility, weights


def _manual_forward(model, batch, penalty):
    subjects = batch.support_inputs.shape[2]
    weights = model.initial_fast_weights(subjects)
    for inputs in batch.support_inputs.unbind(0):
        _, _, _, _, weights = _manual_sequence(
            model,
            inputs,
            model.initial_hidden(subjects),
            model.initial_eligibility(subjects),
            weights,
            True,
        )
    queries = batch.targets.numel() // subjects
    query_weights = (
        weights.unsqueeze(0)
        .expand(queries, -1, -1, -1)
        .reshape(batch.targets.numel(), model.model_config.hidden_size, -1)
    )
    margins, _, _, _, _ = _manual_sequence(
        model,
        batch.query_inputs,
        model.initial_hidden(batch.targets.numel()),
        model.initial_eligibility(batch.targets.numel()),
        query_weights,
        False,
    )
    query_loss = margin_loss(margins, batch.targets)
    return margins, weights, query_loss + penalty * weights.square().mean()


def _max_error(first, second) -> float:
    return float(torch.max(torch.abs(first - second)).detach())


def _fixture(seed: int, batch_size: int = 3):
    parent = __import__(
        "fsrl.experiments.clean_single_p.protocol", fromlist=["inherited_recipe"]
    ).inherited_recipe(1)
    task = make_task_generator({"task": parent["task"]})
    episodes = sample_episodes(
        task, np.random.default_rng(seed), batch_size, validation=False
    )
    return prepare_single_p(episodes, "clean", observation_seed=seed)


def _structure_checks() -> dict:
    models = {level: make_model(level, 9401) for level in LEVELS}
    counts = {
        level: sum(parameter.numel() for parameter in model.parameters())
        for level, model in models.items()
    }
    expected = {level: level_settings(level)["parameter_count"] for level in LEVELS}
    c0_hashes = tensor_hashes(models["C0"])
    m1_hashes = tensor_hashes(models["M1"])
    common = {
        "cue_projection.weight",
        "cue_projection.bias",
        "evidence_weight",
        "w",
        "h2margin.weight",
        "h2margin.bias",
    }
    shared = {
        level: {
            name: value
            for name, value in tensor_hashes(models[level]).items()
            if name in common
        }
        for level in ("M1", "M2", "M3")
    }
    alpha_presence = {level: models[level].alpha is not None for level in LEVELS}
    eta_presence = {level: models[level].etaet is not None for level in LEVELS}
    zero_modulation = {
        level: bool(
            torch.count_nonzero(models[level].h2modulation.weight) == 0
            and torch.count_nonzero(models[level].h2modulation.bias) == 0
        )
        for level in LEVELS
    }
    source = Path(__file__).with_name("model.py").read_text(encoding="utf-8")
    passed = (
        counts == expected
        and c0_hashes == m1_hashes
        and len({tuple(sorted(value.items())) for value in shared.values()}) == 1
        and alpha_presence == {level: level in {"C0", "M1"} for level in LEVELS}
        and eta_presence == {level: level in {"C0", "M1", "M2"} for level in LEVELS}
        and all(not zero_modulation[level] for level in ("C0", "M1"))
        and all(zero_modulation[level] for level in LEVELS[2:])
        and all(not list(model.named_buffers()) for model in models.values())
        and "LinearModulationRNN" not in source
        and "map_shadow" not in source
    )
    return {
        "counts": counts,
        "expected_counts": expected,
        "C0_M1_bit_identical": c0_hashes == m1_hashes,
        "shared_H200_bit_identical": len(
            {tuple(sorted(value.items())) for value in shared.values()}
        )
        == 1,
        "alpha_presence": alpha_presence,
        "eta_presence": eta_presence,
        "zero_modulation": zero_modulation,
        "no_buffers": all(not list(model.named_buffers()) for model in models.values()),
        "no_shadow_symbol": "LinearModulationRNN" not in source
        and "map_shadow" not in source,
        "passed": passed,
    }


def _equation_checks(cpu) -> dict:
    batch, _ = cpu.to("cpu")
    results = {}
    for level in ("C0", "M1", "M2", "M3"):
        first = make_model(level, 9402, hidden_size=9)
        second = make_model(level, 9402, hidden_size=9)
        penalty = 1e-4 if level == "C0" else 0.0
        first_result = forward_batch(
            first, MinimalSinglePSequence(first), batch, penalty=penalty
        )
        second_margin, second_weights, second_loss = _manual_forward(
            second, batch, penalty
        )
        first_result.loss.backward()
        second_loss.backward()
        gradient_error = max(
            _max_error(left.grad, right.grad)
            for left, right in zip(first.parameters(), second.parameters(), strict=True)
            if left.grad is not None and right.grad is not None
        )
        output_error = max(
            _max_error(first_result.margins, second_margin),
            _max_error(first_result.fast_weights, second_weights),
            _max_error(first_result.loss, second_loss),
        )
        left_optimizer = make_optimizer(first, 1e-4)
        right_optimizer = make_optimizer(second, 1e-4)
        left_optimizer.step()
        right_optimizer.step()
        parameter_error = max(
            _max_error(left, right)
            for left, right in zip(first.parameters(), second.parameters(), strict=True)
        )
        results[level] = {
            "output_max_abs_error": output_error,
            "gradient_max_abs_error": gradient_error,
            "parameter_max_abs_error": parameter_error,
            "passed": max(output_error, gradient_error, parameter_error) <= 1e-6,
        }
    return {"levels": results, "passed": all(row["passed"] for row in results.values())}


def _stream_checks() -> dict:
    first = _fixture(9403)
    second = _fixture(9403)
    changed = _fixture(9404)
    return {
        "deterministic": first.fingerprint() == second.fingerprint(),
        "seed_sensitive": first.fingerprint() != changed.fingerprint(),
        "input_size": first.arrays["support_inputs"].shape[-1],
        "support_steps": first.arrays["support_inputs"].shape[1],
        "query_steps": first.arrays["query_inputs"].shape[0],
        "passed": first.fingerprint() == second.fingerprint()
        and first.fingerprint() != changed.fingerprint()
        and first.arrays["support_inputs"].shape[-1] == 32
        and first.arrays["support_inputs"].shape[1] == 4
        and first.arrays["query_inputs"].shape[0] == 2,
    }


def _decision_checks() -> dict:
    outcomes = {
        "all": classify_level({"a": True, "b": True, "c": True}),
        "mixed_one": classify_level({"a": True, "b": False, "c": False}),
        "mixed_two": classify_level({"a": True, "b": True, "c": False}),
        "none": classify_level({"a": False, "b": False, "c": False}),
    }
    expected = {
        "all": "clear_continue",
        "mixed_one": "mixed_boundary",
        "mixed_two": "mixed_boundary",
        "none": "structural_collapse",
    }
    transitions = {level: successor(level, "clear_continue") for level in LEVELS}
    expected_transitions = {
        level: (LEVELS[index + 1] if index + 1 < len(LEVELS) else None)
        for index, level in enumerate(LEVELS)
    }
    return {
        "outcomes": outcomes,
        "transitions": transitions,
        "passed": outcomes == expected and transitions == expected_transitions,
    }


def _geometry_checks() -> dict:
    rng = np.random.default_rng(9405)
    potentials = rng.normal(size=(3, 8))
    # A direct incidence construction is clearer than relying on the projection
    # pseudoinverse for the actual exact-gradient fixture.
    incidence = np.zeros((28, 8), dtype=np.float64)
    for index, (first, second) in enumerate(GENERIC_GEOMETRY.pairs):
        incidence[index, first] = 1.0
        incidence[index, second] = -1.0
    fields = potentials @ incidence.T
    pairs = np.empty((28, 3, 2), dtype=np.int64)
    margins = np.empty((3, 28), dtype=np.float64)
    for query, pair in enumerate(GENERIC_GEOMETRY.pairs):
        for subject in range(3):
            if (query + subject) % 2:
                pairs[query, subject] = pair[::-1]
                margins[subject, query] = -fields[subject, query]
            else:
                pairs[query, subject] = pair
                margins[subject, query] = fields[subject, query]
    canonical = _canonical_fields(margins, pairs)
    from fsrl.analysis.hodge import gradient_energy_fraction

    coherence = gradient_energy_fraction(canonical, GENERIC_GEOMETRY)
    return {
        "canonical_max_abs_error": float(np.max(np.abs(canonical - fields))),
        "coherence": coherence.tolist(),
        "passed": bool(np.max(np.abs(canonical - fields)) == 0.0)
        and bool(np.allclose(coherence, 1.0, atol=1e-12, rtol=0)),
    }


def _cuda_checks(cpu) -> dict:
    batch, _ = cpu.to("cuda")
    levels = {}
    for level in ("C0", "M2", "M3"):
        torch.compiler.reset()
        eager = make_model(level, 9406, device="cuda", hidden_size=9)
        compiled = copy.deepcopy(eager)
        first, _ = training_step(
            eager,
            MinimalSinglePSequence(eager),
            batch,
            make_optimizer(eager, 1e-4),
            penalty=1e-4 if level == "C0" else 0.0,
        )
        second, _ = training_step(
            compiled,
            compile_module(MinimalSinglePSequence(compiled), PROFILE),
            batch,
            make_optimizer(compiled, 1e-4),
            penalty=1e-4 if level == "C0" else 0.0,
        )
        output_error = max(
            _max_error(first.margins, second.margins),
            _max_error(first.fast_weights, second.fast_weights),
            _max_error(first.loss, second.loss),
        )
        parameter_error = max(
            _max_error(left, right)
            for left, right in zip(
                eager.parameters(), compiled.parameters(), strict=True
            )
        )
        levels[level] = {
            "output_max_abs_error": output_error,
            "parameter_max_abs_error": parameter_error,
            "passed": max(output_error, parameter_error) <= 1e-5,
        }
    return {"levels": levels, "passed": all(row["passed"] for row in levels.values())}


def run_qualification() -> dict:
    runtime = configure_execution()
    cpu = _fixture(9402)
    result = {
        "schema_version": 1,
        "protocol_sha256": PROTOCOL_SHA256,
        "sources": sources(),
        "runtime": runtime,
        "structure": _structure_checks(),
        "equations": _equation_checks(cpu),
        "stream": _stream_checks(),
        "decisions": _decision_checks(),
        "geometry": _geometry_checks(),
        "cuda": _cuda_checks(cpu),
        "human_outcomes_exposed": False,
    }
    result["passed"] = all(
        result[name]["passed"]
        for name in (
            "structure",
            "equations",
            "stream",
            "decisions",
            "geometry",
            "cuda",
        )
    )
    write_json_exclusive(QUALIFICATION, result)
    return result


__all__ = ["run_qualification"]
