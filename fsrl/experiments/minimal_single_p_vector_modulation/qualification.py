"""Pre-outcome qualification for the vector-modulation candidate."""

from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
import torch
import torch.nn.functional as F

from fsrl.core.sequence import RecurrentSequence
from fsrl.experiments.clean_single_p.batches import prepare_single_p
from fsrl.experiments.minimal_single_p.model import MinimalSinglePSequence
from fsrl.experiments.minimal_single_p.model import make_model as make_m2
from fsrl.experiments.minimal_single_p.optimization import (
    forward_batch,
    make_optimizer,
)
from fsrl.experiments.pl_direct_training.execution import PROFILE, configure_execution
from fsrl.experiments.pl_direct_training.task import make_task_generator
from fsrl.experiments.training_strategy.batches import sample_episodes
from fsrl.infra.provenance import load_json, tensor_hashes, write_json_exclusive
from fsrl.infra.runtime import compile_module

from .adapter import evaluation_adapter
from .decisions import outcome
from .diagnostics import geometry_records, summarize_geometry
from .locks import PARENT_MODELS, PARENT_SOURCE, sources, verify_reference
from .model import make_model
from .protocol import PROTOCOL_SHA256, QUALIFICATION, specification


def _max_error(left, right) -> float:
    return float(torch.max(torch.abs(left - right)).detach())


def _pairing() -> dict:
    errors = []
    shared_equal = True
    for seed in specification()["design"]["network_seeds"]:
        vector = make_model(seed)
        parent = make_m2("M2", seed)
        vector_hashes = tensor_hashes(vector)
        parent_hashes = tensor_hashes(parent)
        for name, value in parent_hashes.items():
            if name.startswith("h2modulation."):
                continue
            shared_equal &= vector_hashes[name] == value

        embedded = copy.deepcopy(vector)
        with torch.no_grad():
            embedded.h2modulation.weight.copy_(
                parent.h2modulation.weight.expand_as(embedded.h2modulation.weight)
            )
            embedded.h2modulation.bias.copy_(
                parent.h2modulation.bias.expand_as(embedded.h2modulation.bias)
            )
        inputs = torch.randn(4, 5, 32)
        vector_values = MinimalSinglePSequence(embedded)(
            inputs,
            embedded.initial_hidden(5),
            embedded.initial_eligibility(5),
            embedded.initial_fast_weights(5),
            True,
        )
        parent_values = MinimalSinglePSequence(parent)(
            inputs,
            parent.initial_hidden(5),
            parent.initial_eligibility(5),
            parent.initial_fast_weights(5),
            True,
        )
        errors.extend(
            _max_error(vector_values[index], parent_values[index])
            for index in (0, 2, 3, 4)
        )
        errors.append(
            _max_error(
                vector_values[1],
                parent_values[1].expand_as(vector_values[1]),
            )
        )
    model = make_model(9901)
    zero = bool(
        torch.count_nonzero(model.h2modulation.weight) == 0
        and torch.count_nonzero(model.h2modulation.bias) == 0
    )
    count = sum(value.numel() for value in model.parameters())
    return {
        "shared_tensor_hashes_equal": shared_equal,
        "scalar_embedding_max_abs_error": max(errors),
        "zero_vector_head": zero,
        "parameter_count": count,
        "passed": shared_equal and max(errors) == 0.0 and zero and count == 87002,
    }


def _manual_step(model, inputs, hidden, eligibility, weights):
    batch, size = hidden.shape
    drive = model.cue_projection(inputs[:, :31]) + inputs[:, 31:32] * (
        model.evidence_weight
    )
    next_hidden = torch.tanh(
        drive.view(batch, size, 1)
        + torch.matmul(model.w + weights, hidden.view(batch, size, 1))
    ).view(batch, size)
    margin = F.linear(next_hidden, model.h2margin.weight, model.h2margin.bias)
    modulation = F.linear(
        next_hidden, model.h2modulation.weight, model.h2modulation.bias
    )
    next_weights = torch.clamp(
        weights + modulation.view(batch, size, 1) * eligibility, -50.0, 50.0
    )
    increment = torch.tanh(
        torch.bmm(next_hidden.view(batch, size, 1), hidden.view(batch, 1, size))
    )
    next_eligibility = (1.0 - model.etaet) * eligibility + model.etaet * increment
    return margin, modulation, next_hidden, next_eligibility, next_weights


def _manual_sequence(model, inputs, hidden, eligibility, weights, update):
    margin = modulation = None
    for current in inputs.unbind(0):
        margin, modulation, hidden, eligibility, proposal = _manual_step(
            model, current, hidden, eligibility, weights
        )
        if update:
            weights = proposal
    assert margin is not None and modulation is not None
    return margin, modulation, hidden, eligibility, weights


def _equation_check() -> dict:
    first = make_model(9902, hidden_size=9)
    second = make_model(9902, hidden_size=9)
    with torch.no_grad():
        first.h2modulation.weight.normal_(0.0, 0.1)
        first.h2modulation.bias.normal_(0.0, 0.1)
        second.load_state_dict(first.state_dict())
    inputs = torch.randn(4, 6, 32)
    targets = torch.randint(0, 2, (6,))
    left = MinimalSinglePSequence(first)(
        inputs,
        first.initial_hidden(6),
        first.initial_eligibility(6),
        first.initial_fast_weights(6),
        True,
    )
    right = _manual_sequence(
        second,
        inputs,
        second.initial_hidden(6),
        second.initial_eligibility(6),
        second.initial_fast_weights(6),
        True,
    )
    left_loss = F.softplus(-(2 * targets - 1) * left[0][:, 0]).mean()
    right_loss = F.softplus(-(2 * targets - 1) * right[0][:, 0]).mean()
    left_loss.backward()
    right_loss.backward()
    output_error = max(
        [_max_error(a, b) for a, b in zip(left, right, strict=True)]
        + [_max_error(left_loss, right_loss)]
    )
    gradient_error = max(
        _max_error(a.grad, b.grad)
        for a, b in zip(first.parameters(), second.parameters(), strict=True)
    )
    left_optimizer = make_optimizer(first, 1e-4)
    right_optimizer = make_optimizer(second, 1e-4)
    left_optimizer.step()
    right_optimizer.step()
    parameter_error = max(
        _max_error(a, b)
        for a, b in zip(first.parameters(), second.parameters(), strict=True)
    )
    return {
        "output_max_abs_error": output_error,
        "gradient_max_abs_error": gradient_error,
        "parameter_max_abs_error": parameter_error,
        "passed": max(output_error, gradient_error, parameter_error) <= 1e-6,
    }


def _gradient_symmetry_break() -> dict:
    source = load_json(PARENT_SOURCE)
    task = make_task_generator({"task": source["task"]})
    episodes = sample_episodes(task, np.random.default_rng(9903), 8, validation=False)
    cpu = prepare_single_p(episodes, "clean", observation_seed=9903)
    batch, _ = cpu.to("cpu")
    model = make_model(9903)
    result = forward_batch(model, MinimalSinglePSequence(model), batch, penalty=0.0)
    result.loss.backward()
    weight = model.h2modulation.weight.grad
    bias = model.h2modulation.bias.grad
    assert weight is not None and bias is not None
    row_norms = torch.linalg.vector_norm(weight, dim=1)
    nonzero = bool(torch.all(row_norms > 0) and torch.all(bias != 0))
    nonidentical = bool(torch.unique(row_norms).numel() > 1)
    finite = bool(torch.isfinite(weight).all() and torch.isfinite(bias).all())
    return {
        "finite": finite,
        "all_rows_nonzero": nonzero,
        "row_gradient_norm_std": float(row_norms.std()),
        "nonidentical_rows": nonidentical,
        "passed": finite and nonzero and nonidentical,
    }


def _adapter_parity(device: str, compiled: bool) -> dict:
    model = make_model(9904, device=device, hidden_size=9)
    with torch.no_grad():
        model.h2modulation.weight.normal_(0.0, 0.1)
        model.h2modulation.bias.normal_(0.0, 0.1)
    adapter = evaluation_adapter(copy.deepcopy(model))
    native = MinimalSinglePSequence(model)
    legacy = RecurrentSequence(adapter)
    if compiled:
        native = compile_module(native, PROFILE)
        legacy = compile_module(legacy, PROFILE)
    inputs = torch.randn(4, 5, 32, device=device)
    widened = torch.zeros(4, 5, 38, device=device)
    widened[..., :31] = inputs[..., :31]
    widened[..., 37] = inputs[..., 31]
    left = native(
        inputs,
        model.initial_hidden(5),
        model.initial_eligibility(5),
        model.initial_fast_weights(5),
        True,
    )
    right = legacy(
        widened,
        adapter.initial_hidden(5),
        adapter.initial_eligibility(5),
        adapter.initial_fast_weights(5),
        True,
    )
    errors = {
        "margin": _max_error(left[0], right[0][:, 1:2] - right[0][:, 0:1]),
        "modulation": _max_error(left[1], right[2]),
        "hidden": _max_error(left[2], right[3]),
        "eligibility": _max_error(left[3], right[4]),
        "P": _max_error(left[4], right[5]),
    }
    tolerance = 1e-5 if device == "cuda" else 1e-6
    return {
        "errors": errors,
        "tolerance": tolerance,
        "passed": max(errors.values()) <= tolerance,
    }


def _compiled_update() -> dict:
    eager = make_model(9905, device="cuda")
    compiled = make_model(9905, device="cuda")
    runner = compile_module(MinimalSinglePSequence(compiled), PROFILE)
    inputs = torch.randn(4, 32, 32, device="cuda")
    targets = torch.randint(0, 2, (32,), device="cuda")

    def step(model, sequence):
        values = sequence(
            inputs,
            model.initial_hidden(32),
            model.initial_eligibility(32),
            model.initial_fast_weights(32),
            True,
        )
        loss = F.softplus(-(2 * targets - 1) * values[0][:, 0]).mean()
        loss.backward()
        optimizer = make_optimizer(model, 1e-4)
        optimizer.step()
        return values, loss

    left, left_loss = step(eager, MinimalSinglePSequence(eager))
    right, right_loss = step(compiled, runner)
    errors = [_max_error(a, b) for a, b in zip(left, right, strict=True)]
    errors.append(_max_error(left_loss, right_loss))
    for (_, a), (_, b) in zip(
        eager.named_parameters(), compiled.named_parameters(), strict=True
    ):
        errors.append(_max_error(a, b))
    return {"max_abs_error": max(errors), "passed": max(errors) <= 1e-5}


def _geometry_checks() -> dict:
    inputs = torch.randn(2, 4, 5, 32)
    scalar = make_model(9906, hidden_size=9)
    with torch.no_grad():
        scalar.h2modulation.bias.fill_(0.2)
    scalar_summary = summarize_geometry(geometry_records(scalar, inputs))
    vector = make_model(9906, hidden_size=9)
    with torch.no_grad():
        vector.h2modulation.bias.copy_(torch.linspace(-0.2, 0.2, 9))
    vector_summary = summarize_geometry(geometry_records(vector, inputs))
    inactive = make_model(9906, hidden_size=9)
    inactive_summary = summarize_geometry(geometry_records(inactive, inputs))
    passed = (
        scalar_summary["R"]["maximum"] is not None
        and scalar_summary["R"]["maximum"] <= 1e-10
        and not scalar_summary["vector_used"]
        and vector_summary["vector_used"]
        and vector_summary["R"]["maximum"] > 1e-10
        and inactive_summary["active_records"] == 0
        and not inactive_summary["vector_used"]
    )
    return {
        "scalar": scalar_summary,
        "heterogeneous": vector_summary,
        "inactive": inactive_summary,
        "passed": passed,
    }


def _replay_streams() -> dict:
    parent_source = load_json(PARENT_SOURCE)
    parent_models = load_json(PARENT_MODELS)
    results = {}
    for seed in specification()["design"]["network_seeds"]:
        rows = [
            json.loads(line)
            for line in verify_reference(
                parent_models["runs"][str(seed)]["files"]["train_log.jsonl"]
            )
            .read_text()
            .splitlines()
        ]
        task = make_task_generator({"task": parent_source["task"]})
        rng = np.random.default_rng(151000 + seed)
        digest = hashlib.sha256()
        for step, row in enumerate(rows):
            episodes = sample_episodes(task, rng, 32, validation=False)
            batch = prepare_single_p(
                episodes, "clean", observation_seed=910000000 + seed * 10000 + step
            )
            fingerprint = batch.fingerprint()
            if fingerprint != row["batch_fingerprint"]:
                raise RuntimeError(f"parent replay differs: {seed}/{step}")
            digest.update(bytes.fromhex(fingerprint))
        if len(rows) != 1500 or digest.hexdigest() != rows[-1]["stream_fingerprint"]:
            raise RuntimeError(f"parent cumulative stream differs: {seed}")
        results[str(seed)] = {
            "steps": len(rows),
            "stream_fingerprint": digest.hexdigest(),
            "all_batch_fingerprints_equal": True,
        }
        print(f"qualified paired stream {seed}", flush=True)
    return results


def _decisions() -> dict:
    observed = {
        "generic": outcome(1, 0, 0, 3),
        "damage": outcome(3, 2, 1, 3),
        "rescue": outcome(3, 2, 0, 3),
        "used": outcome(3, 1, 0, 3),
        "unused": outcome(3, 0, 0, 0),
    }
    expected = {
        "generic": "generic_inadequate",
        "damage": "mixed_or_damaging",
        "rescue": "candidate_rescue",
        "used": "used_without_rescue",
        "unused": "vector_not_used",
    }
    return {"observed": observed, "passed": observed == expected}


def qualify() -> dict:
    runtime = configure_execution()
    result = {
        "schema_version": 1,
        "protocol_sha256": PROTOCOL_SHA256,
        "sources": sources(),
        "runtime": runtime,
        "pairing": _pairing(),
        "equation": _equation_check(),
        "gradient_symmetry_break": _gradient_symmetry_break(),
        "geometry": _geometry_checks(),
        "adapter_cpu": _adapter_parity("cpu", False),
        "adapter_cuda_eager": _adapter_parity("cuda", False),
        "adapter_cuda_compiled": _adapter_parity("cuda", True),
        "compiled_update": _compiled_update(),
        "decisions": _decisions(),
        "stream_replay": _replay_streams(),
        "scientific_outcomes_exposed": False,
    }
    checks = (
        "pairing",
        "equation",
        "gradient_symmetry_break",
        "geometry",
        "adapter_cpu",
        "adapter_cuda_eager",
        "adapter_cuda_compiled",
        "compiled_update",
        "decisions",
    )
    result["passed"] = (
        all(result[name]["passed"] for name in checks)
        and len(result["stream_replay"]) == 3
    )
    write_json_exclusive(QUALIFICATION, result)
    return {
        "passed": result["passed"],
        "paired_streams": len(result["stream_replay"]),
        "source_files": len(result["sources"]),
    }


__all__ = ["qualify"]
