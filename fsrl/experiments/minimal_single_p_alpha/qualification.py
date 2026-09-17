"""Pre-outcome qualification for the paired M2-alpha candidate."""

from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
import torch
import torch.nn.functional as F

from fsrl.core.sequence import RecurrentSequence
from fsrl.experiments.clean_single_p.batches import prepare_single_p
from fsrl.experiments.minimal_single_p.model import (
    MinimalSinglePSequence,
)
from fsrl.experiments.minimal_single_p.model import (
    make_model as make_m2,
)
from fsrl.experiments.minimal_single_p.optimization import make_optimizer
from fsrl.experiments.pl_direct_training.execution import PROFILE, configure_execution
from fsrl.experiments.pl_direct_training.task import make_task_generator
from fsrl.experiments.training_strategy.batches import sample_episodes
from fsrl.infra.provenance import load_json, tensor_hashes, write_json_exclusive
from fsrl.infra.runtime import compile_module

from .adapter import evaluation_adapter
from .decisions import outcome
from .locks import PARENT_MODELS, PARENT_SOURCE, sources, verify_reference
from .model import make_model
from .protocol import PROTOCOL_SHA256, QUALIFICATION, specification


def _max_error(left, right) -> float:
    return float(torch.max(torch.abs(left - right)).detach())


def _pairing() -> dict:
    seed = 9901
    alpha = make_model(seed)
    parent = make_m2("M2", seed)
    alpha_hashes = tensor_hashes(alpha)
    parent_hashes = tensor_hashes(parent)
    common = set(parent_hashes)
    if any(alpha_hashes[name] != parent_hashes[name] for name in common):
        raise RuntimeError("shared M2-alpha initialization differs from M2")
    batch = 5
    inputs = torch.randn(4, batch, 32)
    alpha_values = MinimalSinglePSequence(alpha)(
        inputs,
        alpha.initial_hidden(batch),
        alpha.initial_eligibility(batch),
        alpha.initial_fast_weights(batch),
        True,
    )
    parent_values = MinimalSinglePSequence(parent)(
        inputs,
        parent.initial_hidden(batch),
        parent.initial_eligibility(batch),
        parent.initial_fast_weights(batch),
        True,
    )
    errors = [
        _max_error(left, right)
        for left, right in zip(alpha_values, parent_values, strict=True)
    ]
    return {
        "shared_tensor_hashes_equal": True,
        "alpha_all_ones": bool(torch.all(alpha.alpha == 1.0)),
        "initial_computation_max_abs_error": max(errors),
        "parameter_count": sum(value.numel() for value in alpha.parameters()),
        "zero_modulation": bool(
            torch.count_nonzero(alpha.h2modulation.weight) == 0
            and torch.count_nonzero(alpha.h2modulation.bias) == 0
        ),
        "passed": max(errors) == 0.0
        and bool(torch.all(alpha.alpha == 1.0))
        and sum(value.numel() for value in alpha.parameters()) == 87003,
    }


def _adapter_parity(device: str, compiled: bool) -> dict:
    model = make_model(9902, device=device)
    with torch.no_grad():
        model.alpha.add_(
            torch.linspace(-0.2, 0.2, model.alpha.numel(), device=device).view_as(
                model.alpha
            )
        )
    adapter = evaluation_adapter(copy.deepcopy(model))
    native = MinimalSinglePSequence(model)
    legacy = RecurrentSequence(adapter)
    if compiled:
        native = compile_module(native, PROFILE)
        legacy = compile_module(legacy, PROFILE)
    batch = 5
    inputs = torch.randn(4, batch, 32, device=device)
    widened = torch.zeros(4, batch, 38, device=device)
    widened[..., :31] = inputs[..., :31]
    widened[..., 37] = inputs[..., 31]
    left = native(
        inputs,
        model.initial_hidden(batch),
        model.initial_eligibility(batch),
        model.initial_fast_weights(batch),
        True,
    )
    right = legacy(
        widened,
        adapter.initial_hidden(batch),
        adapter.initial_eligibility(batch),
        adapter.initial_fast_weights(batch),
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
    eager = make_model(9903, device="cuda")
    compiled = make_model(9903, device="cuda")
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
        "generic": outcome(39, 0, 0),
        "rescue": outcome(60, 40, 0),
        "inflation": outcome(60, 0, 40),
        "mixed": outcome(60, 1, 0),
        "none": outcome(60, 0, 0),
    }
    expected = {
        "generic": "generic_inadequate",
        "rescue": "robust_constrained_rescue",
        "inflation": "error_inflation",
        "mixed": "partial_or_mixed",
        "none": "no_constrained_rescue",
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
        "adapter_cpu": _adapter_parity("cpu", False),
        "adapter_cuda_eager": _adapter_parity("cuda", False),
        "adapter_cuda_compiled": _adapter_parity("cuda", True),
        "compiled_update": _compiled_update(),
        "decisions": _decisions(),
        "stream_replay": _replay_streams(),
        "scientific_outcomes_exposed": False,
    }
    result["passed"] = (
        all(
            result[name]["passed"]
            for name in (
                "pairing",
                "adapter_cpu",
                "adapter_cuda_eager",
                "adapter_cuda_compiled",
                "compiled_update",
                "decisions",
            )
        )
        and len(result["stream_replay"]) == 20
    )
    write_json_exclusive(QUALIFICATION, result)
    return {
        "passed": result["passed"],
        "paired_streams": len(result["stream_replay"]),
        "source_files": len(result["sources"]),
    }


__all__ = ["qualify"]
