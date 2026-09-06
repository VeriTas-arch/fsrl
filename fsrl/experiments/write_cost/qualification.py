"""Independent step accounting and prospective numerical parity."""

import copy

import numpy as np
import torch

from fsrl.core.sequence import RecurrentSequence
from fsrl.experiments.evidence_routing.qualification import compare
from fsrl.experiments.memory_structure.inputs import generator, prepare_shared
from fsrl.experiments.memory_structure.model import (
    forward_batch,
    make_model,
    objective,
    optimizer_for,
)
from fsrl.experiments.memory_structure.model import (
    update as old_update,
)
from fsrl.experiments.training_strategy.batches import sample_episodes
from fsrl.infra.provenance import write_json_exclusive

from .execution import configure_execution
from .model import WriteSequence, loss_for, rollout, sequences, update
from .protocol import RECORDS, specification


def independent_trial(cell, inputs, hidden, eligibility, weights):
    deltas = []
    for x in inputs.unbind(0):
        next_h = torch.tanh(
            cell.i2h(x)
            + ((cell.w + cell.alpha * weights) @ hidden[..., None]).squeeze(-1)
        )
        da = torch.tanh(cell.h2DA(next_h))
        m = cell.DAmult * (da[:, 0] - da[:, 1])
        updated = torch.clamp(weights + m[:, None, None] * eligibility, -50.0, 50.0)
        deltas.append(
            ((updated - weights) * cell.alpha).abs().flatten(1).sum(1)
            / weights.shape[1] ** 2
        )
        eligibility = (1 - cell.etaet) * eligibility + cell.etaet * torch.tanh(
            next_h[..., None] * hidden[:, None, :]
        )
        hidden, weights = next_h, updated
    return weights, torch.stack(deltas).sum(0)


def qualify(spec, device="cpu", compiled=False):
    seed = spec["seeds"]["smoke"]
    backbone, local = make_model(spec, seed, "dual", device)
    assert local is not None
    cpu = prepare_shared(
        sample_episodes(generator(spec), np.random.default_rng(seed), 2)
    )
    batch = cpu.to(device)
    support, query = sequences(backbone)
    old = forward_batch(backbone, local, RecurrentSequence(backbone), batch)
    new, cost, _ = rollout(backbone, local, support, query, batch)
    checks = {
        "lambda_zero_logits": compare(old.logits, new.logits),
        "lambda_zero_P": compare(old.weights, new.weights),
    }
    old_loss, _ = objective(old, batch, spec["optimization"]["fast_weight_penalty"])
    old_loss.backward()
    gradients = {
        n: p.grad.clone() for n, p in backbone.named_parameters() if p.grad is not None
    }
    backbone.zero_grad(set_to_none=True)
    local.zero_grad(set_to_none=True)
    loss, _ = loss_for(new, cost, batch, spec, 0.0)
    loss.backward()
    for name, parameter in backbone.named_parameters():
        if name in gradients:
            checks[f"zero_gradient_{name}"] = compare(gradients[name], parameter.grad)
    first, second = copy.deepcopy(backbone), copy.deepcopy(backbone)
    a, b = copy.deepcopy(local), copy.deepcopy(local)
    old_update(
        first, a, RecurrentSequence(first), batch, optimizer_for(first, a, spec), spec
    )
    s, q = sequences(second)
    update(second, b, s, q, batch, optimizer_for(second, b, spec), spec, 0.0)
    for name, value in first.state_dict().items():
        checks[f"zero_update_{name}"] = compare(value, second.state_dict()[name])
    checks["zero_update_local"] = compare(a.raw_gain, b.raw_gain)
    h = backbone.initial_hidden(2)
    e = torch.ones_like(backbone.initial_eligibility(2)) * 0.2
    p = torch.ones_like(backbone.initial_fast_weights(2)) * 49.9
    expected_p, expected_c = independent_trial(
        backbone, batch.support_inputs[0], h, e, p
    )
    actual_p, actual_c = WriteSequence(backbone)(batch.support_inputs[0], h, e, p)
    checks["independent_P"] = compare(expected_p, actual_p)
    checks["independent_cost"] = compare(expected_c, actual_c)
    if compiled:
        checks.update(qualify_compiled(spec, backbone, local, batch))
    return checks


def qualify_compiled(spec, backbone, local, batch):
    other, other_local = copy.deepcopy(backbone), copy.deepcopy(local)
    eager = sequences(backbone)
    compiled = sequences(other, compiled=True)
    checks = {}
    for coefficient in (0.0, 10.0):
        backbone.zero_grad(set_to_none=True)
        other.zero_grad(set_to_none=True)
        local.zero_grad(set_to_none=True)
        other_local.zero_grad(set_to_none=True)
        a, ac, _ = rollout(backbone, local, *eager, batch)
        b, bc, _ = rollout(other, other_local, *compiled, batch)
        al, _ = loss_for(a, ac, batch, spec, coefficient)
        bl, _ = loss_for(b, bc, batch, spec, coefficient)
        al.backward()
        bl.backward()
        checks[f"compiled_cost_{coefficient}"] = compare(ac, bc)
        checks[f"compiled_loss_{coefficient}"] = compare(al, bl)
        for name in ("i2h.weight", "alpha", "h2DA.weight", "h2o.weight"):
            checks[f"compiled_gradient_{coefficient}_{name}"] = compare(
                dict(backbone.named_parameters())[name].grad,
                dict(other.named_parameters())[name].grad,
            )
    update(
        backbone, local, *eager, batch, optimizer_for(backbone, local, spec), spec, 10.0
    )
    update(
        other,
        other_local,
        *compiled,
        batch,
        optimizer_for(other, other_local, spec),
        spec,
        10.0,
    )
    for name, value in backbone.state_dict().items():
        checks[f"compiled_update_{name}"] = compare(value, other.state_dict()[name])
    checks["compiled_update_local"] = compare(local.raw_gain, other_local.raw_gain)
    return checks


def run_qualification():
    from .locks import sources

    runtime = configure_execution()
    result = {
        "passed": True,
        "runtime": runtime,
        "checks": qualify(specification(), "cuda", True),
        "sources": sources(),
        "liu_evaluated": False,
    }
    write_json_exclusive(RECORDS / "benchmarks/qualification.final.json", result)
    return {"passed": True, "checks": len(result["checks"])}
