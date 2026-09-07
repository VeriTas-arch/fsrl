"""Independent finite-state accounting and bounded forward/backward checks."""

import copy

import numpy as np
import torch

from fsrl.experiments.evidence_routing.qualification import compare
from fsrl.experiments.memory_structure.inputs import generator, prepare_shared
from fsrl.experiments.memory_structure.model import make_model, objective, optimizer_for
from fsrl.experiments.training_strategy.batches import sample_episodes
from fsrl.experiments.write_cost.execution import configure_execution
from fsrl.experiments.write_cost.model import rollout as continuous_rollout
from fsrl.experiments.write_cost.model import sequences as continuous_sequences
from fsrl.experiments.write_cost.model import update as continuous_update
from fsrl.infra.provenance import write_json_exclusive

from .model import FiniteSequence, rollout, sequences, stochastic_store, update
from .protocol import RECORDS, specification


def independent_trial(cell, inputs, hidden, eligibility, weights, uniforms, states):
    """Reconstruct the cell and choose grid endpoints explicitly (no STE)."""
    spacing = 100.0 / (states - 1)
    amounts = []
    for step, current in enumerate(inputs.unbind(0)):
        next_h = torch.tanh(
            cell.i2h(current)
            + ((cell.w + cell.alpha * weights) @ hidden[..., None]).squeeze(-1)
        )
        da = torch.tanh(cell.h2DA(next_h))
        modulation = cell.DAmult * (da[:, 0] - da[:, 1])
        candidate = (weights + modulation[:, None, None] * eligibility).clamp(-50, 50)
        low = torch.floor(candidate / spacing) * spacing
        high = low + spacing
        updated = torch.where(uniforms[step] * spacing < candidate - low, high, low)
        amounts.append(
            ((updated - weights) * cell.alpha).abs().flatten(1).sum(1)
            / weights.shape[1] ** 2
        )
        eligibility = (1 - cell.etaet) * eligibility + cell.etaet * torch.tanh(
            next_h[..., None] * hidden[:, None, :]
        )
        hidden, weights = next_h, updated
    return weights, torch.stack(amounts).sum(0)


def analytic_checks(device):
    u = torch.tensor([0.125, 0.375, 0.625, 0.875], device=device)
    proposal = torch.full_like(u, 0.125, requires_grad=True)
    stored = stochastic_store(proposal, u, 201)
    checks = {
        "rounding_mean": compare(stored.mean(), proposal.mean()),
        "rounding_variance": compare(
            stored.var(unbiased=False), proposal.new_tensor(0.125 * 0.375)
        ),
    }
    stored.sum().backward()
    checks["identity_STE"] = compare(proposal.grad, torch.ones_like(proposal))
    grid = torch.arange(-32, 33, device=device) * (100.0 / 64)
    checks["grid_fixed_points"] = compare(
        stochastic_store(grid, torch.zeros_like(grid), 65), grid
    )
    cost = stored.detach().abs() + (proposal.abs() - proposal.abs().detach())
    proposal.grad = None
    cost.mean().backward()
    checks["expected_cost_gradient"] = compare(
        proposal.grad, torch.full_like(proposal, 0.25)
    )
    checks["realized_cost"] = compare(cost, stored.abs())
    return checks


def qualify(spec, device="cpu", compiled=False):
    seed = spec["seeds"]["smoke"]
    small = copy.deepcopy(spec)
    if not compiled:
        small["architecture"]["hidden_size"] = 16
    backbone, local = make_model(small, seed, "dual", device)
    assert local is not None
    cpu = prepare_shared(
        sample_episodes(generator(spec), np.random.default_rng(seed), 2)
    )
    # Three support trials exercise writeback and trial resets; no behavioral outcome.
    cpu.arrays["support_inputs"] = cpu.arrays["support_inputs"][:3]
    cpu.arrays["local_evidence"] = cpu.arrays["local_evidence"][:3]
    batch = cpu.to(device)
    checks = analytic_checks(device)
    checks.update(continuous_gradients(backbone, local, batch, spec, seed))
    for coefficient in (0.0, spec["cost"]["coefficient"]):
        first, second = copy.deepcopy(backbone), copy.deepcopy(backbone)
        a, b = copy.deepcopy(local), copy.deepcopy(local)
        continuous_update(
            first,
            a,
            *continuous_sequences(first),
            batch,
            optimizer_for(first, a, spec),
            spec,
            coefficient,
        )
        update(
            second,
            b,
            sequences(second, None),
            batch,
            optimizer_for(second, b, spec),
            spec,
            None,
            coefficient,
            seed,
        )
        for name, value in first.state_dict().items():
            checks[f"continuous_update_{coefficient}_{name}"] = compare(
                value, second.state_dict()[name]
            )
        checks[f"continuous_local_{coefficient}"] = compare(a.raw_gain, b.raw_gain)
    h, e = backbone.initial_hidden(2), backbone.initial_eligibility(2) + 0.2
    p = backbone.initial_fast_weights(2) + 50.0
    uniforms = torch.rand((4, *p.shape), device=device)
    for states in spec["storage"]["state_candidates"]:
        actual = FiniteSequence(backbone, states)(
            batch.support_inputs[0], h, e, p, uniforms
        )
        expected = independent_trial(
            backbone, batch.support_inputs[0], h, e, p, uniforms, states
        )
        checks[f"independent_P_{states}"] = compare(actual[0], expected[0])
        checks[f"independent_cost_{states}"] = compare(actual[1], expected[1])
    if compiled:
        checks.update(compiled_checks(backbone, local, batch, spec, seed))
    return checks


def continuous_gradients(backbone, local, batch, spec, seed):
    checks = {}
    old, old_cost, _ = continuous_rollout(
        backbone, local, *continuous_sequences(backbone), batch
    )
    new, new_cost, _ = rollout(
        backbone, local, sequences(backbone, None), batch, None, seed
    )
    checks["continuous_logits"] = compare(old.logits, new.logits)
    checks["continuous_state"] = compare(old.weights, new.weights)
    parameters = list(backbone.named_parameters()) + [
        ("local.raw_gain", local.raw_gain)
    ]
    coefficient = spec["cost"]["coefficient"]
    old_loss = (
        objective(old, batch, spec["optimization"]["fast_weight_penalty"])[0]
        + coefficient * old_cost.mean()
    )
    new_loss = (
        objective(new, batch, spec["optimization"]["fast_weight_penalty"])[0]
        + coefficient * new_cost.mean()
    )
    checks["continuous_loss"] = compare(old_loss, new_loss)
    first = torch.autograd.grad(old_loss, [p for _, p in parameters], allow_unused=True)
    second = torch.autograd.grad(
        new_loss, [p for _, p in parameters], allow_unused=True
    )
    for (name, _), a, b in zip(parameters, first, second, strict=True):
        if a is None or b is None:
            assert a is None and b is None
        else:
            checks[f"continuous_gradient_{name}"] = compare(a, b)
    return checks


def compiled_checks(backbone, local, batch, spec, seed):
    checks = {}
    for states in (None, *spec["storage"]["state_candidates"]):
        first, second = copy.deepcopy(backbone), copy.deepcopy(backbone)
        a, b = copy.deepcopy(local), copy.deepcopy(local)
        eager, compiled = (
            sequences(first, states),
            sequences(second, states, compiled=True),
        )
        x, xc, _ = rollout(first, a, eager, batch, states, seed)
        y, yc, _ = rollout(second, b, compiled, batch, states, seed)
        checks[f"compiled_P_{states}"] = compare(x.weights, y.weights)
        checks[f"compiled_logits_{states}"] = compare(x.logits, y.logits)
        checks[f"compiled_cost_{states}"] = compare(xc, yc)
        coefficient = spec["cost"]["coefficient"]
        update(
            first,
            a,
            eager,
            batch,
            optimizer_for(first, a, spec),
            spec,
            states,
            coefficient,
            seed,
        )
        update(
            second,
            b,
            compiled,
            batch,
            optimizer_for(second, b, spec),
            spec,
            states,
            coefficient,
            seed,
        )
        for name, value in first.state_dict().items():
            checks[f"compiled_update_{states}_{name}"] = compare(
                value, second.state_dict()[name]
            )
        checks[f"compiled_local_{states}"] = compare(a.raw_gain, b.raw_gain)
    return checks


def run_qualification():
    from .locks import sources

    runtime = configure_execution()
    checks = qualify(specification(), "cuda", True)
    result = {
        "passed": True,
        "runtime": runtime,
        "checks": checks,
        "sources": sources(),
        "liu_evaluated": False,
    }
    write_json_exclusive(RECORDS / "benchmarks/qualification.json", result)
    return {"passed": True, "checks": len(checks)}
