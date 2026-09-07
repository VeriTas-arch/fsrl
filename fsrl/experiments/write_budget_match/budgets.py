"""Support-only L1 measurements and bounded two-gain calibration."""

from typing import Any, cast

import numpy as np
import torch
from torch import nn

from fsrl.experiments.duplicate_observation.inputs import observed
from fsrl.experiments.training_strategy.execution import PROFILE
from fsrl.experiments.write_cost.inputs import load_input
from fsrl.infra.runtime import compile_module

from .protocol import recipe, specification


class BudgetSupport(nn.Module):
    def __init__(self, cell, native):
        super().__init__()
        self.cell = cell
        self.native = native

    def forward(self, inputs, hidden, eligibility, weights, values):
        writes, clips = [], []
        for phase, current in enumerate(inputs.unbind(0)):
            _, _, m, hidden, next_e, native = self.cell(
                current, hidden, eligibility, weights
            )
            proposed = (
                weights
                + (m[:, :, None] if self.native else values[phase]) * eligibility
            )
            updated = native if self.native else proposed.clamp(-50, 50)
            writes.append(((updated - weights) * self.cell.alpha).abs().mean((1, 2)))
            clips.append(((proposed < -50) | (proposed > 50)).float().mean((1, 2)))
            eligibility, weights = next_e, updated
        return weights, torch.stack(writes), torch.stack(clips)


def inputs_for(panel, *, generic_only):
    return {
        arm + "/" + name: observed(load_input(ref), arm, recipe(panel["id"]))
        for arm in ("clean", "noisy")
        for name, ref in sorted(panel["inputs"].items())
        if not generic_only or name.startswith("test-")
    }


def measure(net, query, probe, cpu, values):
    batch = cpu.to(net.w.device)
    n = batch.support_inputs.shape[2]
    h, e, p = (
        net.initial_hidden(n),
        net.initial_eligibility(n),
        net.initial_fast_weights(n),
    )
    blank = p.new_zeros(2, n, net.model_config.input_size)
    _, _, _, _, _, p = query(blank, h, e, p, True)
    per_trial, clipped = [], []
    values = p.new_tensor(values)
    for current in batch.support_inputs.unbind(0):
        p, w, c = probe(current, h, e, p, values)
        per_trial.append(w)
        clipped.append(c)
    w = torch.stack(per_trial)
    assert torch.count_nonzero(w[:, :2]) == 0
    return {
        "writes": w.cpu().numpy(),
        "clipping": torch.stack(clipped).cpu().numpy(),
        "episode_indices": cpu.arrays.get("episode_indices", np.arange(n)),
    }, p


def probes(net):
    return {
        mode: compile_module(BudgetSupport(net, mode == "B"), PROFILE)
        for mode in ("B", "M")
    }


def totals(raw):
    # Each row is one episode, retaining all four phases for zero-write checks.
    return raw["writes"].astype(float).sum(0).T


def means(batches):
    # Equal archives within observation, then distinct observation/phase targets.
    return np.stack(
        [
            np.mean(
                [
                    totals(v).mean(0)[2:]
                    for k, v in batches.items()
                    if k.startswith(arm)
                ],
                axis=0,
            )
            for arm in ("clean/", "noisy/")
        ]
    )


def solve(evaluate, target, config):
    """Fit no behavior: log-ratio residuals of four closed-loop budget moments."""
    from scipy.optimize import least_squares

    history = []
    bound = np.log(config["gain_limit"])

    def residual(x):
        measured = evaluate(np.exp(x))
        ratio = measured / target
        assert np.all(np.isfinite(ratio)) and np.all(ratio > 0)
        history.append({"log_gain": x.tolist(), "ratios": ratio.tolist()})
        if len(history) % 10 == 0:
            print({"budget_calls": len(history), "ratios": ratio.tolist()}, flush=True)
        return np.log(ratio).ravel()

    def jacobian(x):
        columns = []
        for direction in np.eye(2):
            high = np.clip(x + direction * config["difference_step"], -bound, bound)
            low = np.clip(x - direction * config["difference_step"], -bound, bound)
            columns.append((residual(high) - residual(low)) / np.sum(high - low))
        return np.column_stack(columns)

    fitted = least_squares(
        residual,
        np.zeros(2),
        # SciPy accepts callable Jacobians; its installed inferred type is str.
        jac=cast(Any, jacobian),
        bounds=(-bound, bound),
        max_nfev=config["max_nfev"],
        ftol=config["tolerance"],
        xtol=config["tolerance"],
        gtol=config["tolerance"],
    )
    gain = np.exp(fitted.x)
    ratios = evaluate(gain) / target
    return {
        "gain": gain.tolist(),
        "ratios": ratios.tolist(),
        "matched": bool(np.all(np.abs(ratios - 1) <= config["budget_tolerance"])),
        "solver_status": int(fitted.status),
        "solver_message": fitted.message,
        "history": history,
    }


def calibrate_network(net, seqs, panel, kappa):
    inputs = inputs_for(panel, generic_only=True)
    blocks = probes(net)

    def collect(mode, values):
        return {
            key: measure(net, seqs[1], blocks[mode], cpu, values)[0]
            for key, cpu in inputs.items()
        }

    baseline = collect("B", kappa)
    target = means(baseline)

    def evaluate(gain):
        return means(collect("M", [0, 0, *np.multiply(kappa[2:], gain)]))

    result = solve(evaluate, target, specification()["solver"])
    result["values"] = [0.0, 0.0, *np.multiply(kappa[2:], result["gain"]).tolist()]
    final = collect("M", result["values"])
    np.testing.assert_allclose(means(final) / target, result["ratios"], rtol=0, atol=0)
    result["target"] = target.tolist()
    return result, {"B": baseline, "M": final}
