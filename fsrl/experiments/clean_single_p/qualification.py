"""Pre-outcome qualification for the clean single-P computation."""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F

from fsrl.core.model_config import RetroModelConfig
from fsrl.core.sequence import RecurrentSequence
from fsrl.experiments.linear_modulation.model import LinearModulationRNN
from fsrl.experiments.pl_direct_training.execution import PROFILE, configure_execution
from fsrl.experiments.pl_direct_training.task import make_task_generator
from fsrl.experiments.training_strategy.batches import sample_episodes
from fsrl.infra.provenance import write_json_exclusive
from fsrl.infra.runtime import compile_module

from .batches import prepare_single_p
from .locks import sources
from .model import AffineSinglePSequence, expand_shadow_inputs, map_shadow
from .optimization import (
    clip_gradients,
    forward_batch,
    make_optimizer,
    training_step,
)
from .protocol import PROTOCOL_SHA256, QUALIFICATION, inherited_recipe, specification


def _shadow_forward(shadow, batch, times, penalty):
    sequence = RecurrentSequence(shadow)
    count = batch.support_inputs.shape[2]
    weights = shadow.initial_fast_weights(count)
    for inputs, trial_times in zip(
        batch.support_inputs.unbind(0), times.support.unbind(0), strict=True
    ):
        legacy = expand_shadow_inputs(inputs, trial_times, 15)
        _, _, _, _, _, weights = sequence(
            legacy,
            shadow.initial_hidden(count),
            shadow.initial_eligibility(count),
            weights,
            True,
        )
    query_size = batch.targets.numel()
    query_count = query_size // count
    query_weights = weights.repeat(query_count, 1, 1)
    legacy = expand_shadow_inputs(batch.query_inputs, times.query, 15)
    logits, _, _, _, _, _ = sequence(
        legacy,
        shadow.initial_hidden(query_size),
        shadow.initial_eligibility(query_size),
        query_weights,
        False,
    )
    loss = F.cross_entropy(logits, batch.targets) + penalty * weights.square().mean()
    return logits[:, 1:2] - logits[:, 0:1], weights, loss


def _maximum_difference(first, second) -> float:
    return max(
        float(torch.max(torch.abs(first[name] - second[name])).detach())
        for name in first
    )


def _optimizer_geometry(batch, times, spec):
    torch.manual_seed(931001)
    shadow = LinearModulationRNN(RetroModelConfig(38, 9, 2, 3), device="cpu")
    clean = map_shadow(shadow, "time_retained_control")
    clean_sequence = AffineSinglePSequence(clean)
    old_optimizer = torch.optim.Adam(shadow.parameters(), lr=1e-4)
    new_optimizer = make_optimizer(clean, spec)
    forward_error = parameter_error = moment_error = 0.0
    for _ in range(8):
        old_optimizer.zero_grad(set_to_none=True)
        old_margin, old_weights, old_loss = _shadow_forward(shadow, batch, times, 1e-4)
        old_loss.backward()
        torch.nn.utils.clip_grad_norm_(shadow.parameters(), 2.0)

        new_optimizer.zero_grad(set_to_none=True)
        new = forward_batch(
            "time_retained_control",
            clean,
            clean_sequence,
            batch,
            times,
            penalty=1e-4,
        )
        new.loss.backward()
        clip_gradients(clean, 2.0)
        forward_error = max(
            forward_error,
            float(torch.max(torch.abs(old_margin - new.margins)).detach()),
            float(torch.max(torch.abs(old_weights - new.fast_weights)).detach()),
            float(torch.abs(old_loss - new.loss).detach()),
        )
        old_optimizer.step()
        new_optimizer.step()
        mapped = map_shadow(shadow, "time_retained_control")
        parameter_error = max(
            parameter_error,
            _maximum_difference(
                dict(mapped.named_parameters()), dict(clean.named_parameters())
            ),
        )
        moment_pairs = (
            (
                clean.cue_projection.weight,
                shadow.i2h.weight,
                (slice(None), slice(0, 31)),
            ),
            (clean.cue_projection.bias, shadow.i2h.bias),
            (clean.evidence_weight, shadow.i2h.weight, (slice(None), 34)),
            (clean.time_weight, shadow.i2h.weight, (slice(None), 32)),
            (clean.w, shadow.w),
            (clean.alpha, shadow.alpha),
            (clean.etaet, shadow.etaet),
            (clean.h2modulation.weight, shadow.h2DA.weight),
            (clean.h2modulation.bias, shadow.h2DA.bias),
            (clean.h2margin.weight, shadow.h2o.weight, 1),
            (clean.h2margin.bias, shadow.h2o.bias, 1),
        )
        for pair in moment_pairs:
            new_parameter, old_parameter, *index = pair
            assert new_parameter is not None and old_parameter is not None
            for key in ("exp_avg", "exp_avg_sq"):
                new_value = new_optimizer.state[new_parameter][key]
                old_value = old_optimizer.state[old_parameter][key]
                if index:
                    old_value = old_value[index[0]]
                moment_error = max(
                    moment_error,
                    float(torch.max(torch.abs(new_value - old_value))),
                )
    return {
        "forward_max_abs_error": forward_error,
        "parameter_max_abs_error": parameter_error,
        "moment_max_abs_error": moment_error,
        "passed": forward_error <= 2e-6
        and parameter_error <= 1e-5
        and moment_error <= 1e-5,
    }


def _cuda_parity(cpu, times, spec):
    if not torch.cuda.is_available():
        raise RuntimeError("clean single-P qualification requires CUDA")
    torch.manual_seed(931002)
    shadow = LinearModulationRNN(RetroModelConfig(38, 9, 2, 3), device="cuda")
    eager = map_shadow(shadow, "clean_no_time")
    compiled = copy.deepcopy(eager)
    eager_sequence = AffineSinglePSequence(eager)
    compiled_sequence = compile_module(AffineSinglePSequence(compiled), PROFILE)
    batch, metadata = cpu.to("cuda")
    first = training_step(
        "clean_no_time",
        eager,
        eager_sequence,
        batch,
        metadata,
        make_optimizer(eager, spec),
        spec,
    )
    second = training_step(
        "clean_no_time",
        compiled,
        compiled_sequence,
        batch,
        metadata,
        make_optimizer(compiled, spec),
        spec,
    )
    output_error = max(
        float(torch.max(torch.abs(first.margins - second.margins)).detach()),
        float(torch.max(torch.abs(first.fast_weights - second.fast_weights)).detach()),
        float(torch.abs(first.loss - second.loss).detach()),
    )
    parameter_error = _maximum_difference(
        dict(eager.named_parameters()), dict(compiled.named_parameters())
    )
    return {
        "output_max_abs_error": output_error,
        "parameter_max_abs_error": parameter_error,
        "passed": output_error <= 1e-5 and parameter_error <= 1e-5,
    }


def run_qualification() -> dict:
    spec = specification()
    recipe = inherited_recipe()
    task = make_task_generator({**spec, "task": recipe["task"]})
    episodes = sample_episodes(task, np.random.default_rng(931001), 3)
    cpu = prepare_single_p(episodes, "noisy", observation_seed=931001)
    batch, times = cpu.to("cpu")
    geometry = _optimizer_geometry(batch, times, spec)

    torch.manual_seed(931003)
    shadow = LinearModulationRNN(RetroModelConfig(38, 9, 2, 3), device="cpu")
    candidate = map_shadow(shadow, "clean_no_time")
    candidate_result = forward_batch(
        "clean_no_time",
        candidate,
        AffineSinglePSequence(candidate),
        batch,
        times,
        penalty=1e-4,
    )
    candidate_result.loss.backward()
    gradient_checks = {
        name: value.grad is not None
        and bool(torch.isfinite(value.grad).all())
        and bool(torch.count_nonzero(value.grad))
        for name, value in candidate.named_parameters()
        if name
        in {
            "cue_projection.weight",
            "evidence_weight",
            "w",
            "alpha",
            "etaet",
            "h2modulation.weight",
        }
    }
    torch.manual_seed(931004)
    count_shadow = LinearModulationRNN(RetroModelConfig(38, 200, 2, 3), device="cpu")
    count_control = map_shadow(count_shadow, "time_retained_control")
    count_candidate = map_shadow(count_shadow, "clean_no_time")
    counts = {
        "time_control_parameters": sum(
            value.numel() for value in count_control.parameters()
        ),
        "no_time_parameters": sum(
            value.numel() for value in count_candidate.parameters()
        ),
        "task_input_size": batch.support_inputs.shape[-1],
        "support_steps": batch.support_inputs.shape[1],
        "query_steps": batch.query_inputs.shape[0],
        "blank_steps": 0,
    }
    expected_counts = {
        "time_control_parameters": 87203,
        "no_time_parameters": 87003,
        "task_input_size": 32,
        "support_steps": 4,
        "query_steps": 2,
        "blank_steps": 0,
    }
    cuda = _cuda_parity(cpu, times, spec)
    result = {
        "schema_version": 1,
        "protocol_sha256": PROTOCOL_SHA256,
        "sources": sources(),
        "runtime": configure_execution(),
        "optimizer_geometry": geometry,
        "cuda_parity": cuda,
        "counts": counts,
        "expected_counts": expected_counts,
        "gradient_paths": gradient_checks,
        "time_isolation": {
            "candidate_has_time_parameter": "time_weight"
            in dict(candidate.named_parameters()),
            "candidate_has_buffers": bool(list(candidate.named_buffers())),
        },
    }
    result["passed"] = (
        geometry["passed"]
        and cuda["passed"]
        and counts == expected_counts
        and all(gradient_checks.values())
        and not result["time_isolation"]["candidate_has_time_parameter"]
        and not result["time_isolation"]["candidate_has_buffers"]
    )
    write_json_exclusive(QUALIFICATION, result)
    return result


__all__ = ["run_qualification"]
