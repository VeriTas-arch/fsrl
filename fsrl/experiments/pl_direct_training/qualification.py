"""Pre-seed optimizer, state, gradient, horizon, and CUDA qualification."""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F

from fsrl.core.factorized_plastic_rnn import FactorizedRecurrentSequence
from fsrl.core.local_trace import (
    ConjunctiveLocalTrace,
    PackedConjunctiveLocalTrace,
)
from fsrl.core.model_config import RetroModelConfig
from fsrl.core.plastic_rnn import RetroModulRNN
from fsrl.core.sequence import RecurrentSequence
from fsrl.infra.provenance import write_json_exclusive
from fsrl.infra.run_manifest import ProspectiveRun
from fsrl.infra.runtime import compile_module
from fsrl.tasks.protocol_catalog import load_registered_protocol

from .batches import (
    DirectTensorBatch,
    expand_legacy_inputs,
    prepare_batch,
    sample_episodes,
)
from .execution import PROFILE, configure_execution
from .locks import QUALIFICATION_ATTEMPT, implementation_sources, qualification_path
from .model import NoTimePlasticRNN, NoTimeRecurrentSequence, map_shadow_model
from .optimization import (
    forward_batch,
    make_optimizer,
    training_step,
)
from .protocol import PROTOCOL_SHA256, REPAIR_SHA256, load_specification
from .task import make_task_generator

QUALIFICATION_SEED = 930001


def _check(error: float, tolerance: float, **values) -> dict:
    return {
        **values,
        "max_abs_error": error,
        "tolerance": tolerance,
        "passed": bool(np.isfinite(error) and error <= tolerance),
    }


def _maximum_error(first, second) -> float:
    return float((first - second).detach().abs().max())


def _sequence(backbone, *, compiled: bool = False):
    module = (
        NoTimeRecurrentSequence(backbone)
        if isinstance(backbone, NoTimePlasticRNN)
        else FactorizedRecurrentSequence(backbone)
    )
    return compile_module(module, PROFILE) if compiled else module


def _fixture_batch(specification: dict, seed: int, batch_size: int = 4):
    episodes = sample_episodes(
        make_task_generator(specification),
        np.random.default_rng(seed),
        batch_size,
        validation=False,
    )
    return prepare_batch(episodes)


def _shadow_forward_batch(
    shadow: RetroModulRNN,
    local: ConjunctiveLocalTrace,
    sequence: RecurrentSequence,
    batch: DirectTensorBatch,
    penalty: float,
):
    batch_size = batch.support_inputs.shape[2]
    fast_weights = shadow.initial_fast_weights(batch_size)
    local_state = local.initial_state(batch_size)
    for trial, (inputs, times) in enumerate(
        zip(batch.support_inputs.unbind(), batch.support_times.unbind(), strict=True)
    ):
        legacy = expand_legacy_inputs(inputs, times, local.cue_size)
        _, _, _, _, _, fast_weights = sequence(
            legacy,
            shadow.initial_hidden(batch_size),
            shadow.initial_eligibility(batch_size),
            fast_weights,
            True,
        )
        local_state = local.write(
            local_state,
            inputs[0, :, : 2 * local.cue_size],
            batch.local_evidence[trial],
        )
    query_size = batch.targets.numel()
    n_queries = query_size // batch_size
    query_weights = (
        fast_weights.unsqueeze(0)
        .expand(n_queries, -1, -1, -1)
        .reshape(
            query_size, shadow.model_config.hidden_size, shadow.model_config.hidden_size
        )
    )
    query_legacy = expand_legacy_inputs(
        batch.query_inputs, batch.query_times, local.cue_size
    )
    logits, _, _, _, _, _ = sequence(
        query_legacy,
        shadow.initial_hidden(query_size),
        shadow.initial_eligibility(query_size),
        query_weights,
        False,
    )
    corrected, _, _, _ = local(
        logits,
        local_state.repeat(n_queries, 1),
        batch.query_inputs[0, :, : 2 * local.cue_size],
    )
    margin = corrected[:, 1:2] - corrected[:, 0:1]
    query_loss = F.cross_entropy(corrected, batch.targets)
    loss = query_loss + penalty * fast_weights.square().mean()
    return loss, query_loss, margin, fast_weights


def _mapped_parameter_error(shadow, clean) -> float:
    errors = [
        _maximum_error(
            clean.input_projection.weight[:, :31], shadow.i2h.weight[:, :31]
        ),
        _maximum_error(clean.input_projection.weight[:, 31], shadow.i2h.weight[:, 34]),
        _maximum_error(
            clean.input_projection.bias,
            shadow.i2h.bias + shadow.i2h.weight[:, 31],
        ),
        _maximum_error(clean.time_weight, shadow.i2h.weight[:, 32]),
        _maximum_error(clean.w, shadow.w),
        _maximum_error(clean.alpha, shadow.alpha),
        _maximum_error(clean.etaet, shadow.etaet),
        _maximum_error(clean.modulation_scale, shadow.DAmult),
        _maximum_error(clean.h2modulation.weight, shadow.h2DA.weight),
        _maximum_error(clean.h2modulation.bias, shadow.h2DA.bias),
        _maximum_error(
            clean.h2margin.weight,
            shadow.h2o.weight[1:2] - shadow.h2o.weight[0:1],
        ),
        _maximum_error(
            clean.h2margin.bias,
            shadow.h2o.bias[1:2] - shadow.h2o.bias[0:1],
        ),
    ]
    return max(errors)


def _state_tensor(optimizer, parameter, name: str) -> torch.Tensor:
    value = optimizer.state[parameter][name]
    if not isinstance(value, torch.Tensor):
        raise TypeError("Adam state is not a tensor")
    return value


def _mapped_moment_error(shadow, clean, shadow_optimizer, clean_optimizer) -> float:
    errors = []
    for moment in ("exp_avg", "exp_avg_sq"):
        direct = _state_tensor(clean_optimizer, clean.input_projection.weight, moment)
        old = _state_tensor(shadow_optimizer, shadow.i2h.weight, moment)
        errors.extend(
            (
                _maximum_error(direct[:, :31], old[:, :31]),
                _maximum_error(direct[:, 31], old[:, 34]),
            )
        )
        direct = _state_tensor(clean_optimizer, clean.input_projection.bias, moment)
        old_bias = _state_tensor(shadow_optimizer, shadow.i2h.bias, moment)
        old_constant = _state_tensor(shadow_optimizer, shadow.i2h.weight, moment)[:, 31]
        errors.extend(
            (_maximum_error(direct, old_bias), _maximum_error(direct, old_constant))
        )
        direct = _state_tensor(clean_optimizer, clean.time_weight, moment)
        errors.append(
            _maximum_error(
                direct,
                _state_tensor(shadow_optimizer, shadow.i2h.weight, moment)[:, 32],
            )
        )
        for direct_parameter, old_parameter in (
            (clean.w, shadow.w),
            (clean.alpha, shadow.alpha),
            (clean.etaet, shadow.etaet),
            (clean.modulation_scale, shadow.DAmult),
            (clean.h2modulation.weight, shadow.h2DA.weight),
            (clean.h2modulation.bias, shadow.h2DA.bias),
        ):
            errors.append(
                _maximum_error(
                    _state_tensor(clean_optimizer, direct_parameter, moment),
                    _state_tensor(shadow_optimizer, old_parameter, moment),
                )
            )
        sign = -1.0 if moment == "exp_avg" else 1.0
        for direct_parameter, old_parameter in (
            (clean.h2margin.weight, shadow.h2o.weight),
            (clean.h2margin.bias, shadow.h2o.bias),
        ):
            direct_state = _state_tensor(clean_optimizer, direct_parameter, moment)
            old_state = _state_tensor(shadow_optimizer, old_parameter, moment)
            errors.extend(
                (
                    _maximum_error(direct_state, old_state[1:2]),
                    _maximum_error(direct_state, sign * old_state[0:1]),
                )
            )
    return max(errors)


def optimizer_trajectory_qualification(specification: dict) -> dict:
    torch.manual_seed(QUALIFICATION_SEED)
    hidden_size = 16
    shadow = RetroModulRNN(RetroModelConfig(37, hidden_size, 2, 4), device="cpu")
    clean = map_shadow_model(shadow, "time_retained_control")
    if isinstance(clean, NoTimePlasticRNN):
        raise TypeError("time control mapped to the no-time class")
    full_local = ConjunctiveLocalTrace(
        15,
        initial_gain=specification["optimization"]["initial_local_gain"],
        device="cpu",
    )
    packed_local = PackedConjunctiveLocalTrace(
        15,
        initial_gain=specification["optimization"]["initial_local_gain"],
        device="cpu",
    )
    shadow_sequence = RecurrentSequence(shadow)
    clean_sequence = FactorizedRecurrentSequence(clean)
    optimization = specification["optimization"]
    shadow_optimizer = torch.optim.Adam(
        [
            {"params": list(shadow.parameters()), "lr": 1e-4},
            {"params": [full_local.raw_gain], "lr": 0.01},
        ],
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        amsgrad=False,
    )
    clean_optimizer = make_optimizer(clean, packed_local, optimization)
    forward_error = 0.0
    update_error = 0.0
    moment_error = 0.0
    rng = np.random.default_rng(QUALIFICATION_SEED + 1)
    for _ in range(8):
        episodes = sample_episodes(
            make_task_generator(specification), rng, 4, validation=False
        )
        batch = prepare_batch(episodes).to("cpu")
        shadow_optimizer.zero_grad(set_to_none=True)
        clean_optimizer.zero_grad(set_to_none=True)
        old = _shadow_forward_batch(
            shadow,
            full_local,
            shadow_sequence,
            batch,
            optimization["fast_weight_penalty"],
        )
        new = forward_batch(
            "time_retained_control",
            clean,
            packed_local,
            clean_sequence,
            batch,
            fast_weight_penalty=optimization["fast_weight_penalty"],
        )
        forward_error = max(
            forward_error,
            _maximum_error(old[0], new.loss),
            _maximum_error(old[1], new.query_loss),
            _maximum_error(old[2], new.margins),
            _maximum_error(old[3], new.fast_weights),
        )
        old[0].backward()
        new.loss.backward()
        torch.nn.utils.clip_grad_norm_(
            shadow.parameters(), optimization["gradient_clip"], error_if_nonfinite=True
        )
        torch.nn.utils.clip_grad_norm_(
            [full_local.raw_gain],
            optimization["gradient_clip"],
            error_if_nonfinite=True,
        )
        from .optimization import clip_effective_backbone_gradients

        clip_effective_backbone_gradients(clean, optimization["gradient_clip"])
        torch.nn.utils.clip_grad_norm_(
            [packed_local.raw_gain],
            optimization["gradient_clip"],
            error_if_nonfinite=True,
        )
        shadow_optimizer.step()
        clean_optimizer.step()
        update_error = max(
            update_error,
            _mapped_parameter_error(shadow, clean),
            _maximum_error(full_local.raw_gain, packed_local.raw_gain),
        )
        moment_error = max(
            moment_error,
            _mapped_moment_error(shadow, clean, shadow_optimizer, clean_optimizer),
        )
    return {
        "optimizer_forward_loss": _check(forward_error, 2e-6, steps=8),
        "optimizer_effective_parameters": _check(update_error, 1e-5, steps=8),
        "optimizer_mapped_moments": _check(moment_error, 1e-5, steps=8),
    }


def packed_local_qualification(specification: dict) -> dict:
    torch.manual_seed(QUALIFICATION_SEED)
    cues = torch.randn(12, 30)
    values = torch.linspace(-1.0, 1.0, 12)
    full = ConjunctiveLocalTrace(
        15,
        initial_gain=specification["optimization"]["initial_local_gain"],
        device="cpu",
    )
    packed = PackedConjunctiveLocalTrace(
        15,
        initial_gain=specification["optimization"]["initial_local_gain"],
        device="cpu",
    )
    full_state = full.write(full.initial_state(12), cues, values)
    packed_state = packed.write(packed.initial_state(12), cues, values)
    query = cues.roll(1, dims=0)
    full_raw, full_correction = full.read(full_state, query)
    packed_raw, packed_correction = packed.read(packed_state, query)
    forward_error = max(
        _maximum_error(full_raw, packed_raw),
        _maximum_error(full_correction, packed_correction),
    )
    full_correction.sum().backward()
    packed_correction.sum().backward()
    gradient_error = _maximum_error(full.raw_gain.grad, packed.raw_gain.grad)
    reversal_error = _maximum_error(
        packed.read(packed_state, torch.cat((query[:, 15:], query[:, :15]), 1))[0],
        -packed_raw,
    )
    return {
        "packed_local": _check(max(forward_error, gradient_error, reversal_error), 1e-6)
    }


def structure_and_time_qualification(specification: dict) -> dict:
    torch.manual_seed(QUALIFICATION_SEED)
    shadow = RetroModulRNN(RetroModelConfig(37, 200, 2, 4), device="cpu")
    control = map_shadow_model(shadow, "time_retained_control")
    no_time = map_shadow_model(shadow, "no_time_candidate")
    shared_error = max(
        _maximum_error(dict(control.named_parameters())[name], value)
        for name, value in no_time.named_parameters()
        if name != "time_weight"
    )
    batch = _fixture_batch(specification, QUALIFICATION_SEED + 2).to("cpu")
    local = PackedConjunctiveLocalTrace(15, device="cpu")
    baseline = forward_batch(
        "no_time_candidate",
        no_time,
        local,
        _sequence(no_time),
        batch,
        fast_weight_penalty=0.0,
    )
    changed = DirectTensorBatch(
        batch.support_inputs,
        torch.flip(batch.support_times, dims=(0,)) + 7.0,
        batch.local_evidence,
        batch.query_inputs,
        batch.query_times + 11.0,
        batch.targets,
    )
    repeated = forward_batch(
        "no_time_candidate",
        no_time,
        local,
        _sequence(no_time),
        changed,
        fast_weight_penalty=0.0,
    )
    isolation_error = max(
        _maximum_error(baseline.margins, repeated.margins),
        _maximum_error(baseline.fast_weights, repeated.fast_weights),
        _maximum_error(baseline.loss, repeated.loss),
    )
    counts = {
        "control_parameters": sum(value.numel() for value in control.parameters()),
        "no_time_parameters": sum(value.numel() for value in no_time.parameters()),
        "control_buffers": len(list(control.named_buffers())),
        "no_time_buffers": len(list(no_time.named_buffers())),
        "no_time_has_time_parameter": "time_weight" in dict(no_time.named_parameters()),
    }
    structure_passed = counts == {
        "control_parameters": 87405,
        "no_time_parameters": 87205,
        "control_buffers": 0,
        "no_time_buffers": 0,
        "no_time_has_time_parameter": False,
    }
    return {
        "initialization_and_fresh_boundary": {
            **counts,
            "shared_max_abs_error": shared_error,
            "passed": structure_passed and shared_error == 0.0,
        },
        "no_time_isolation": _check(isolation_error, 0.0),
    }


def horizon_and_gradient_qualification(specification: dict) -> dict:
    generator = make_task_generator(specification)
    horizon_rows = []
    for edges in range(7, 11):
        episodes = tuple(
            generator.sample(
                np.random.default_rng(QUALIFICATION_SEED + edges + row), n_edges=edges
            )
            for row in range(2)
        )
        batch = prepare_batch(episodes)
        support = 4 * edges
        horizon_rows.append(
            {
                "edges": edges,
                "support_trials": support,
                "active_microsteps": int(
                    batch.arrays["support_inputs"].shape[0] * 4 + 56
                ),
                "effective_P_writes": 2 * support,
            }
        )
    liu = load_registered_protocol("liu_v2")
    liu_literal_microsteps = liu.support_trials * 4 + liu.query_trials * 2
    horizon_passed = (
        [row["active_microsteps"] for row in horizon_rows]
        == [
            168,
            184,
            200,
            216,
        ]
        and [row["effective_P_writes"] for row in horizon_rows]
        == [
            56,
            64,
            72,
            80,
        ]
        and liu_literal_microsteps == 688
    )

    gradients = {}
    for condition in ("time_retained_control", "no_time_candidate"):
        torch.manual_seed(QUALIFICATION_SEED)
        shadow = RetroModulRNN(RetroModelConfig(37, 16, 2, 4), device="cpu")
        backbone = map_shadow_model(shadow, condition)
        local = PackedConjunctiveLocalTrace(15, device="cpu")
        batch = _fixture_batch(specification, QUALIFICATION_SEED + 3).to("cpu")
        result = forward_batch(
            condition,
            backbone,
            local,
            _sequence(backbone),
            batch,
            fast_weight_penalty=0.0,
        )
        result.loss.backward()
        selected = {
            "alpha": backbone.alpha.grad,
            "modulation": backbone.h2modulation.weight.grad,
            "modulation_scale": backbone.modulation_scale.grad,
            "local_gain": local.raw_gain.grad,
        }
        gradients[condition] = {
            name: value is not None
            and bool(torch.isfinite(value).all())
            and float(value.abs().sum()) > 0.0
            for name, value in selected.items()
        }
    return {
        "timestep_identity": {
            "rows": horizon_rows,
            "liu_v2_literal_active_microsteps": liu_literal_microsteps,
            "passed": horizon_passed,
        },
        "gradient_paths": {
            "conditions": gradients,
            "passed": all(all(row.values()) for row in gradients.values()),
        },
    }


def cpu_qualification(specification: dict) -> dict:
    return {
        **structure_and_time_qualification(specification),
        **optimizer_trajectory_qualification(specification),
        **packed_local_qualification(specification),
        **horizon_and_gradient_qualification(specification),
    }


def _parameter_error(first, second) -> float:
    first_values = dict(first.named_parameters())
    second_values = dict(second.named_parameters())
    if set(first_values) != set(second_values):
        raise RuntimeError("parameter inventories differ")
    return max(
        _maximum_error(first_values[name], second_values[name]) for name in first_values
    )


def cuda_qualification(specification: dict) -> tuple[dict, dict]:
    runtime = configure_execution()
    checks = {}
    for condition in ("time_retained_control", "no_time_candidate"):
        torch.manual_seed(QUALIFICATION_SEED)
        shadow = RetroModulRNN(RetroModelConfig(37, 16, 2, 4), device="cuda")
        eager_backbone = map_shadow_model(shadow, condition)
        eager_local = PackedConjunctiveLocalTrace(15, device="cuda")
        compiled_backbone = copy.deepcopy(eager_backbone)
        compiled_local = copy.deepcopy(eager_local)
        batch = _fixture_batch(specification, QUALIFICATION_SEED + 4).to("cuda")
        eager_optimizer = make_optimizer(
            eager_backbone, eager_local, specification["optimization"]
        )
        compiled_optimizer = make_optimizer(
            compiled_backbone, compiled_local, specification["optimization"]
        )
        eager = training_step(
            condition,
            eager_backbone,
            eager_local,
            _sequence(eager_backbone),
            batch,
            eager_optimizer,
            optimization=specification["optimization"],
        )
        compiled = training_step(
            condition,
            compiled_backbone,
            compiled_local,
            _sequence(compiled_backbone, compiled=True),
            batch,
            compiled_optimizer,
            optimization=specification["optimization"],
        )
        torch.cuda.synchronize()
        output_error = max(
            _maximum_error(eager.margins, compiled.margins),
            _maximum_error(eager.fast_weights, compiled.fast_weights),
            _maximum_error(eager.loss, compiled.loss),
        )
        gradient_error = max(
            _maximum_error(first.grad, second.grad)
            for first, second in zip(
                eager_backbone.parameters(), compiled_backbone.parameters(), strict=True
            )
        )
        update_error = max(
            _parameter_error(eager_backbone, compiled_backbone),
            _parameter_error(eager_local, compiled_local),
        )
        checks[f"cuda_{condition}_output"] = _check(output_error, 1e-5)
        checks[f"cuda_{condition}_gradients"] = _check(gradient_error, 1e-5)
        checks[f"cuda_{condition}_update"] = _check(update_error, 1e-5)
    return checks, runtime


def run_qualification() -> dict:
    specification = load_specification()
    directory = qualification_path().parent
    with ProspectiveRun.start(
        directory,
        workflow_id="pl_direct_training_v1",
        execution_id=(
            f"qualification-{QUALIFICATION_SEED}-attempt{QUALIFICATION_ATTEMPT}"
        ),
        producer={
            "module": __name__,
            "protocol_sha256": PROTOCOL_SHA256,
            "repair_sha256": REPAIR_SHA256,
        },
        resolved_config={"seed": QUALIFICATION_SEED, "scientific_seed": False},
    ):
        checks = cpu_qualification(specification)
        cuda_checks, runtime = cuda_qualification(specification)
        checks.update(cuda_checks)
        result = {
            "schema_version": 1,
            "protocol_sha256": PROTOCOL_SHA256,
            "repair_sha256": REPAIR_SHA256,
            "seed": QUALIFICATION_SEED,
            "attempt": QUALIFICATION_ATTEMPT,
            "liu_evaluated": False,
            "runtime": runtime,
            "checks": checks,
            "sources": implementation_sources(),
            "passed": all(row["passed"] for row in checks.values()),
        }
        write_json_exclusive(directory / "qualification.json", result)
    return result
