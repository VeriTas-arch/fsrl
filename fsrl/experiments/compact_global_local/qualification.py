"""Pre-seed algebraic, gradient, and CUDA-compile qualification."""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F

from fsrl.core.local_trace import ConjunctiveLocalTrace
from fsrl.experiments.training_strategy.execution import PROFILE, configure_execution
from fsrl.infra.provenance import write_json_exclusive
from fsrl.infra.run_manifest import ProspectiveRun
from fsrl.infra.runtime import compile_module

from .batches import prepare_batch, sample_episodes
from .locks import RUN_ROOT, implementation_sources
from .model import (
    CompactModelConfig,
    CompactPlasticRNN,
    CompactRecurrentSequence,
    PackedLocalTrace,
)
from .optimization import (
    forward_batch,
    make_optimizer,
    margin_loss,
    training_step,
)
from .protocol import PROTOCOL_SHA256, REPAIR_SHA256, load_specification
from .task import make_task_generator


def _check(error: float, tolerance: float, **values) -> dict:
    return {
        **values,
        "max_abs_error": error,
        "tolerance": tolerance,
        "passed": bool(np.isfinite(error) and error <= tolerance),
    }


def _parameter_error(first, second) -> float:
    errors = []
    for (first_name, first_value), (second_name, second_value) in zip(
        first.named_parameters(), second.named_parameters(), strict=True
    ):
        if first_name != second_name:
            raise RuntimeError("parameter order differs")
        errors.append(float((first_value - second_value).detach().abs().max()))
    return max(errors, default=0.0)


def cpu_qualification() -> dict:
    torch.manual_seed(920001)
    cues = torch.sign(torch.randn(12, 10))
    values = torch.linspace(-1.0, 1.0, 12)
    full = ConjunctiveLocalTrace(5, device="cpu")
    packed = PackedLocalTrace(5, device="cpu")
    full_state = full.write(full.initial_state(12), cues, values)
    packed_state = packed.write(packed.initial_state(12), cues, values)
    full_raw, _ = full.read(full_state, cues.roll(1, dims=0))
    packed_raw = packed.read(packed_state, cues.roll(1, dims=0))
    packed_error = float((full_raw - packed_raw).abs().max())

    margins = torch.linspace(-3.0, 3.0, 11)[:, None]
    targets = torch.arange(11) % 2
    symmetric = torch.cat((-0.5 * margins, 0.5 * margins), dim=1)
    margin_error = float(
        (margin_loss(margins, targets) - F.cross_entropy(symmetric, targets)).abs()
    )

    config = CompactModelConfig(cue_size=3, hidden_size=8)
    backbone = CompactPlasticRNN(config, device="cpu")
    sequence = CompactRecurrentSequence(backbone)
    inputs = torch.zeros(3, 2, config.input_size)
    inputs[0, :, :6] = torch.tensor([[1.0, -1.0, 1.0, -1.0, 1.0, -1.0]] * 2)
    inputs[1, :, config.response_index] = 1.0
    zero = backbone.initial_fast_weights(2)
    writes = []
    for steps in (1, 2, 3):
        writes.append(
            sequence(
                inputs[:steps],
                backbone.initial_hidden(2),
                backbone.initial_eligibility(2),
                zero,
                True,
            )[-1]
        )
    write_error = max(
        float(writes[0].detach().abs().max()),
        float(writes[1].detach().abs().max()),
    )
    write_norm = float(writes[2].detach().abs().sum())
    query_initial = torch.randn_like(zero)
    query_returned = sequence(
        inputs[:2],
        backbone.initial_hidden(2),
        backbone.initial_eligibility(2),
        query_initial,
        False,
    )[-1]
    query_error = float((query_returned - query_initial).abs().max())
    return {
        "packed_local_equivalence": _check(packed_error, 1e-6),
        "margin_loss_equivalence": _check(margin_error, 1e-7),
        "write_timing": {
            "zero_through_two_steps": write_error,
            "three_step_l1": write_norm,
            "passed": write_error == 0.0 and write_norm > 0.0,
        },
        "query_write_discard": _check(query_error, 0.0),
    }


def _fixture(specification: dict, *, device: str, hidden_size: int = 16):
    task = make_task_generator(specification)
    episodes = sample_episodes(
        task, np.random.default_rng(1020001), 4, validation=False
    )
    batch = prepare_batch(episodes).to(device)
    config = CompactModelConfig(
        cue_size=specification["task"]["cue_size"], hidden_size=hidden_size
    )
    backbone = CompactPlasticRNN(config, device=device)
    local = PackedLocalTrace(config.cue_size, device=device)
    return backbone, local, batch


def gradient_qualification(specification: dict) -> dict:
    torch.manual_seed(920001)
    backbone, local, batch = _fixture(specification, device="cpu")
    sequence = CompactRecurrentSequence(backbone)
    result = forward_batch(
        backbone,
        local,
        sequence,
        batch,
        local_active=False,
        fast_weight_penalty=0.0,
    )
    result.loss.backward()
    selected = {
        "alpha": backbone.alpha.grad,
        "h2modulation.weight": backbone.h2modulation.weight.grad,
        "modulation_scale": backbone.modulation_scale.grad,
    }
    finite_nonzero = {
        name: gradient is not None
        and bool(torch.isfinite(gradient).all())
        and float(gradient.abs().sum()) > 0.0
        for name, gradient in selected.items()
    }

    optimization = specification["optimization"]
    local_optimizer = make_optimizer(backbone, local, optimization)
    before = {
        name: value.detach().clone() for name, value in backbone.state_dict().items()
    }
    training_step(
        backbone,
        local,
        sequence,
        batch,
        local_optimizer,
        phase="local",
        optimization=optimization,
    )
    freeze_error = max(
        float((value - before[name]).abs().max())
        for name, value in backbone.state_dict().items()
    )
    return {
        "gradient_path": {
            "parameters": finite_nonzero,
            "passed": all(finite_nonzero.values()),
        },
        "backbone_freeze": _check(freeze_error, 0.0),
    }


def cuda_qualification(specification: dict) -> tuple[dict, dict]:
    runtime = configure_execution()
    torch.manual_seed(920001)
    eager_backbone, eager_local, batch = _fixture(specification, device="cuda")
    compiled_backbone = copy.deepcopy(eager_backbone)
    compiled_local = copy.deepcopy(eager_local)
    eager_sequence = CompactRecurrentSequence(eager_backbone)
    compiled_sequence = compile_module(
        CompactRecurrentSequence(compiled_backbone), PROFILE
    )
    optimization = specification["optimization"]
    eager_optimizer = make_optimizer(eager_backbone, eager_local, optimization)
    compiled_optimizer = make_optimizer(compiled_backbone, compiled_local, optimization)
    eager = training_step(
        eager_backbone,
        eager_local,
        eager_sequence,
        batch,
        eager_optimizer,
        phase="global",
        optimization=optimization,
    )
    compiled = training_step(
        compiled_backbone,
        compiled_local,
        compiled_sequence,
        batch,
        compiled_optimizer,
        phase="global",
        optimization=optimization,
    )
    torch.cuda.synchronize()
    output_error = max(
        float((eager.margins - compiled.margins).detach().abs().max()),
        float((eager.loss - compiled.loss).detach().abs()),
    )
    gradient_errors = []
    for eager_parameter, compiled_parameter in zip(
        eager_backbone.parameters(), compiled_backbone.parameters(), strict=True
    ):
        if eager_parameter.grad is None or compiled_parameter.grad is None:
            if eager_parameter.grad is not compiled_parameter.grad:
                gradient_errors.append(float("inf"))
            continue
        gradient_errors.append(
            float((eager_parameter.grad - compiled_parameter.grad).abs().max())
        )
    gradient_error = max(gradient_errors, default=0.0)
    update_error = max(
        _parameter_error(eager_backbone, compiled_backbone),
        _parameter_error(eager_local, compiled_local),
    )
    checks = {
        "cuda_eager_compile_output": _check(output_error, 1e-5),
        "cuda_eager_compile_gradients": _check(gradient_error, 1e-5),
        "cuda_eager_compile_update": _check(update_error, 1e-5),
    }
    return checks, runtime


def run_qualification() -> dict:
    specification = load_specification()
    directory = RUN_ROOT / "qualification"
    with ProspectiveRun.start(
        directory,
        workflow_id="compact_global_local_model_v1",
        execution_id="qualification-920001",
        producer={
            "module": __name__,
            "protocol_sha256": PROTOCOL_SHA256,
            "repair_sha256": REPAIR_SHA256,
        },
        resolved_config={"seed": 920001, "scientific_seed": False},
    ):
        checks = {**cpu_qualification(), **gradient_qualification(specification)}
        cuda_checks, runtime = cuda_qualification(specification)
        checks.update(cuda_checks)
        result = {
            "schema_version": 1,
            "protocol_sha256": PROTOCOL_SHA256,
            "repair_sha256": REPAIR_SHA256,
            "seed": 920001,
            "liu_evaluated": False,
            "runtime": runtime,
            "checks": checks,
            "sources": implementation_sources(),
            "passed": all(row["passed"] for row in checks.values()),
        }
        write_json_exclusive(directory / "qualification.json", result)
    return result
