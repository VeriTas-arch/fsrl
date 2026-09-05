"""Non-Liu CUDA parity and support-gradient audit before source locking."""

from __future__ import annotations

import copy
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from fsrl.core.local_trace import ConjunctiveLocalTrace
from fsrl.core.sequence import RecurrentSequence
from fsrl.infra.provenance import write_json_exclusive
from fsrl.infra.run_manifest import ProspectiveRun
from fsrl.infra.runtime import compile_module
from fsrl.training.backbone import make_model_and_tasks

from .batches import TensorBatch, prepare_batch, sample_episodes
from .execution import PROFILE, configure_execution
from .locks import implementation_sources
from .optimization import forward_batch, make_optimizer, query_from_state, training_step
from .protocol import PROTOCOL_SHA256, load_specification, training_config


def _one_engine(backbone, local, sequence, batch, optimization):
    result = forward_batch(
        backbone, local, sequence, batch, local_active=True, fast_weight_penalty=0.0
    )
    tensors = (
        result.fast_weights,
        result.first_support_write,
        backbone.h2DA.weight,
        local.raw_gain,
    )
    gradients = torch.autograd.grad(result.query_loss, tensors)
    for gradient in gradients:
        if not torch.isfinite(gradient).all() or gradient.abs().max() == 0:
            raise RuntimeError(
                "query CE does not reach finite nonzero support/gain gradients"
            )
    query_size = batch.targets.numel()
    size = result.fast_weights.shape[0]
    per_query = []
    with torch.no_grad():
        for start in range(0, query_size, size):
            small = TensorBatch(
                batch.support_inputs,
                batch.local_evidence,
                batch.query_inputs[:, start : start + size],
                batch.targets[start : start + size],
            )
            logits, _ = query_from_state(
                backbone,
                local,
                RecurrentSequence(backbone),
                small,
                result.fast_weights.detach(),
                result.local_state,
                local_active=True,
            )
            per_query.append(logits)
    torch.testing.assert_close(
        result.logits, torch.cat(per_query), atol=1e-5, rtol=1e-4
    )
    before = {
        "logits": result.logits.detach().cpu(),
        "P_T": result.fast_weights.detach().cpu(),
        "loss": result.query_loss.detach().cpu(),
    }
    before.update(
        {
            f"gradient_{index}": value.detach().cpu()
            for index, value in enumerate(gradients)
        }
    )
    optimizer = make_optimizer(backbone, local, optimization)
    training_step(
        backbone,
        local,
        sequence,
        batch,
        optimizer,
        phase="joint",
        optimization=optimization,
    )
    before.update(
        {
            f"updated_backbone_{key}": value.detach().cpu()
            for key, value in backbone.state_dict().items()
        }
    )
    before["updated_raw_gain"] = local.raw_gain.detach().cpu()
    return before


def run_smoke(directory: Path) -> dict:
    specification = load_specification()
    runtime = configure_execution()
    config = replace(training_config(specification, 910001), batch_size=2)
    torch.manual_seed(910001)
    _, backbone, tasks = make_model_and_tasks(config, device="cuda")
    local = ConjunctiveLocalTrace(config.cue_size, device="cuda")
    second_backbone, second_local = copy.deepcopy(backbone), copy.deepcopy(local)
    cpu_batch = prepare_batch(sample_episodes(tasks, np.random.default_rng(910001), 2))
    batch = cpu_batch.to("cuda")
    started = time.perf_counter()
    with ProspectiveRun.start(
        directory,
        workflow_id="joint_training_strategy_v1",
        execution_id="cuda-smoke-910001",
        producer={"module": __name__, "protocol_sha256": PROTOCOL_SHA256},
        resolved_config={"training": config.__dict__, "runtime": runtime},
    ):
        eager = _one_engine(
            backbone,
            local,
            RecurrentSequence(backbone),
            batch,
            specification["optimization"],
        )
        compiled = _one_engine(
            second_backbone,
            second_local,
            compile_module(RecurrentSequence(second_backbone), PROFILE),
            batch,
            specification["optimization"],
        )
        checks = {}
        for key in eager:
            torch.testing.assert_close(eager[key], compiled[key], atol=1e-5, rtol=1e-4)
            checks[key] = {
                "passed": True,
                "max_abs_error": float((eager[key] - compiled[key]).abs().max()),
            }
        result = {
            "passed": True,
            "seed": 910001,
            "runtime": runtime,
            "sources": implementation_sources(),
            "batch_fingerprint": cpu_batch.fingerprint(),
            "checks": checks,
            "seconds": time.perf_counter() - started,
            "liu_evaluated": False,
        }
        write_json_exclusive(directory / "smoke.json", result)
    return result
