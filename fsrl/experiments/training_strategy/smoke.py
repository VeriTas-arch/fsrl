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
from fsrl.evaluation.contracts import FrozenEvaluationBackend
from fsrl.evaluation.frozen_fast_weight import FrozenFastWeightEvaluator
from fsrl.evaluation.relational_query import readout_relational_query_bundle
from fsrl.infra.provenance import write_json_exclusive
from fsrl.infra.run_manifest import ProspectiveRun
from fsrl.infra.runtime import compile_module
from fsrl.tasks.protocol import RankingProtocol, ordered_pairs
from fsrl.training.backbone import make_model_and_tasks

from .batches import TensorBatch, prepare_batch, sample_episodes
from .execution import PROFILE, configure_execution
from .liu_rollout import readout_bundle
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
    model_config, backbone, tasks = make_model_and_tasks(config, device="cuda")
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
        checks.update(evaluation_parity(backbone, local, model_config))
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


def evaluation_parity(backbone, local, model_config) -> dict:
    """Non-Liu synthetic query interface, with no scientific outcome measured."""

    protocol = RankingProtocol(
        "synthetic-cuda-query-parity",
        tuple(str(i) for i in range(8)),
        tuple(range(8)),
        tuple((i, i + 1) for i in range(7)) + ((0, 2),),
        1,
        2,
        {},
    )
    evaluator = FrozenFastWeightEvaluator(
        backbone,
        model_config,
        protocol,
        cue_seed=910001,
        support_seed=910001,
        backend=FrozenEvaluationBackend.BATCHED_SEQUENCE,
        execution_profile=PROFILE,
    )
    weights = (
        torch.randn(model_config.bs, model_config.hs, model_config.hs, device="cuda")
        * 0.1
    )
    state = torch.randn(model_config.bs, local.cue_size**2, device="cuda") * 0.1
    shuffled = np.broadcast_to(
        np.roll(np.arange(56).reshape(28, 2), 1, axis=0).reshape(56),
        (model_config.bs, 56),
    ).copy()
    checks = {}
    for name, local_off, global_off, indices in (
        ("intact", False, False, None),
        ("local_off", True, False, None),
        ("P_off", False, True, None),
        ("query_shuffle", False, False, shuffled),
    ):
        compiled = readout_bundle(
            evaluator,
            local,
            weights,
            state,
            local_off=local_off,
            global_off=global_off,
            shuffled_indices=indices,
        )
        eager = readout_relational_query_bundle(
            evaluator,
            local,
            weights,
            state,
            (ordered_pairs(8),) * model_config.bs,
            local_off=local_off,
            global_off=global_off,
            shuffled_indices=indices,
        )
        for key, values in compiled.items():
            np.testing.assert_allclose(values, eager[key], atol=1e-5, rtol=1e-4)
            checks[f"evaluation_{name}_{key}"] = {
                "passed": True,
                "max_abs_error": float(np.max(np.abs(values - eager[key]))),
            }
    return checks
