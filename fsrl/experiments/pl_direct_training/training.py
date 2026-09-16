"""Final-step-only paired direct training with complete stream accounting."""

from __future__ import annotations

import gc
import hashlib
import json
import time
from dataclasses import asdict

import numpy as np
import torch

from fsrl.core.factorized_plastic_rnn import (
    FactorizedPlasticRNNConfig,
    FactorizedRecurrentSequence,
)
from fsrl.core.local_trace import PackedConjunctiveLocalTrace
from fsrl.core.model_config import RetroModelConfig
from fsrl.core.plastic_rnn import RetroModulRNN
from fsrl.infra.provenance import tensor_hashes, write_json_exclusive
from fsrl.infra.run_manifest import ProspectiveRun
from fsrl.infra.runtime import compile_module

from .batches import prepare_batch, sample_episodes
from .execution import PROFILE, configure_execution
from .locks import reference, run_directory, validate_source_lock, validate_training_run
from .model import NoTimePlasticRNN, NoTimeRecurrentSequence, map_shadow_model
from .optimization import forward_batch, make_optimizer, training_step
from .protocol import PROTOCOL_SHA256, REPAIR_SHA256, load_specification
from .task import make_task_generator


def _sequence(backbone):
    module = (
        NoTimeRecurrentSequence(backbone)
        if isinstance(backbone, NoTimePlasticRNN)
        else FactorizedRecurrentSequence(backbone)
    )
    return compile_module(module, PROFILE)


def _shared_hashes(backbone) -> dict:
    return {
        name: value
        for name, value in tensor_hashes(backbone).items()
        if name != "time_weight"
    }


def _warm_sequence(condition, backbone, local, sequence, batch, optimization) -> float:
    before = (tensor_hashes(backbone), tensor_hashes(local))
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = forward_batch(
        condition,
        backbone,
        local,
        sequence,
        batch,
        fast_weight_penalty=optimization["fast_weight_penalty"],
    )
    result.loss.backward()
    for parameter in (*backbone.parameters(), *local.parameters()):
        parameter.grad = None
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    if before != (tensor_hashes(backbone), tensor_hashes(local)):
        raise RuntimeError("compiler warmup changed direct-training parameters")
    return elapsed


def _train_steps(
    specification, condition, backbone, local, sequence, task, rng, directory
):
    optimization = specification["optimization"]
    optimizer = make_optimizer(backbone, local, optimization)
    digest = hashlib.sha256()
    warmup_seconds = None
    training_seconds = 0.0
    torch.cuda.reset_peak_memory_stats()
    with (directory / "train_log.jsonl").open("x", encoding="utf-8") as handle:
        for step in range(optimization["total_steps"]):
            started = time.perf_counter()
            episodes = sample_episodes(
                task, rng, optimization["batch_size"], validation=False
            )
            cpu_batch = prepare_batch(episodes)
            fingerprint = cpu_batch.fingerprint()
            digest.update(bytes.fromhex(fingerprint))
            batch = cpu_batch.to("cuda")
            if warmup_seconds is None:
                warmup_seconds = _warm_sequence(
                    condition, backbone, local, sequence, batch, optimization
                )
                started = time.perf_counter()
            result = training_step(
                condition,
                backbone,
                local,
                sequence,
                batch,
                optimizer,
                optimization=optimization,
            )
            metrics = (
                torch.stack(
                    (
                        result.loss.detach(),
                        result.query_loss.detach(),
                        ((result.margins[:, 0] > 0).long() == batch.targets)
                        .float()
                        .mean(),
                        local.gain.detach().reshape(()),
                        result.fast_weights.detach().abs().mean(),
                    )
                )
                .cpu()
                .tolist()
            )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            training_seconds += elapsed
            row = {
                "step": step,
                "batch_fingerprint": fingerprint,
                "stream_fingerprint": digest.hexdigest(),
                "loss": metrics[0],
                "query_loss": metrics[1],
                "query_accuracy": metrics[2],
                "local_gain": metrics[3],
                "mean_abs_fast_weight": metrics[4],
                "seconds": elapsed,
            }
            handle.write(json.dumps(row, allow_nan=False) + "\n")
            if step % 50 == 0 or step == optimization["total_steps"] - 1:
                handle.flush()
                print(json.dumps({"condition": condition, **row}), flush=True)
            del result, batch
    if warmup_seconds is None:
        raise RuntimeError("the registered training budget contained no updates")
    optimizer_steps = {
        f"{prefix}.{name}": int(optimizer.state.get(parameter, {}).get("step", 0))
        for prefix, module in (("backbone", backbone), ("local", local))
        for name, parameter in module.named_parameters()
    }
    return {
        "stream_fingerprint": digest.hexdigest(),
        "optimizer_parameter_steps": optimizer_steps,
        "training_seconds": training_seconds,
        "warmup_seconds": warmup_seconds,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def train_one(
    specification: dict,
    seed: int,
    condition: str,
    source_lock: dict,
    cohort: str,
) -> dict:
    directory = run_directory(seed, condition, cohort)
    if directory.exists():
        return validate_training_run(directory, specification, cohort)
    runtime = configure_execution()
    torch.manual_seed(seed)
    architecture = specification["architecture"]["common"]
    shadow = RetroModulRNN(
        RetroModelConfig(
            input_size=37,
            hidden_size=architecture["hidden_size"],
            output_size=2,
            batch_size=specification["optimization"]["batch_size"],
        ),
        device="cuda",
    )
    initial_shadow = tensor_hashes(shadow)
    backbone = map_shadow_model(shadow, condition)
    del shadow
    local = PackedConjunctiveLocalTrace(
        specification["task"]["cue_size"],
        initial_gain=specification["optimization"]["initial_local_gain"],
        device="cuda",
    )
    initial_backbone = tensor_hashes(backbone)
    initial_shared = _shared_hashes(backbone)
    initial_local = tensor_hashes(local)
    sequence = _sequence(backbone)
    task = make_task_generator(specification)
    rng = np.random.default_rng(100000 + seed)
    producer = {
        "module": __name__,
        "source_commit": source_lock["source_commit"],
        "protocol_sha256": PROTOCOL_SHA256,
        "repair_sha256": REPAIR_SHA256,
    }
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with ProspectiveRun.start(
        directory,
        workflow_id="pl_direct_training_v1",
        execution_id=f"{cohort}-{seed}-{condition}",
        producer=producer,
        resolved_config={
            "seed": seed,
            "condition": condition,
            "cohort": cohort,
            "optimization": specification["optimization"],
            "runtime": runtime,
        },
    ):
        stats = _train_steps(
            specification,
            condition,
            backbone,
            local,
            sequence,
            task,
            rng,
            directory,
        )
        checkpoint = directory / "model.pth"
        with checkpoint.open("xb") as handle:
            torch.save(
                {
                    "condition": condition,
                    "model_config": asdict(
                        FactorizedPlasticRNNConfig(
                            cue_size=15,
                            hidden_size=architecture["hidden_size"],
                        )
                    ),
                    "backbone": backbone.state_dict(),
                    "local": local.state_dict(),
                },
                handle,
            )
        metadata = {
            "schema_version": 1,
            "protocol_sha256": PROTOCOL_SHA256,
            "repair_sha256": REPAIR_SHA256,
            "source_commit": source_lock["source_commit"],
            "seed": seed,
            "condition": condition,
            "cohort": cohort,
            "runtime": runtime,
            "architecture": {
                "task_input_size": backbone.model_config.input_size,
                "hidden_size": backbone.model_config.hidden_size,
                "support_steps": architecture["support_trial_steps"],
                "query_steps": architecture["query_trial_steps"],
                "P_scalars": architecture["persistent_state_sizes"]["P"],
                "L_scalars": architecture["persistent_state_sizes"]["L"],
                "backbone_parameters": sum(
                    value.numel() for value in backbone.parameters()
                ),
                "time_parameters": (
                    architecture["hidden_size"]
                    if condition == "time_retained_control"
                    else 0
                ),
                "compatibility_buffers": len(list(backbone.named_buffers())),
            },
            "optimization": specification["optimization"],
            "initial_shadow": initial_shadow,
            "initial_backbone": initial_backbone,
            "initial_shared": initial_shared,
            "initial_local": initial_local,
            "final_backbone": tensor_hashes(backbone),
            "final_local": tensor_hashes(local),
            "raw_gain": float(local.raw_gain.detach()),
            "local_gain": float(local.gain.detach()),
            "episode_exposures": (
                specification["optimization"]["batch_size"]
                * specification["optimization"]["total_steps"]
            ),
            "checkpoint": reference(checkpoint),
            "cost": {
                "total_seconds": time.perf_counter() - started,
                "backbone_parameters": sum(
                    value.numel() for value in backbone.parameters()
                ),
                "local_parameters": local.raw_gain.numel(),
                "persistent_P_scalars_per_episode": architecture[
                    "persistent_state_sizes"
                ]["P"],
                "persistent_L_scalars_per_episode": architecture[
                    "persistent_state_sizes"
                ]["L"],
            },
            **stats,
        }
        write_json_exclusive(directory / "config.json", metadata)
    return validate_training_run(directory, specification, cohort)


def train_cohort(cohort: str) -> dict:
    specification = load_specification()
    if cohort == "confirmation":
        from fsrl.infra.provenance import load_json

        from .reporting import result_path

        development = load_json(result_path("development"))
        if development["outcome"] != "development_admitted":
            raise RuntimeError(
                "reserved confirmation requires complete development admission"
            )
    source = validate_source_lock()
    from .protocol import registered_seeds

    completed = {}
    for seed in registered_seeds(specification, cohort):
        order = specification["execution"]["condition_order"].get(
            str(seed), specification["design"]["conditions"]
        )
        for condition in order:
            completed[f"{seed}/{condition}"] = train_one(
                specification, seed, condition, source, cohort
            )
            gc.collect()
            torch.cuda.empty_cache()
    return {
        "cohort": cohort,
        "completed": sorted(completed),
        "liu_evaluated": False,
    }
