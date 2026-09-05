"""Final-step-only paired training with complete data and compute accounting."""

from __future__ import annotations

import gc
import hashlib
import json
import time
from dataclasses import asdict

import numpy as np
import torch

from fsrl.core.local_trace import ConjunctiveLocalTrace
from fsrl.core.sequence import RecurrentSequence
from fsrl.infra.provenance import tensor_hashes, write_json_exclusive
from fsrl.infra.run_manifest import ProspectiveRun
from fsrl.infra.runtime import compile_module
from fsrl.training.backbone import make_model_and_tasks

from .batches import prepare_batch, sample_episodes
from .execution import PROFILE, configure_execution
from .locks import reference, run_directory, validate_source_lock, validate_training_run
from .optimization import configure_phase, forward_batch, make_optimizer, training_step
from .protocol import (
    PROTOCOL_SHA256,
    load_specification,
    phase_for_step,
    training_config,
)


def warm_phase(backbone, local, sequence, batch, phase: str) -> float:
    configure_phase(backbone, local, phase)
    before = (tensor_hashes(backbone), tensor_hashes(local))
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = forward_batch(
        backbone,
        local,
        sequence,
        batch,
        local_active=phase != "global",
        fast_weight_penalty=0.0,
    )
    result.loss.backward()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    configure_phase(backbone, local, phase)
    if before != (tensor_hashes(backbone), tensor_hashes(local)):
        raise RuntimeError("compiler warmup changed parameters")
    return elapsed


def _train_steps(
    specification, condition, backbone, local, sequence, tasks, rng, directory
):
    optimization = specification["optimization"]
    optimizer = make_optimizer(backbone, local, optimization)
    digest = hashlib.sha256()
    phases = {}
    previous_phase = None
    boundary = None
    with (directory / "train_log.jsonl").open("x", encoding="utf-8") as handle:
        for step in range(optimization["total_steps"]):
            phase = phase_for_step(specification, condition, step)
            start = time.perf_counter()
            episodes = sample_episodes(tasks, rng, optimization["batch_size"])
            cpu_batch = prepare_batch(episodes)
            fingerprint = cpu_batch.fingerprint()
            digest.update(bytes.fromhex(fingerprint))
            batch = cpu_batch.to("cuda")
            if phase != previous_phase:
                torch.cuda.reset_peak_memory_stats()
                phases[phase] = {
                    "warmup_seconds": warm_phase(
                        backbone, local, sequence, batch, phase
                    ),
                    "steps": 0,
                    "training_seconds": 0.0,
                    "trainable_backbone_parameters": sum(
                        p.numel() for p in backbone.parameters() if p.requires_grad
                    ),
                    "trainable_local_parameters": sum(
                        p.numel() for p in local.parameters() if p.requires_grad
                    ),
                    "total_backbone_parameters": sum(
                        p.numel() for p in backbone.parameters()
                    ),
                    "total_local_parameters": sum(
                        p.numel() for p in local.parameters()
                    ),
                    "persistent_P_scalars_per_episode": backbone.model_config.hidden_size
                    ** 2,
                    "persistent_L_scalars_per_episode": local.cue_size**2,
                }
                if phase == "local":
                    boundary = tensor_hashes(backbone)
                previous_phase = phase
            result = training_step(
                backbone,
                local,
                sequence,
                batch,
                optimizer,
                phase=phase,
                optimization=optimization,
            )
            metrics = (
                torch.stack(
                    (
                        result.loss.detach(),
                        result.query_loss.detach(),
                        (result.logits.argmax(1) == batch.targets).float().mean(),
                        local.gain.detach().reshape(()),
                    )
                )
                .cpu()
                .tolist()
            )
            elapsed = time.perf_counter() - start
            if phases[phase]["steps"] == 0:
                elapsed -= phases[phase]["warmup_seconds"]
            phases[phase]["steps"] += 1
            phases[phase]["training_seconds"] += elapsed
            phases[phase].update(
                {
                    "episode_exposures": phases[phase]["steps"]
                    * optimization["batch_size"],
                    "total_seconds": phases[phase]["training_seconds"]
                    + phases[phase]["warmup_seconds"],
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                }
            )
            row = {
                "step": step,
                "phase": phase,
                "batch_fingerprint": fingerprint,
                "stream_fingerprint": digest.hexdigest(),
                "loss": metrics[0],
                "query_cross_entropy": metrics[1],
                "query_accuracy": metrics[2],
                "local_gain": metrics[3],
                "seconds": elapsed,
            }
            handle.write(json.dumps(row, allow_nan=False) + "\n")
            if step % 50 == 0 or step == optimization["total_steps"] - 1:
                handle.flush()
                print(json.dumps({"condition": condition, **row}), flush=True)
            del result, batch
    optimizer_steps = {
        f"{prefix}.{name}": int(optimizer.state.get(parameter, {}).get("step", 0))
        for prefix, module in (("backbone", backbone), ("local", local))
        for name, parameter in module.named_parameters()
    }
    return {
        "phases": phases,
        "stream_fingerprint": digest.hexdigest(),
        "stage_boundary_backbone": boundary,
        "optimizer_parameter_steps": optimizer_steps,
        "backbone_updates": max(
            count
            for name, count in optimizer_steps.items()
            if name.startswith("backbone.")
        ),
        "local_updates": optimizer_steps["local.raw_gain"],
    }


def train_one(
    specification: dict, seed: int, condition: str, source_lock: dict
) -> dict:
    directory = run_directory(seed, condition)
    if directory.exists():
        return validate_training_run(directory, specification)
    runtime = configure_execution()
    torch.manual_seed(seed)
    config = training_config(specification, seed)
    _, backbone, tasks = make_model_and_tasks(config, device="cuda")
    local = ConjunctiveLocalTrace(
        config.cue_size,
        initial_gain=specification["optimization"]["initial_local_gain"],
        device="cuda",
    )
    initial_backbone, initial_local = tensor_hashes(backbone), tensor_hashes(local)
    sequence = compile_module(RecurrentSequence(backbone), PROFILE)
    rng = np.random.default_rng(100000 + seed)
    producer = {
        "module": __name__,
        "source_commit": source_lock["source_commit"],
        "protocol_sha256": PROTOCOL_SHA256,
    }
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with ProspectiveRun.start(
        directory,
        workflow_id="joint_training_strategy_v1",
        execution_id=f"{seed}-{condition}",
        producer=producer,
        resolved_config={
            "seed": seed,
            "condition": condition,
            "training": asdict(config),
            "optimization": specification["optimization"],
            "runtime": runtime,
        },
    ):
        stats = _train_steps(
            specification, condition, backbone, local, sequence, tasks, rng, directory
        )
        checkpoint = directory / "net.pth"
        with checkpoint.open("xb") as handle:
            torch.save(backbone.state_dict(), handle)
        with (directory / "local.pth").open("xb") as handle:
            torch.save(local.state_dict(), handle)
        metadata = {
            "schema_version": 1,
            "protocol_sha256": PROTOCOL_SHA256,
            "seed": seed,
            "condition": condition,
            "training": asdict(config),
            "model": asdict(backbone.model_config),
            "task_distribution": {
                "liu_graph_held_out": True,
                "held_out_rank_graph_signatures": [
                    [list(pair) for pair in graph]
                    for graph in sorted(tasks.excluded_signatures)
                ],
                "validation_graph_partition": "sha256-reflection-canonical bucket 0 excluded",
            },
            "runtime": runtime,
            "initial_backbone": initial_backbone,
            "initial_local": initial_local,
            "final_backbone": tensor_hashes(backbone),
            "final_local": tensor_hashes(local),
            "raw_gain": float(local.raw_gain.detach()),
            "local_gain": float(local.gain.detach()),
            "episode_exposures": config.batch_size * config.outer_steps,
            "checkpoint": reference(checkpoint),
            "cost": {
                "total_seconds": time.perf_counter() - start,
                "warm_training_seconds": sum(
                    phase["training_seconds"] for phase in stats["phases"].values()
                ),
                "warmup_seconds": sum(
                    phase["warmup_seconds"] for phase in stats["phases"].values()
                ),
                "peak_allocated_bytes": max(
                    phase["peak_allocated_bytes"] for phase in stats["phases"].values()
                ),
                "peak_reserved_bytes": max(
                    phase["peak_reserved_bytes"] for phase in stats["phases"].values()
                ),
                "backbone_parameters": sum(p.numel() for p in backbone.parameters()),
                "local_parameters": local.raw_gain.numel(),
                "persistent_P_scalars_per_episode": config.hidden_size**2,
                "persistent_L_scalars_per_episode": config.cue_size**2,
            },
            **stats,
        }
        if (
            condition == "matched_staged"
            and metadata["stage_boundary_backbone"] != metadata["final_backbone"]
        ):
            raise RuntimeError("staged backbone moved during local adaptation")
        write_json_exclusive(directory / "config.json", metadata)
    return validate_training_run(directory, specification)


def train_all() -> dict:
    specification = load_specification()
    source = validate_source_lock()
    completed = {}
    for seed in specification["seeds"]["mandatory"]:
        for condition in specification["execution"]["condition_execution_order"][
            str(seed)
        ]:
            completed[f"{seed}/{condition}"] = train_one(
                specification, seed, condition, source
            )
            gc.collect()
            torch.cuda.empty_cache()
    return {"completed": sorted(completed), "liu_evaluated": False}
