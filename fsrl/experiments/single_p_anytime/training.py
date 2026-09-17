"""Paired FH/VH final-checkpoint training for single-P anytime V1."""

from __future__ import annotations

import gc
import hashlib
import json
import time

import numpy as np
import torch

from fsrl.core.model_config import RetroModelConfig
from fsrl.experiments.clean_single_p.model import AffineSinglePSequence, map_shadow
from fsrl.experiments.clean_single_p.optimization import (
    clip_gradients,
    forward_batch,
    make_optimizer,
)
from fsrl.experiments.linear_modulation.model import LinearModulationRNN
from fsrl.experiments.pl_direct_training.execution import PROFILE, configure_execution
from fsrl.infra.provenance import tensor_hashes, write_json_exclusive
from fsrl.infra.run_manifest import ProspectiveRun
from fsrl.infra.runtime import compile_module

from .locks import (
    locked_schedule,
    reference,
    validate_source_lock,
    validate_training_run,
)
from .protocol import (
    PROTOCOL_SHA256,
    optimizer_specification,
    specification,
    training_directory,
)
from .streams import generate_max_stream, make_generator, prefix_batch


def _update_digest(digest: hashlib._Hash, *values: str) -> None:
    for value in values:
        digest.update(bytes.fromhex(value))


def train_one(seed: int, recipe: str, arm: str, source: dict) -> dict:
    directory = training_directory(seed, recipe, arm)
    if directory.exists():
        return validate_training_run(seed, recipe, arm)
    if recipe not in {"fixed_horizon", "variable_horizon"}:
        raise ValueError(f"unknown anytime recipe: {recipe}")
    if arm not in {"clean", "noisy"}:
        raise ValueError(f"unknown observation arm: {arm}")
    spec = specification()
    parent = optimizer_specification()
    schedule = locked_schedule(source, seed)
    torch.manual_seed(seed)
    shadow = LinearModulationRNN(
        RetroModelConfig(38, 200, 2, spec["training_schedule"]["batch_size"]),
        device="cuda",
    )
    initial_shadow = tensor_hashes(shadow)
    backbone = map_shadow(shadow, "clean_no_time")
    del shadow
    initial = tensor_hashes(backbone)
    sequence = compile_module(AffineSinglePSequence(backbone), PROFILE)
    optimizer = make_optimizer(backbone, parent)
    task = make_generator(source["task"], support_blocks=6)
    max_digest = hashlib.sha256()
    prefix_digest = hashlib.sha256()
    runtime = configure_execution()
    started = time.perf_counter()
    identity = {
        "seed": seed,
        "recipe": recipe,
        "arm": arm,
        "protocol_sha256": PROTOCOL_SHA256,
        "source_commit": source["source_commit"],
    }
    with ProspectiveRun.start(
        directory,
        workflow_id="single_p_anytime_v1",
        execution_id=f"train-{seed}-{recipe}-{arm}",
        producer=identity,
        resolved_config={
            "training_schedule": spec["training_schedule"],
            "optimization": spec["optimization"],
            "runtime": runtime,
        },
    ):
        warm_seconds = None
        with (directory / "train_log.jsonl").open("x", encoding="utf-8") as log:
            for step, (scheduled_b, edge_count) in enumerate(schedule):
                scheduled_b, edge_count = int(scheduled_b), int(edge_count)
                stream = generate_max_stream(
                    task,
                    network_seed=seed,
                    update=step,
                    edge_count=edge_count,
                    batch_size=spec["training_schedule"]["batch_size"],
                )
                clean_fingerprint = stream.clean.fingerprint()
                noisy_fingerprint = stream.noisy.fingerprint()
                retained_b = 4 if recipe == "fixed_horizon" else scheduled_b
                cpu = prefix_batch(
                    stream.clean if arm == "clean" else stream.noisy,
                    edge_count=edge_count,
                    blocks=retained_b,
                )
                prefix_fingerprint = cpu.fingerprint()
                _update_digest(
                    max_digest,
                    stream.task_fingerprint,
                    clean_fingerprint,
                    noisy_fingerprint,
                )
                max_digest.update(
                    np.asarray((scheduled_b, edge_count), dtype=np.int16).tobytes()
                )
                _update_digest(prefix_digest, prefix_fingerprint)
                batch, times = cpu.to("cuda")
                if warm_seconds is None:
                    before = tensor_hashes(backbone)
                    torch.cuda.synchronize()
                    begin = time.perf_counter()
                    warm = forward_batch(
                        "clean_no_time",
                        backbone,
                        sequence,
                        batch,
                        times,
                        penalty=parent["training"]["fast_weight_penalty"],
                    )
                    warm.loss.backward()
                    torch.cuda.synchronize()
                    warm_seconds = time.perf_counter() - begin
                    optimizer.zero_grad(set_to_none=True)
                    if tensor_hashes(backbone) != before:
                        raise RuntimeError("anytime warmup changed parameters")
                    del warm
                optimizer.zero_grad(set_to_none=True)
                result = forward_batch(
                    "clean_no_time",
                    backbone,
                    sequence,
                    batch,
                    times,
                    penalty=parent["training"]["fast_weight_penalty"],
                )
                result.loss.backward()
                gradient_norm = clip_gradients(backbone, 2.0)
                optimizer.step()
                torch.cuda.synchronize()
                fast = result.fast_weights.detach()
                row = {
                    "step": step,
                    "macro_cycle": step // 20,
                    "scheduled_B": scheduled_b,
                    "retained_B": retained_b,
                    "edge_count": edge_count,
                    "max_task_fingerprint": stream.task_fingerprint,
                    "max_clean_fingerprint": clean_fingerprint,
                    "max_noisy_fingerprint": noisy_fingerprint,
                    "prefix_fingerprint": prefix_fingerprint,
                    "max_stream_fingerprint": max_digest.hexdigest(),
                    "prefix_stream_fingerprint": prefix_digest.hexdigest(),
                    "loss": float(result.loss.detach()),
                    "query_loss": float(result.query_loss.detach()),
                    "terminal_penalty": float(
                        parent["training"]["fast_weight_penalty"] * fast.square().mean()
                    ),
                    "query_accuracy": float(
                        ((result.margins[:, 0] > 0).long() == batch.targets)
                        .float()
                        .mean()
                    ),
                    "P_frobenius": float(torch.linalg.matrix_norm(fast).mean()),
                    "effective_P_frobenius": float(
                        torch.linalg.matrix_norm(backbone.alpha.detach() * fast).mean()
                    ),
                    "clamp_fraction": float((fast.abs() == 50.0).float().mean()),
                    "weighted_backbone_gradient_norm": float(gradient_norm.detach()),
                    "prefix_q_sha256": hashlib.sha256(
                        cpu.arrays["realized_q"].tobytes()
                    ).hexdigest(),
                }
                log.write(json.dumps(row, allow_nan=False) + "\n")
                if step % 100 == 0 or step == len(schedule) - 1:
                    log.flush()
                    print(json.dumps({**identity, **row}), flush=True)
        checkpoint = directory / "model.pth"
        with checkpoint.open("xb") as handle:
            torch.save(
                {
                    "recipe": recipe,
                    "arm": arm,
                    "config": {"cue_size": 15, "hidden_size": 200},
                    "state_dict": backbone.state_dict(),
                },
                handle,
            )
        metadata = {
            **identity,
            "runtime": runtime,
            "initial_shadow": initial_shadow,
            "initial_model": initial,
            "final_model": tensor_hashes(backbone),
            "parameters": sum(value.numel() for value in backbone.parameters()),
            "schedule_sha256": source["schedules"][str(seed)]["sha256"],
            "max_stream_fingerprint": max_digest.hexdigest(),
            "prefix_stream_fingerprint": prefix_digest.hexdigest(),
            "optimizer_steps": {
                name: int(optimizer.state.get(value, {}).get("step", 0))
                for name, value in backbone.named_parameters()
            },
            "episode_exposures": (
                spec["training_schedule"]["updates"]
                * spec["training_schedule"]["batch_size"]
            ),
            "warmup_seconds": warm_seconds,
            "training_seconds": time.perf_counter() - started - (warm_seconds or 0.0),
            "checkpoint": reference(checkpoint),
        }
        write_json_exclusive(directory / "result.json", metadata)
    return validate_training_run(seed, recipe, arm)


def train_all() -> dict:
    source, spec = validate_source_lock(), specification()
    completed = []
    for index, seed in enumerate(spec["design"]["network_seeds"]):
        recipes = spec["design"]["training_recipes"][:: 1 if index % 2 == 0 else -1]
        for recipe in recipes:
            arms = spec["design"]["observation_training_arms"][
                :: 1 if recipe == "fixed_horizon" else -1
            ]
            for arm in arms:
                train_one(seed, recipe, arm, source)
                completed.append(f"{seed}/{recipe}/{arm}")
                gc.collect()
                torch.cuda.empty_cache()
    return {"completed": completed, "scientific_outcomes_exposed": False}


__all__ = ["train_all", "train_one"]
