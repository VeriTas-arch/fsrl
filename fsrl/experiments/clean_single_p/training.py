"""Paired final-checkpoint training for the clean single-P study."""

from __future__ import annotations

import gc
import hashlib
import json
import time

import numpy as np
import torch

from fsrl.core.model_config import RetroModelConfig
from fsrl.experiments.linear_modulation.model import LinearModulationRNN
from fsrl.experiments.pl_direct_training.batches import prepare_batch
from fsrl.experiments.pl_direct_training.execution import PROFILE, configure_execution
from fsrl.experiments.pl_direct_training.task import make_task_generator
from fsrl.experiments.training_strategy.batches import sample_episodes
from fsrl.infra.provenance import tensor_hashes, write_json_exclusive
from fsrl.infra.run_manifest import ProspectiveRun
from fsrl.infra.runtime import compile_module

from .batches import prepare_single_p
from .locks import reference, validate_source_lock, validate_training_run
from .model import AffineSinglePSequence, map_shadow
from .optimization import forward_batch, make_optimizer, training_step
from .protocol import PROTOCOL_SHA256, specification, training_directory


def train_one(seed: int, condition: str, arm: str, source: dict) -> dict:
    directory = training_directory(seed, condition, arm)
    if directory.exists():
        return validate_training_run(seed, condition, arm)
    spec = specification()
    torch.manual_seed(seed)
    shadow = LinearModulationRNN(
        RetroModelConfig(38, 200, 2, spec["training"]["batch_size"]),
        device="cuda",
    )
    initial_shadow = tensor_hashes(shadow)
    backbone = map_shadow(shadow, condition)
    del shadow
    initial = tensor_hashes(backbone)
    sequence = compile_module(AffineSinglePSequence(backbone), PROFILE)
    optimizer = make_optimizer(backbone, spec)
    task = make_task_generator({**spec, "task": source["task"]})
    rng = np.random.default_rng(151000 + seed)
    digest = hashlib.sha256()
    runtime = configure_execution()
    started = time.perf_counter()
    identity = {
        "seed": seed,
        "condition": condition,
        "arm": arm,
        "protocol_sha256": PROTOCOL_SHA256,
        "source_commit": source["source_commit"],
    }
    with ProspectiveRun.start(
        directory,
        workflow_id="clean_single_p_v1",
        execution_id=f"train-{seed}-{condition}-{arm}",
        producer=identity,
        resolved_config={"training": spec["training"], "runtime": runtime},
    ):
        warm = None
        with (directory / "train_log.jsonl").open("x", encoding="utf-8") as log:
            for step in range(spec["training"]["steps"]):
                episodes = sample_episodes(
                    task, rng, spec["training"]["batch_size"], validation=False
                )
                batch_fingerprint = prepare_batch(episodes).fingerprint()
                observation_seed = 410000000 + seed * 10000 + step
                cpu = prepare_single_p(episodes, arm, observation_seed=observation_seed)
                observed_fingerprint = cpu.fingerprint()
                digest.update(bytes.fromhex(batch_fingerprint))
                batch, times = cpu.to("cuda")
                if warm is None:
                    before = tensor_hashes(backbone)
                    torch.cuda.synchronize()
                    begin = time.perf_counter()
                    trial = forward_batch(
                        condition,
                        backbone,
                        sequence,
                        batch,
                        times,
                        penalty=spec["training"]["fast_weight_penalty"],
                    )
                    trial.loss.backward()
                    torch.cuda.synchronize()
                    warm = time.perf_counter() - begin
                    optimizer.zero_grad(set_to_none=True)
                    if tensor_hashes(backbone) != before:
                        raise RuntimeError("warmup changed clean single-P parameters")
                    del trial
                result = training_step(
                    condition,
                    backbone,
                    sequence,
                    batch,
                    times,
                    optimizer,
                    spec,
                )
                torch.cuda.synchronize()
                row = {
                    "step": step,
                    "batch_fingerprint": batch_fingerprint,
                    "observed_fingerprint": observed_fingerprint,
                    "stream_fingerprint": digest.hexdigest(),
                    "observation_seed": observation_seed,
                    "q_sha256": hashlib.sha256(
                        cpu.arrays["realized_q"].tobytes()
                    ).hexdigest(),
                    "loss": float(result.loss.detach()),
                    "query_loss": float(result.query_loss.detach()),
                    "query_accuracy": float(
                        ((result.margins[:, 0] > 0).long() == batch.targets)
                        .float()
                        .mean()
                    ),
                    "mean_abs_P": float(result.fast_weights.detach().abs().mean()),
                }
                log.write(json.dumps(row, allow_nan=False) + "\n")
                if step % 100 == 0 or step == spec["training"]["steps"] - 1:
                    log.flush()
                    print(json.dumps({**identity, **row}), flush=True)
        checkpoint = directory / "model.pth"
        with checkpoint.open("xb") as handle:
            torch.save(
                {
                    "condition": condition,
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
            "stream_fingerprint": digest.hexdigest(),
            "optimizer_steps": {
                name: int(optimizer.state.get(value, {}).get("step", 0))
                for name, value in backbone.named_parameters()
            },
            "episode_exposures": (
                spec["training"]["steps"] * spec["training"]["batch_size"]
            ),
            "warmup_seconds": warm,
            "training_seconds": time.perf_counter() - started - (warm or 0.0),
            "checkpoint": reference(checkpoint),
        }
        write_json_exclusive(directory / "result.json", metadata)
    return validate_training_run(seed, condition, arm)


def train_all() -> dict:
    source, spec = validate_source_lock(), specification()
    completed = []
    for index, seed in enumerate(spec["design"]["network_seeds"]):
        conditions = spec["design"]["architecture_conditions"][
            :: 1 if index % 2 == 0 else -1
        ]
        for condition in conditions:
            arms = spec["design"]["observation_training_arms"][
                :: 1 if condition == "time_retained_control" else -1
            ]
            for arm in arms:
                train_one(seed, condition, arm, source)
                completed.append(f"{seed}/{condition}/{arm}")
                gc.collect()
                torch.cuda.empty_cache()
    return {"completed": completed, "scientific_outcomes_exposed": False}


__all__ = ["train_all", "train_one"]
