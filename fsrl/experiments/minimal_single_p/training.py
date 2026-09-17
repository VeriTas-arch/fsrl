"""Final-checkpoint training for one authorized minimal single-P level."""

from __future__ import annotations

import gc
import hashlib
import json
import time

import numpy as np
import torch

from fsrl.experiments.clean_single_p.batches import prepare_single_p
from fsrl.experiments.pl_direct_training.execution import PROFILE, configure_execution
from fsrl.experiments.pl_direct_training.task import make_task_generator
from fsrl.experiments.training_strategy.batches import sample_episodes
from fsrl.infra.provenance import tensor_hashes, write_json_exclusive
from fsrl.infra.run_manifest import ProspectiveRun
from fsrl.infra.runtime import compile_module

from .locks import (
    reference,
    validate_predecessor,
    validate_source_lock,
    validate_training_run,
)
from .model import MinimalSinglePSequence, level_settings, make_model
from .optimization import forward_batch, make_optimizer, training_step
from .protocol import PROTOCOL_SHA256, specification, training_directory, validate_level


def _shared_hashes(model) -> dict[str, str]:
    names = {
        "cue_projection.weight",
        "cue_projection.bias",
        "evidence_weight",
        "w",
        "h2margin.weight",
        "h2margin.bias",
    }
    return {
        name: value for name, value in tensor_hashes(model).items() if name in names
    }


def _penalty(level: str) -> float:
    return 1e-4 if level == "C0" else 0.0


def train_one(seed: int, level: str, source: dict) -> dict:
    level = validate_level(level)
    directory = training_directory(seed, level)
    if directory.exists():
        return validate_training_run(seed, level)
    spec = specification()
    if seed not in spec["design"]["network_seeds"]:
        raise ValueError("unregistered minimal single-P seed")
    model = make_model(level, seed, device="cuda")
    initial_model = tensor_hashes(model)
    initial_shared = _shared_hashes(model)
    sequence = compile_module(MinimalSinglePSequence(model), PROFILE)
    optimizer = make_optimizer(model, spec["training"]["learning_rate"])
    task = make_task_generator({"task": source["task"]})
    rng = np.random.default_rng(151000 + seed)
    digest = hashlib.sha256()
    runtime = configure_execution()
    penalty = _penalty(level)
    identity = {
        "seed": seed,
        "level": level,
        "protocol_sha256": PROTOCOL_SHA256,
        "source_commit": source["source_commit"],
    }
    started = time.perf_counter()
    with ProspectiveRun.start(
        directory,
        workflow_id="minimal_single_p_v1",
        execution_id=f"train-{level}-{seed}",
        producer=identity,
        resolved_config={
            "level": spec["architecture"]["levels"][level],
            "training": spec["training"],
            "runtime": runtime,
        },
    ):
        warm_seconds = None
        with (directory / "train_log.jsonl").open("x", encoding="utf-8") as log:
            for step in range(spec["training"]["updates"]):
                episodes = sample_episodes(
                    task, rng, spec["training"]["batch_size"], validation=False
                )
                cpu = prepare_single_p(
                    episodes,
                    "clean",
                    observation_seed=910000000 + seed * 10000 + step,
                )
                fingerprint = cpu.fingerprint()
                digest.update(bytes.fromhex(fingerprint))
                batch, _ = cpu.to("cuda")
                if warm_seconds is None:
                    before = tensor_hashes(model)
                    torch.cuda.synchronize()
                    begin = time.perf_counter()
                    warm = forward_batch(model, sequence, batch, penalty=penalty)
                    warm.loss.backward()
                    torch.cuda.synchronize()
                    warm_seconds = time.perf_counter() - begin
                    optimizer.zero_grad(set_to_none=True)
                    if tensor_hashes(model) != before:
                        raise RuntimeError("warmup changed minimal single-P parameters")
                    del warm
                result, gradient_norm = training_step(
                    model,
                    sequence,
                    batch,
                    optimizer,
                    penalty=penalty,
                )
                torch.cuda.synchronize()
                if not bool(torch.isfinite(result.loss)):
                    raise RuntimeError("nonfinite minimal single-P loss")
                fast = result.fast_weights.detach()
                effective = model.effective_fast_weights(fast).detach()
                row = {
                    "step": step,
                    "batch_fingerprint": fingerprint,
                    "stream_fingerprint": digest.hexdigest(),
                    "edge_count": len(episodes[0].graph_rank_pairs),
                    "loss": float(result.loss.detach()),
                    "query_loss": float(result.query_loss.detach()),
                    "terminal_penalty": float(penalty * fast.square().mean()),
                    "query_accuracy": float(
                        ((result.margins[:, 0] > 0).long() == batch.targets)
                        .float()
                        .mean()
                    ),
                    "P_frobenius": float(torch.linalg.matrix_norm(fast).mean()),
                    "effective_P_frobenius": float(
                        torch.linalg.matrix_norm(effective).mean()
                    ),
                    "clamp_fraction": float((fast.abs() == 50.0).float().mean()),
                    "gradient_norm": float(gradient_norm.detach()),
                }
                log.write(json.dumps(row, allow_nan=False) + "\n")
                if step % 100 == 0 or step == spec["training"]["updates"] - 1:
                    log.flush()
                    print(json.dumps({**identity, **row}), flush=True)
        checkpoint = directory / "model.pth"
        with checkpoint.open("xb") as handle:
            torch.save(
                {
                    "level": level,
                    "seed": seed,
                    "config": {
                        "cue_size": 15,
                        "hidden_size": level_settings(level)["hidden_size"],
                    },
                    "state_dict": model.state_dict(),
                },
                handle,
            )
        metadata = {
            **identity,
            "runtime": runtime,
            "initial_model": initial_model,
            "initial_shared": initial_shared,
            "final_model": tensor_hashes(model),
            "parameters": sum(value.numel() for value in model.parameters()),
            "stream_fingerprint": digest.hexdigest(),
            "optimizer_steps": {
                name: int(optimizer.state.get(value, {}).get("step", 0))
                for name, value in model.named_parameters()
            },
            "episode_exposures": (
                spec["training"]["updates"] * spec["training"]["batch_size"]
            ),
            "warmup_seconds": warm_seconds,
            "training_seconds": time.perf_counter() - started - (warm_seconds or 0.0),
            "checkpoint": reference(checkpoint),
        }
        write_json_exclusive(directory / "result.json", metadata)
    return validate_training_run(seed, level)


def train_all(level: str) -> dict:
    level = validate_level(level)
    source = validate_source_lock()
    validate_predecessor(level)
    completed = []
    for seed in specification()["design"]["network_seeds"]:
        train_one(seed, level, source)
        completed.append(seed)
        gc.collect()
        torch.cuda.empty_cache()
    return {"level": level, "completed_seeds": completed, "outcomes_exposed": False}


__all__ = ["train_all", "train_one"]
