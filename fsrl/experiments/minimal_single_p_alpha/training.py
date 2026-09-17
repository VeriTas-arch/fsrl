"""Paired final-checkpoint training for all M2-alpha networks."""

from __future__ import annotations

import gc
import hashlib
import json
import time

import numpy as np
import torch

from fsrl.experiments.clean_single_p.batches import prepare_single_p
from fsrl.experiments.minimal_single_p.model import MinimalSinglePSequence
from fsrl.experiments.minimal_single_p.optimization import (
    forward_batch,
    make_optimizer,
    training_step,
)
from fsrl.experiments.pl_direct_training.execution import PROFILE, configure_execution
from fsrl.experiments.pl_direct_training.task import make_task_generator
from fsrl.experiments.training_strategy.batches import sample_episodes
from fsrl.infra.provenance import tensor_hashes, write_json_exclusive
from fsrl.infra.run_manifest import ProspectiveRun
from fsrl.infra.runtime import compile_module

from .locks import reference, validate_source_lock, validate_training, verify_reference
from .model import make_model
from .protocol import PROTOCOL_SHA256, specification, training_directory


def train_one(seed: int, source: dict) -> dict:
    directory = training_directory(seed)
    if directory.exists():
        return validate_training(seed)
    model = make_model(seed, device="cuda")
    sequence = compile_module(MinimalSinglePSequence(model), PROFILE)
    optimizer = make_optimizer(model, 1e-4)
    task = make_task_generator({"task": source["task"]})
    rng = np.random.default_rng(151000 + seed)
    digest = hashlib.sha256()
    parent_rows = [
        json.loads(line)
        for line in verify_reference(source["parent_runs"][str(seed)]["train_log"])
        .read_text()
        .splitlines()
    ]
    runtime = configure_execution()
    identity = {
        "seed": seed,
        "protocol_sha256": PROTOCOL_SHA256,
        "source_commit": source["source_commit"],
    }
    started = time.perf_counter()
    with ProspectiveRun.start(
        directory,
        workflow_id="minimal_single_p_alpha_v1",
        execution_id=f"train-{seed}",
        producer=identity,
        resolved_config={
            "model": specification()["model"],
            "training": specification()["training"],
            "runtime": runtime,
        },
    ):
        warm_seconds = None
        with (directory / "train_log.jsonl").open("x", encoding="utf-8") as log:
            for step in range(1500):
                episodes = sample_episodes(task, rng, 32, validation=False)
                cpu = prepare_single_p(
                    episodes, "clean", observation_seed=910000000 + seed * 10000 + step
                )
                fingerprint = cpu.fingerprint()
                if fingerprint != parent_rows[step]["batch_fingerprint"]:
                    raise RuntimeError(f"paired batch differs: {seed}/{step}")
                digest.update(bytes.fromhex(fingerprint))
                batch, _ = cpu.to("cuda")
                if warm_seconds is None:
                    before = tensor_hashes(model)
                    torch.cuda.synchronize()
                    begin = time.perf_counter()
                    warm = forward_batch(model, sequence, batch, penalty=0.0)
                    warm.loss.backward()
                    torch.cuda.synchronize()
                    warm_seconds = time.perf_counter() - begin
                    optimizer.zero_grad(set_to_none=True)
                    if tensor_hashes(model) != before:
                        raise RuntimeError("warmup changed M2-alpha parameters")
                    del warm
                result, gradient_norm = training_step(
                    model, sequence, batch, optimizer, penalty=0.0
                )
                torch.cuda.synchronize()
                if not bool(torch.isfinite(result.loss)):
                    raise RuntimeError("nonfinite M2-alpha loss")
                fast = result.fast_weights.detach()
                effective = model.effective_fast_weights(fast).detach()
                alpha = model.alpha.detach()
                row = {
                    "step": step,
                    "batch_fingerprint": fingerprint,
                    "stream_fingerprint": digest.hexdigest(),
                    "loss": float(result.loss.detach()),
                    "query_accuracy": float(
                        ((result.margins[:, 0] > 0).long() == batch.targets)
                        .float()
                        .mean()
                    ),
                    "P_frobenius": float(torch.linalg.matrix_norm(fast).mean()),
                    "effective_P_frobenius": float(
                        torch.linalg.matrix_norm(effective).mean()
                    ),
                    "alpha_mean": float(alpha.mean()),
                    "alpha_std": float(alpha.std()),
                    "alpha_frobenius": float(torch.linalg.matrix_norm(alpha)),
                    "P_clamp_fraction": float((fast.abs() == 50.0).float().mean()),
                    "gradient_norm": float(gradient_norm.detach()),
                    "eta": float(model.etaet.detach()),
                }
                log.write(json.dumps(row, allow_nan=False) + "\n")
                if step % 100 == 0 or step == 1499:
                    log.flush()
                    print(json.dumps({**identity, **row}), flush=True)
        if digest.hexdigest() != parent_rows[-1]["stream_fingerprint"]:
            raise RuntimeError(f"paired cumulative stream differs: {seed}")
        checkpoint = directory / "model.pth"
        with checkpoint.open("xb") as handle:
            torch.save(
                {"level": "M2-alpha", "seed": seed, "state_dict": model.state_dict()},
                handle,
            )
        metadata = {
            **identity,
            "runtime": runtime,
            "initial_alpha": "ones",
            "final_model": tensor_hashes(model),
            "parameters": sum(value.numel() for value in model.parameters()),
            "final_eta": float(model.etaet.detach()),
            "final_model_summary": {
                "alpha_mean": float(model.alpha.detach().mean()),
                "alpha_std": float(model.alpha.detach().std()),
                "alpha_frobenius": float(
                    torch.linalg.matrix_norm(model.alpha.detach())
                ),
            },
            "stream_fingerprint": digest.hexdigest(),
            "optimizer_steps": {
                name: int(optimizer.state.get(value, {}).get("step", 0))
                for name, value in model.named_parameters()
            },
            "warmup_seconds": warm_seconds,
            "training_seconds": time.perf_counter() - started - (warm_seconds or 0.0),
            "checkpoint": reference(checkpoint),
        }
        write_json_exclusive(directory / "result.json", metadata)
    return validate_training(seed)


def train_all() -> dict:
    source = validate_source_lock()
    completed = []
    for seed in specification()["design"]["network_seeds"]:
        train_one(seed, source)
        completed.append(seed)
        gc.collect()
        torch.cuda.empty_cache()
    return {"completed_seeds": completed, "outcomes_exposed": False}


__all__ = ["train_all", "train_one"]
