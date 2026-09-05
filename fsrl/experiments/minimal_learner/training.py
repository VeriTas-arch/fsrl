"""Paired final-only scalar training with bounded compiled CUDA execution."""

from __future__ import annotations

import gc
import hashlib
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from fsrl.experiments.training_strategy.batches import sample_episodes
from fsrl.experiments.training_strategy.execution import configure_execution
from fsrl.infra.provenance import tensor_hashes, write_json_exclusive
from fsrl.infra.run_manifest import ProspectiveRun

from .data import generic_batch
from .locks import reference, validate_source, validate_training
from .model import make_model
from .protocol import PROTOCOL_SHA256, run_directory, specification, task_generator


def runtime() -> dict:
    snapshot = configure_execution()
    torch._dynamo.config.recompile_limit = 64
    snapshot["recompile_limit"] = 64
    return snapshot


def compiled(model):
    return torch.compile(model, backend="inductor", fullgraph=True, mode="default")


def physical_parameters(model) -> dict:
    return {
        "eta": model.eta.item(),
        "gamma_G": model.global_gain.item(),
        "gamma_L": model.local.gain.item() if model.local is not None else 0.0,
    }


def optimize(model, runner, batch, optimizer, clip):
    optimizer.zero_grad(set_to_none=True)
    margin = runner(*batch.tensors("cuda"))[0]
    signs = torch.as_tensor(2 * batch.arrays["targets"] - 1, device="cuda")
    loss = F.softplus(-signs * margin).mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip, error_if_nonfinite=True)
    optimizer.step()
    return loss


def train_steps(model, directory, spec, seed) -> dict:
    settings = spec["optimization"]
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])
    runner = compiled(model)
    generator = task_generator()
    rng = np.random.default_rng(spec["seeds"]["training_rng_offset"] + seed)
    digest = hashlib.sha256()
    shapes = set()
    warm_seconds = 0.0
    compile_seconds = 0.0
    torch.cuda.reset_peak_memory_stats()
    total_start = time.perf_counter()
    with (directory / "train_log.jsonl").open("x") as handle:
        for step in range(settings["total_steps"]):
            start = time.perf_counter()
            batch = generic_batch(
                sample_episodes(generator, rng, settings["batch_size"])
            )
            batch_hash = batch.fingerprint()
            digest.update(bytes.fromhex(batch_hash))
            shape = batch.arrays["signed"].shape
            warmup = 0.0
            if shape not in shapes:
                before = tensor_hashes(model)
                torch.cuda.synchronize()
                stamp = time.perf_counter()
                warm_margin = runner(*batch.tensors("cuda"))[0]
                F.softplus(-warm_margin).mean().backward()
                torch.cuda.synchronize()
                warmup = time.perf_counter() - stamp
                compile_seconds += warmup
                if before != tensor_hashes(model):
                    raise RuntimeError("compiler warmup changed parameters")
                shapes.add(shape)
            loss = optimize(model, runner, batch, optimizer, settings["gradient_clip"])
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start - warmup
            warm_seconds += elapsed
            row = {
                "step": step,
                "loss": loss.item(),
                "parameters": physical_parameters(model),
                "batch_sha256": batch_hash,
                "stream_sha256": digest.hexdigest(),
                "training_seconds": elapsed,
                "warmup_seconds": warmup,
            }
            if not np.isfinite(row["loss"]) or not 0 < row["parameters"]["eta"] < 1:
                raise RuntimeError(
                    "fixed recipe diverged or reached a constraint boundary"
                )
            handle.write(json.dumps(row, allow_nan=False) + "\n")
            if step % 250 == 0 or step == settings["total_steps"] - 1:
                handle.flush()
                print(
                    seed, step + 1, row["parameters"], "loss", row["loss"], flush=True
                )
    return {
        "stream_sha256": digest.hexdigest(),
        "final_parameters": tensor_hashes(model),
        "physical_parameters": physical_parameters(model),
        "optimizer_steps": {
            name: int(optimizer.state[p]["step"].item())
            for name, p in model.named_parameters()
        },
        "cost": {
            "warm_training_seconds": warm_seconds,
            "warmup_seconds": compile_seconds,
            "total_seconds": time.perf_counter() - total_start,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "trainable_parameters": sum(p.numel() for p in model.parameters()),
            "global_persistent_entries": model.cue_size,
            "local_persistent_entries": model.cue_size**2
            if model.local is not None
            else 0,
        },
    }


def train_all() -> dict:
    source = validate_source()
    execution = runtime()
    spec = specification()
    records = {}
    for seed in spec["seeds"]["mandatory"]:
        for condition in spec["seeds"]["execution_order"][str(seed)]:
            directory = run_directory(seed, condition)
            if directory.exists():
                records[f"{seed}/{condition}"] = validate_training(seed, condition)
                continue
            model = make_model(condition, spec, "cuda")
            initial = tensor_hashes(model)
            with ProspectiveRun.start(
                directory,
                workflow_id="minimal_relational_learner_v1",
                execution_id=f"train-{seed}-{condition}",
                producer={"module": __name__, "source_commit": source["source_commit"]},
                resolved_config={"specification": spec, "runtime": execution},
            ):
                result = train_steps(model, directory, spec, seed)
                torch.save(model.state_dict(), directory / "model.pth")
                result.update(
                    {
                        "seed": seed,
                        "condition": condition,
                        "protocol_sha256": PROTOCOL_SHA256,
                        "source_commit": source["source_commit"],
                        "optimization": spec["optimization"],
                        "episodes": spec["optimization"]["total_episode_exposures"],
                        "initial_parameters": initial,
                        "runtime": execution,
                        "checkpoint": reference(directory / "model.pth"),
                    }
                )
                write_json_exclusive(directory / "config.json", result)
            records[f"{seed}/{condition}"] = validate_training(seed, condition)
            del model
            gc.collect()
            torch.cuda.empty_cache()
    return {"complete_runs": sorted(records)}
