"""Paired single-stage training under the three frozen encoder structures."""

from __future__ import annotations

import gc
import hashlib
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from fsrl.experiments.adaptive_plasticity.data import model_tensors
from fsrl.experiments.adaptive_plasticity.model import make_model
from fsrl.experiments.minimal_learner.data import generic_batch
from fsrl.experiments.minimal_learner.protocol import task_generator
from fsrl.experiments.minimal_learner.training import compiled, runtime
from fsrl.experiments.training_strategy.batches import EpisodeBatch, sample_episodes
from fsrl.infra.provenance import tensor_hashes, write_json_exclusive
from fsrl.infra.run_manifest import ProspectiveRun

from .encoding import draw_streams, encode_common
from .protocol import (
    CONDITIONS,
    DESIGN_HASH,
    EXECUTION_ORDER,
    SEEDS,
    resolved_specification,
    run_directory,
)


def physical_parameters(model) -> dict:
    return {"eta0": model.eta.item(), "gamma_G": model.global_gain.item()}


def optimize(model, runner, batch, optimizer, clip):
    optimizer.zero_grad(set_to_none=True)
    margins = runner(*model_tensors(batch, "cuda"))[0]
    signs = torch.as_tensor(2 * batch.arrays["targets"] - 1, device="cuda")
    loss = F.softplus(-signs * margins).mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip, error_if_nonfinite=True)
    optimizer.step()
    return loss


def train_steps(
    model, directory, spec, seed: int, condition: str, selected_rho: float
) -> dict:
    settings = spec["optimization"]
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])
    runner = compiled(model)
    generator = task_generator()
    task_rng = np.random.default_rng(spec["seeds"]["training_rng_offset"] + seed)
    encoding_rng = np.random.default_rng(spec["seeds"]["encoding_rng_offset"] + seed)
    digests = {name: hashlib.sha256() for name in ("base", "latent", "encoded")}
    shapes: set[tuple[int, ...]] = set()
    compile_seconds = 0.0
    training_seconds = 0.0
    torch.cuda.reset_peak_memory_stats()
    total_start = time.perf_counter()
    with (directory / "train_log.jsonl").open("x") as handle:
        for step in range(settings["total_steps"]):
            start = time.perf_counter()
            base = generic_batch(
                sample_episodes(generator, task_rng, settings["batch_size"])
            )
            streams = draw_streams(base, encoding_rng)
            batch, _ = encode_common(base, condition, selected_rho, streams)
            hashes = {
                "base": base.fingerprint(),
                "latent": EpisodeBatch(streams).fingerprint(),
                "encoded": batch.fingerprint(),
            }
            for name, digest in digests.items():
                digest.update(bytes.fromhex(hashes[name]))
            shape = batch.arrays["signed"].shape
            warmup = 0.0
            if shape not in shapes:
                before = tensor_hashes(model)
                stamp = time.perf_counter()
                F.softplus(-runner(*model_tensors(batch, "cuda"))[0]).mean().backward()
                torch.cuda.synchronize()
                warmup = time.perf_counter() - stamp
                if tensor_hashes(model) != before:
                    raise RuntimeError("compiler warmup changed model parameters")
                model.zero_grad(set_to_none=True)
                compile_seconds += warmup
                shapes.add(shape)
            loss = optimize(model, runner, batch, optimizer, settings["gradient_clip"])
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start - warmup
            training_seconds += elapsed
            parameters = physical_parameters(model)
            if not np.isfinite(loss.item()) or not 0 < parameters["eta0"] < 1:
                raise RuntimeError("common-encoding fit reached a constraint boundary")
            row = {
                "step": step,
                "loss": loss.item(),
                "parameters": parameters,
                "training_seconds": elapsed,
                "warmup_seconds": warmup,
            }
            for name, digest in digests.items():
                row[f"{name}_batch_sha256"] = hashes[name]
                row[f"{name}_stream_sha256"] = digest.hexdigest()
            handle.write(json.dumps(row, allow_nan=False) + "\n")
            if step % 250 == 0 or step + 1 == settings["total_steps"]:
                handle.flush()
                print(seed, condition, step + 1, parameters, loss.item(), flush=True)
    return {
        **{
            f"{name}_stream_sha256": digest.hexdigest()
            for name, digest in digests.items()
        },
        "final_parameters": tensor_hashes(model),
        "raw_parameters": {
            name: value.detach().cpu().tolist()
            for name, value in model.named_parameters()
        },
        "physical_parameters": physical_parameters(model),
        "optimizer_steps": {
            name: int(optimizer.state[value]["step"].item())
            for name, value in model.named_parameters()
        },
        "cost": {
            "total_seconds": time.perf_counter() - total_start,
            "training_seconds": training_seconds,
            "warmup_seconds": compile_seconds,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "trainable_scalars": 2,
            "score_entries": model.cue_size,
            "common_state_entries_per_subject": 1
            if condition == "episode_common"
            else 0,
        },
    }


def train_all() -> dict:
    from .evidence import validate_recovery, validate_source, validate_training

    source = validate_source()
    recovery = validate_recovery()
    selected_rho = recovery["selected_rho"]
    execution = runtime()
    spec = resolved_specification()
    complete = []
    for seed in SEEDS:
        for condition in EXECUTION_ORDER[seed]:
            directory = run_directory(seed, condition)
            if directory.exists():
                validate_training(seed, condition)
            else:
                model = make_model(
                    "adaptive_eta_resampled", spec, "cuda", scheduler="global"
                )
                initial = tensor_hashes(model)
                with ProspectiveRun.start(
                    directory,
                    workflow_id="common_encoding_state_v1",
                    execution_id=f"train-{seed}-{condition}",
                    producer={
                        "module": __name__,
                        "source_commit": source["source_commit"],
                    },
                    resolved_config={"specification": spec, "runtime": execution},
                ):
                    config = train_steps(
                        model, directory, spec, seed, condition, selected_rho
                    )
                    config.update(
                        {
                            "seed": seed,
                            "condition": condition,
                            "model_condition": "adaptive_eta_resampled",
                            "scheduler": "global",
                            "rho": selected_rho,
                            "rho_lock": recovery["lock_reference"],
                            "protocol_sha256": DESIGN_HASH,
                            "source_commit": source["source_commit"],
                            "optimization": spec["optimization"],
                            "episodes": spec["optimization"]["total_episode_exposures"],
                            "initial_parameters": initial,
                            "runtime": execution,
                        }
                    )
                    write_json_exclusive(directory / "config.json", config)
                validate_training(seed, condition)
                del model
                gc.collect()
                torch.cuda.empty_cache()
            complete.append(f"{seed}/{condition}")
    expected = {f"{seed}/{condition}" for seed in SEEDS for condition in CONDITIONS}
    if set(complete) != expected:
        raise RuntimeError("training omitted a mandatory common-encoding fit")
    return {"complete_runs": complete, "evaluation_performed": False}
