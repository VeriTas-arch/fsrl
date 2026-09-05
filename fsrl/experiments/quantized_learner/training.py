"""All mandatory matched streams, one optimizer and final-only selection."""

import gc
import hashlib
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from fsrl.experiments.minimal_learner.data import ModelBatch, generic_batch
from fsrl.experiments.minimal_learner.locks import validate_complete
from fsrl.experiments.minimal_learner.protocol import task_generator
from fsrl.experiments.minimal_learner.training import (
    compiled,
    optimize,
    physical_parameters,
    runtime,
)
from fsrl.experiments.training_strategy.batches import sample_episodes
from fsrl.experiments.training_strategy.locks import reference
from fsrl.infra.provenance import load_json, tensor_hashes, write_json_exclusive
from fsrl.infra.run_manifest import ProspectiveRun

from .encoding import encode_batch
from .evidence import RECOVERY_RESULT, validate_recovery, validate_training_record
from .protocol import (
    PROTOCOL_HASH,
    make_model,
    resolved_specification,
    run_directory,
    specification,
)


def train_steps(model, directory, spec, seed, condition) -> dict:
    settings = spec["optimization"]
    candidate = specification()
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])
    runner = compiled(model)
    generator = task_generator()
    rng = np.random.default_rng(spec["seeds"]["training_rng_offset"] + seed)
    encoding_rng = np.random.default_rng(spec["seeds"]["encoding_rng_offset"] + seed)
    digests = {name: hashlib.sha256() for name in ("base", "uniform", "encoded")}
    shapes = set()
    compile_seconds, training_seconds = 0.0, 0.0
    entries = []
    torch.cuda.reset_peak_memory_stats()
    total_start = time.perf_counter()
    with (directory / "train_log.jsonl").open("x") as handle:
        for step in range(settings["total_steps"]):
            start = time.perf_counter()
            base = generic_batch(
                sample_episodes(generator, rng, settings["batch_size"])
            )
            uniforms = encoding_rng.random(base.arrays["signed"].shape)
            batch, witness = encode_batch(
                base, condition, uniforms, candidate["encoding"]["codebook"]
            )
            hashes = {
                "base": base.fingerprint(),
                "uniform": ModelBatch({"uniforms": uniforms}).fingerprint(),
                "encoded": batch.fingerprint(),
            }
            for channel, digest in digests.items():
                digest.update(bytes.fromhex(hashes[channel]))
            entries.extend(witness["cache_entries"].tolist())
            shape = batch.arrays["signed"].shape
            warmup = 0.0
            if shape not in shapes:
                before = tensor_hashes(model)
                stamp = time.perf_counter()
                F.softplus(-runner(*batch.tensors("cuda"))[0]).mean().backward()
                torch.cuda.synchronize()
                warmup = time.perf_counter() - stamp
                if tensor_hashes(model) != before:
                    raise RuntimeError("compiler warmup changed model parameters")
                shapes.add(shape)
                compile_seconds += warmup
            loss = optimize(model, runner, batch, optimizer, settings["gradient_clip"])
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start - warmup
            training_seconds += elapsed
            parameters = physical_parameters(model)
            if not np.isfinite(loss.item()) or not 0 < parameters["eta"] < 1:
                raise RuntimeError(
                    "registered recipe reached a nonfinite/constraint boundary"
                )
            row = {
                "step": step,
                "loss": loss.item(),
                "parameters": parameters,
                "training_seconds": elapsed,
                "warmup_seconds": warmup,
            }
            for channel, digest in digests.items():
                row[f"{channel}_batch_sha256"] = hashes[channel]
                row[f"{channel}_stream_sha256"] = digest.hexdigest()
            handle.write(json.dumps(row, allow_nan=False) + "\n")
            if step % 250 == 0 or step + 1 == settings["total_steps"]:
                handle.flush()
                print(
                    seed,
                    condition,
                    step + 1,
                    parameters,
                    "loss",
                    loss.item(),
                    flush=True,
                )
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
            name: int(optimizer.state[p]["step"].item())
            for name, p in model.named_parameters()
        },
        "cost": {
            "total_seconds": time.perf_counter() - total_start,
            "training_seconds": training_seconds,
            "warmup_seconds": compile_seconds,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "trainable_scalars": 2,
            "score_entries": model.cue_size,
            "cache_entries_mean": float(np.mean(entries)),
            "cache_entries_max": max(entries),
            "code_content_bits_max": 2 * max(entries),
            "address_payload_bits_max": 2 * model.cue_size * max(entries),
            "boundary": "Code/address payload excludes container overhead, continuous score state, stable admission latents/masks and transient buffers; no total-memory claim.",
        },
    }


def train_all() -> dict:
    recovery = validate_recovery()  # No optimizer or task RNG before admission.
    execution = runtime()
    spec = resolved_specification()
    completed = []
    for seed in spec["seeds"]["mandatory"]:
        for condition in spec["seeds"]["execution_order"][str(seed)]:
            directory = run_directory(seed, condition)
            if directory.exists():
                validate_complete(directory)
                config = load_json(directory / "config.json")
                logs = [
                    json.loads(line)
                    for line in (directory / "train_log.jsonl").read_text().splitlines()
                ]
                validate_training_record(config, logs, seed, condition)
                if config["recovery"] != reference(RECOVERY_RESULT):
                    raise RuntimeError("existing run belongs to different recovery")
            else:
                model = make_model(spec, "cuda")
                initial = tensor_hashes(model)
                with ProspectiveRun.start(
                    directory,
                    workflow_id=spec["experiment_id"],
                    execution_id=f"train-{seed}-{condition}",
                    producer={
                        "module": __name__,
                        "source_commit": recovery["source_commit"],
                    },
                    resolved_config={"specification": spec, "runtime": execution},
                ):
                    config = train_steps(model, directory, spec, seed, condition)
                    torch.save(model.state_dict(), directory / "model.pth")
                    config.update(
                        {
                            "seed": seed,
                            "condition": condition,
                            "protocol_sha256": PROTOCOL_HASH,
                            "source_commit": recovery["source_commit"],
                            "recovery": reference(RECOVERY_RESULT),
                            "optimization": spec["optimization"],
                            "episodes": spec["optimization"]["total_episode_exposures"],
                            "initial_parameters": initial,
                            "runtime": execution,
                        }
                    )
                    write_json_exclusive(directory / "config.json", config)
                del model
                gc.collect()
                torch.cuda.empty_cache()
            completed.append(f"{seed}/{condition}")
    return {"complete_runs": completed, "evaluation_performed": False}
