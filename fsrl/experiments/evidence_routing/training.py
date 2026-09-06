"""Matched joint training with condition-specific global evidence routing."""

import gc
import hashlib
import json
import time

import numpy as np
import torch

from fsrl.core.sequence import RecurrentSequence
from fsrl.experiments.memory_structure.inputs import generator, prepare_shared
from fsrl.experiments.memory_structure.model import (
    forward_batch,
    make_model,
    objective,
    optimizer_for,
    update,
)
from fsrl.experiments.training_strategy.batches import sample_episodes
from fsrl.experiments.training_strategy.execution import PROFILE, configure_execution
from fsrl.infra.provenance import tensor_hashes, write_json_exclusive
from fsrl.infra.run_manifest import ProspectiveRun
from fsrl.infra.runtime import compile_module

from .inputs import route_batch
from .locks import validate_run, validate_source_lock
from .protocol import PROTOCOL_SHA256, run_directory, specification


def train_steps(backbone, local, sequence, spec, seed, directory, condition) -> dict:
    optimizer = optimizer_for(backbone, local, spec)
    task = generator(spec)
    rng = np.random.default_rng(spec["seeds"]["training_rng_offset"] + seed)
    digest = hashlib.sha256()
    base_digest = hashlib.sha256()
    warmup = 0.0
    started = time.perf_counter()
    with (directory / "train_log.jsonl").open("x") as log:
        for step in range(spec["optimization"]["total_steps"]):
            cpu = prepare_shared(
                sample_episodes(task, rng, spec["optimization"]["batch_size"])
            )
            base_fingerprint = cpu.fingerprint()
            base_digest.update(bytes.fromhex(base_fingerprint))
            cpu = route_batch(cpu, condition)
            fingerprint = cpu.fingerprint()
            digest.update(bytes.fromhex(fingerprint))
            batch = cpu.to("cuda")
            if step == 0:
                before = tensor_hashes(backbone)
                torch.cuda.synchronize()
                start = time.perf_counter()
                loss, _ = objective(
                    forward_batch(backbone, local, sequence, batch), batch, 0.0
                )
                loss.backward()
                torch.cuda.synchronize()
                warmup = time.perf_counter() - start
                optimizer.zero_grad(set_to_none=True)
                if before != tensor_hashes(backbone):
                    raise RuntimeError("warmup changed parameters")
                del loss
                torch.cuda.reset_peak_memory_stats()
            loss, ce = update(backbone, local, sequence, batch, optimizer, spec)
            row = {
                "step": step,
                "loss": loss,
                "query_cross_entropy": ce,
                "batch_fingerprint": fingerprint,
                "base_fingerprint": base_fingerprint,
                "stream_fingerprint": digest.hexdigest(),
            }
            log.write(json.dumps(row, allow_nan=False) + "\n")
            if step % 100 == 0 or step == spec["optimization"]["total_steps"] - 1:
                log.flush()
                print(
                    json.dumps({"seed": seed, "condition": condition, **row}),
                    flush=True,
                )
    modules = [("backbone", backbone)] + ([] if local is None else [("local", local)])
    return {
        "warmup_seconds": warmup,
        "training_seconds": time.perf_counter() - started - warmup,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "stream_fingerprint": digest.hexdigest(),
        "base_stream_fingerprint": base_digest.hexdigest(),
        "optimizer_steps": {
            f"{prefix}.{name}": int(optimizer.state.get(parameter, {}).get("step", 0))
            for prefix, module in modules
            for name, parameter in module.named_parameters()
        },
    }


def train_one(spec: dict, source: dict, seed: int, condition: str):
    directory = run_directory(seed, condition)
    if directory.exists():
        return validate_run(seed, condition)
    runtime = configure_execution()
    backbone, local = make_model(spec, seed, "dual", "cuda")
    initial = tensor_hashes(backbone)
    initial_local = tensor_hashes(local)
    if json.loads(json.dumps(runtime)) != source["runtime"]:
        raise RuntimeError("training runtime differs from qualification")
    sequence = compile_module(RecurrentSequence(backbone), PROFILE)
    identity = {
        "seed": seed,
        "condition": condition,
        "protocol_sha256": PROTOCOL_SHA256,
        "source_commit": source["source_commit"],
    }
    with ProspectiveRun.start(
        directory,
        workflow_id="weak_evidence_routing_v1",
        execution_id=f"train-{seed}-{condition}",
        producer=identity,
        resolved_config={"specification": spec, "runtime": runtime},
    ):
        stats = train_steps(backbone, local, sequence, spec, seed, directory, condition)
        with (directory / "net.pth").open("xb") as handle:
            torch.save(backbone.state_dict(), handle)
        if local is not None:
            with (directory / "local.pth").open("xb") as handle:
                torch.save(local.state_dict(), handle)
        write_json_exclusive(
            directory / "config.json",
            {
                **identity,
                **stats,
                "initial_backbone": initial,
                "initial_local": initial_local,
                "final_backbone": tensor_hashes(backbone),
                "final_local": None if local is None else tensor_hashes(local),
                "local_gain": None if local is None else float(local.gain.detach()),
                "local_scalars": 0 if local is None else local.cue_size**2,
                "P_scalars": backbone.model_config.hidden_size**2,
                "backbone_parameters": sum(p.numel() for p in backbone.parameters()),
                "local_parameters": 0
                if local is None
                else sum(p.numel() for p in local.parameters()),
                "episode_exposures": spec["optimization"]["total_episode_exposures"],
                "runtime": runtime,
            },
        )
    return validate_run(seed, condition)


def train_all():
    source = validate_source_lock()
    spec = specification()
    completed = []
    for index, seed in enumerate(spec["seeds"]["mandatory"]):
        conditions = spec["seeds"]["conditions"][:: 1 if index % 2 == 0 else -1]
        for condition in conditions:
            train_one(spec, source, seed, condition)
            completed.append(f"{seed}/{condition}")
            gc.collect()
            torch.cuda.empty_cache()
    return {"completed": completed, "outcomes_evaluated": False}
