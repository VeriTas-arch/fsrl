"""Joint optimization with one prospectively selected write-cost coefficient."""

import gc
import hashlib
import json
import time

import numpy as np
import torch

from fsrl.experiments.memory_structure.inputs import generator, prepare_shared
from fsrl.experiments.memory_structure.model import make_model, optimizer_for
from fsrl.experiments.training_strategy.batches import sample_episodes
from fsrl.infra.provenance import tensor_hashes, write_json_exclusive
from fsrl.infra.run_manifest import ProspectiveRun

from .execution import configure_execution
from .locks import (
    SCALE,
    SELECTION,
    completed,
    training_dir,
    validate_phase,
    validate_source,
)
from .model import sequences, update
from .protocol import PROTOCOL_SHA256, specification


def train_steps(backbone, local, support, query, spec, seed, coefficient, directory):
    optimizer = optimizer_for(backbone, local, spec)
    task = generator(spec)
    rng = np.random.default_rng(spec["seeds"]["training_rng_offset"] + seed)
    digest = hashlib.sha256()
    started = time.perf_counter()
    with (directory / "train_log.jsonl").open("x") as log:
        for step in range(spec["optimization"]["total_steps"]):
            cpu = prepare_shared(
                sample_episodes(task, rng, spec["optimization"]["batch_size"])
            )
            fingerprint = cpu.fingerprint()
            digest.update(bytes.fromhex(fingerprint))
            row: dict = update(
                backbone,
                local,
                support,
                query,
                cpu.to("cuda"),
                optimizer,
                spec,
                coefficient,
            )
            row.update(
                step=step,
                batch_fingerprint=fingerprint,
                stream_fingerprint=digest.hexdigest(),
            )
            log.write(json.dumps(row, allow_nan=False) + "\n")
            if step % 100 == 0 or step == spec["optimization"]["total_steps"] - 1:
                log.flush()
                print(
                    json.dumps({"seed": seed, "lambda": coefficient, **row}), flush=True
                )
    return {
        "training_seconds": time.perf_counter() - started,
        "stream_fingerprint": digest.hexdigest(),
        "optimizer_steps": {
            f"{prefix}.{name}": int(optimizer.state.get(p, {}).get("step", 0))
            for prefix, module in [("backbone", backbone), ("local", local)]
            for name, p in module.named_parameters()
        },
    }


def train_one(seed, arm, coefficient, source):
    directory = training_dir(seed, arm)
    if directory.exists():
        return completed(directory)
    spec = specification()
    runtime = configure_execution()
    if json.loads(json.dumps(runtime)) != source["runtime"]:
        raise RuntimeError("runtime differs from qualification")
    backbone, local = make_model(spec, seed, "dual", "cuda")
    assert local is not None
    initial = {
        "initial_backbone": tensor_hashes(backbone),
        "initial_local": tensor_hashes(local),
    }
    support, query = sequences(backbone, compiled=True)
    identity = {
        "seed": seed,
        "arm": arm,
        "coefficient": coefficient,
        "source_commit": source["source_commit"],
        "protocol_sha256": PROTOCOL_SHA256,
    }
    with ProspectiveRun.start(
        directory,
        workflow_id="effective_write_cost_v1",
        execution_id=f"train-{seed}-{arm}",
        producer=identity,
        resolved_config={"specification": spec, "runtime": runtime},
    ):
        stats = train_steps(
            backbone, local, support, query, spec, seed, coefficient, directory
        )
        for name, module in [("net", backbone), ("local", local)]:
            with (directory / f"{name}.pth").open("xb") as handle:
                torch.save(module.state_dict(), handle)
        result = {
            **identity,
            **initial,
            **stats,
            "final_backbone": tensor_hashes(backbone),
            "final_local": tensor_hashes(local),
            "local_gain": float(local.gain.detach()),
            "runtime": runtime,
        }
        write_json_exclusive(directory / "result.json", result)
    return result


def train(stage):
    source = validate_source()
    spec = specification()
    if stage == "develop-baseline":
        jobs = [(spec["seeds"]["development"], "shared", 0.0)]
    elif stage == "develop-cost":
        scale = validate_phase(SCALE)
        jobs = [
            (spec["seeds"]["development"], f"kappa-{k}", value)
            for k, value in scale["coefficients"].items()
        ]
    else:
        selection = validate_phase(SELECTION)
        if selection["selected"] is None:
            return {"status": "not_triggered"}
        value = selection["selected"]["coefficient"]
        jobs = [
            (seed, arm, 0.0 if arm == "shared" else value)
            for i, seed in enumerate(spec["seeds"]["mandatory"])
            for arm in spec["seeds"]["conditions"][:: 1 if i % 2 == 0 else -1]
        ]
    for seed, arm, value in jobs:
        train_one(seed, arm, value, source)
        gc.collect()
        torch.cuda.empty_cache()
    return {"completed": [f"{seed}/{arm}" for seed, arm, _ in jobs]}
