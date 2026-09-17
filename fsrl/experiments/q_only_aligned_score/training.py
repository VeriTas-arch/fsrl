"""Train all twenty final-only two-scalar comparators on parent M2 streams."""

from __future__ import annotations

import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from fsrl.experiments.clean_single_p.batches import prepare_single_p
from fsrl.experiments.pl_direct_training.execution import configure_execution
from fsrl.experiments.pl_direct_training.task import make_task_generator
from fsrl.experiments.training_strategy.batches import sample_episodes
from fsrl.infra.provenance import tensor_hashes, write_json_exclusive
from fsrl.infra.run_manifest import ProspectiveRun

from .data import visible_batch
from .locks import reference, validate_source_input_lock, validate_training
from .model import QOnlyScore, physical_parameters
from .protocol import PROTOCOL_SHA256, specification, training_directory


def _optimizer(model: QOnlyScore) -> torch.optim.Adam:
    return torch.optim.Adam(
        model.parameters(),
        lr=0.01,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        amsgrad=False,
    )


def train_one(seed: int, source: dict) -> dict:
    directory = training_directory(seed)
    if directory.exists():
        return validate_training(seed)
    parent_rows = [
        json.loads(line)
        for line in Path(source["parents"]["training_logs"][str(seed)]["path"])
        .read_text()
        .splitlines()
    ]
    model = QOnlyScore(device="cuda")
    initial = tensor_hashes(model)
    runner = torch.compile(model, backend="inductor", fullgraph=True, mode="default")
    optimizer = _optimizer(model)
    task = make_task_generator({"task": source["parents"]["task"]})
    rng = np.random.default_rng(151000 + seed)
    digest = hashlib.sha256()
    identity = {
        "seed": seed,
        "protocol_sha256": PROTOCOL_SHA256,
        "source_commit": source["source_commit"],
    }
    started = time.perf_counter()
    with ProspectiveRun.start(
        directory,
        workflow_id="q_only_aligned_score_comparator_v1",
        execution_id=f"train-{seed}",
        producer=identity,
        resolved_config={
            "model": specification()["model"],
            "training": specification()["training"],
        },
    ):
        shapes: set[int] = set()
        compile_seconds = 0.0
        with (directory / "train_log.jsonl").open("x", encoding="utf-8") as log:
            for step in range(1500):
                episodes = sample_episodes(task, rng, 32, validation=False)
                parent = prepare_single_p(
                    episodes,
                    "clean",
                    observation_seed=910000000 + seed * 10000 + step,
                )
                fingerprint = parent.fingerprint()
                if fingerprint != parent_rows[step]["batch_fingerprint"]:
                    raise RuntimeError(f"parent stream replay diverged: {seed}/{step}")
                digest.update(bytes.fromhex(fingerprint))
                batch = visible_batch(parent.arrays)
                support, q, query, targets = batch.tensors("cuda")
                trials = support.shape[0]
                if trials not in shapes:
                    before = tensor_hashes(model)
                    torch.cuda.synchronize()
                    stamp = time.perf_counter()
                    warm = runner(support, q, query)[0]
                    F.softplus(-(2 * targets - 1) * warm).mean().backward()
                    torch.cuda.synchronize()
                    compile_seconds += time.perf_counter() - stamp
                    optimizer.zero_grad(set_to_none=True)
                    if tensor_hashes(model) != before:
                        raise RuntimeError("compiler warmup changed scalar parameters")
                    shapes.add(trials)
                optimizer.zero_grad(set_to_none=True)
                margins, _ = runner(support, q, query)
                loss = F.softplus(-(2 * targets - 1) * margins).mean()
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 2.0, error_if_nonfinite=True
                )
                optimizer.step()
                row = {
                    "step": step,
                    "batch_fingerprint": fingerprint,
                    "stream_fingerprint": digest.hexdigest(),
                    "loss": float(loss.detach()),
                    "gradient_norm": float(norm.detach()),
                    "parameters": physical_parameters(model),
                }
                log.write(json.dumps(row, allow_nan=False) + "\n")
                if step % 250 == 0 or step == 1499:
                    log.flush()
                    print(json.dumps({"seed": seed, **row}), flush=True)
        if digest.hexdigest() != parent_rows[-1]["stream_fingerprint"]:
            raise RuntimeError(f"parent cumulative stream differs: {seed}")
        checkpoint = directory / "model.pth"
        torch.save(model.state_dict(), checkpoint)
        result = {
            **identity,
            "parameters": 2,
            "initial_parameters": initial,
            "final_parameters": tensor_hashes(model),
            "physical_parameters": physical_parameters(model),
            "stream_fingerprint": digest.hexdigest(),
            "optimizer_steps": {
                name: int(optimizer.state[value]["step"].item())
                for name, value in model.named_parameters()
            },
            "episode_exposures": 48000,
            "compile_seconds": compile_seconds,
            "training_seconds": time.perf_counter() - started,
            "runtime": configure_execution(),
            "checkpoint": reference(checkpoint),
        }
        write_json_exclusive(directory / "result.json", result)
    return validate_training(seed)


def train_all() -> dict:
    source = validate_source_input_lock()
    completed = []
    for seed in specification()["design"]["training_streams"]:
        train_one(seed, source)
        completed.append(seed)
        gc.collect()
        torch.cuda.empty_cache()
    return {"completed_streams": completed, "outcomes_exposed": False}


__all__ = ["train_all", "train_one"]
