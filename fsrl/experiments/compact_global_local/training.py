"""Final-step-only staged training for the compact P/L candidate."""

from __future__ import annotations

import gc
import hashlib
import json
import time
from dataclasses import asdict

import numpy as np
import torch

from fsrl.experiments.training_strategy.execution import PROFILE, configure_execution
from fsrl.infra.provenance import load_json, tensor_hashes, write_json_exclusive
from fsrl.infra.run_manifest import ProspectiveRun
from fsrl.infra.runtime import compile_module
from fsrl.paths import STUDIES_ROOT

from .batches import prepare_batch, sample_episodes
from .locks import reference, run_directory, validate_source_lock, validate_training_run
from .model import (
    CompactModelConfig,
    CompactPlasticRNN,
    CompactRecurrentSequence,
    PackedLocalTrace,
)
from .optimization import (
    make_optimizer,
    training_step,
)
from .protocol import (
    PROTOCOL_SHA256,
    REPAIR_SHA256,
    load_specification,
    phase_for_step,
    registered_seeds,
)
from .task import make_task_generator


def _train_steps(specification, backbone, local, sequence, task, rng, directory):
    optimization = specification["optimization"]
    optimizer = make_optimizer(backbone, local, optimization)
    digest = hashlib.sha256()
    phase_stats = {}
    previous_phase = None
    stage_boundary = None
    with (directory / "train_log.jsonl").open("x", encoding="utf-8") as handle:
        for step in range(optimization["total_steps"]):
            phase = phase_for_step(specification, step)
            if phase != previous_phase:
                if phase == "local":
                    stage_boundary = tensor_hashes(backbone)
                phase_stats[phase] = {
                    "steps": 0,
                    "seconds": 0.0,
                    "peak_allocated_bytes": 0,
                    "peak_reserved_bytes": 0,
                }
                torch.cuda.reset_peak_memory_stats()
                previous_phase = phase
            start = time.perf_counter()
            episodes = sample_episodes(
                task, rng, optimization["batch_size"], validation=False
            )
            cpu_batch = prepare_batch(episodes)
            fingerprint = cpu_batch.fingerprint()
            digest.update(bytes.fromhex(fingerprint))
            batch = cpu_batch.to("cuda")
            result = training_step(
                backbone,
                local,
                sequence,
                batch,
                optimizer,
                phase=phase,
                optimization=optimization,
            )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            stats = phase_stats[phase]
            stats["steps"] += 1
            stats["seconds"] += elapsed
            stats["peak_allocated_bytes"] = max(
                stats["peak_allocated_bytes"], torch.cuda.max_memory_allocated()
            )
            stats["peak_reserved_bytes"] = max(
                stats["peak_reserved_bytes"], torch.cuda.max_memory_reserved()
            )
            accuracy = (
                ((result.margins[:, 0] > 0).long() == batch.targets).float().mean()
            )
            row = {
                "step": step,
                "phase": phase,
                "batch_fingerprint": fingerprint,
                "stream_fingerprint": digest.hexdigest(),
                "loss": float(result.loss.detach()),
                "query_loss": float(result.query_loss.detach()),
                "query_accuracy": float(accuracy.detach()),
                "local_gain": float(local.gain.detach()),
                "mean_abs_fast_weight": float(
                    result.fast_weights.detach().abs().mean()
                ),
                "seconds": elapsed,
            }
            handle.write(json.dumps(row, allow_nan=False) + "\n")
            if step % 50 == 0 or step == optimization["total_steps"] - 1:
                handle.flush()
                print(json.dumps(row), flush=True)
            del result, batch
    if stage_boundary is None:
        raise RuntimeError("the registered local phase did not begin")
    optimizer_steps = {
        f"{prefix}.{name}": int(optimizer.state.get(parameter, {}).get("step", 0))
        for prefix, module in (("backbone", backbone), ("local", local))
        for name, parameter in module.named_parameters()
    }
    return {
        "phase_stats": phase_stats,
        "stream_fingerprint": digest.hexdigest(),
        "stage_boundary_backbone": stage_boundary,
        "optimizer_parameter_steps": optimizer_steps,
    }


def train_one(specification: dict, seed: int, cohort: str, source_lock: dict) -> dict:
    directory = run_directory(seed, cohort)
    if directory.exists():
        return validate_training_run(directory, specification)
    runtime = configure_execution()
    torch.manual_seed(seed)
    model_config = CompactModelConfig(
        cue_size=specification["task"]["cue_size"],
        hidden_size=specification["architecture"]["hidden_size"],
    )
    backbone = CompactPlasticRNN(model_config, device="cuda")
    local = PackedLocalTrace(
        model_config.cue_size,
        initial_gain=specification["optimization"]["initial_local_gain"],
        device="cuda",
    )
    initial_backbone = tensor_hashes(backbone)
    initial_local = tensor_hashes(local)
    sequence = compile_module(CompactRecurrentSequence(backbone), PROFILE)
    task = make_task_generator(specification)
    rng = np.random.default_rng(100000 + seed)
    producer = {
        "module": __name__,
        "source_commit": source_lock["source_commit"],
        "protocol_sha256": PROTOCOL_SHA256,
        "repair_sha256": REPAIR_SHA256,
    }
    start = time.perf_counter()
    with ProspectiveRun.start(
        directory,
        workflow_id="compact_global_local_model_v1",
        execution_id=f"{cohort}-{seed}",
        producer=producer,
        resolved_config={
            "seed": seed,
            "cohort": cohort,
            "model": asdict(model_config),
            "optimization": specification["optimization"],
            "runtime": runtime,
        },
    ):
        stats = _train_steps(
            specification, backbone, local, sequence, task, rng, directory
        )
        checkpoint = directory / "model.pth"
        with checkpoint.open("xb") as handle:
            torch.save(
                {
                    "model_config": asdict(model_config),
                    "backbone": backbone.state_dict(),
                    "local": local.state_dict(),
                },
                handle,
            )
        metadata = {
            "schema_version": 1,
            "protocol_sha256": PROTOCOL_SHA256,
            "repair_sha256": REPAIR_SHA256,
            "source_commit": source_lock["source_commit"],
            "seed": seed,
            "cohort": cohort,
            "model": asdict(model_config),
            "architecture": {
                "input_size": model_config.input_size,
                "support_steps": specification["architecture"]["support_trial_steps"],
                "query_steps": specification["architecture"]["query_steps"],
                "P_scalars": model_config.hidden_size**2,
                "L_scalars": local.state_size,
                "parameters": sum(p.numel() for p in backbone.parameters())
                + sum(p.numel() for p in local.parameters()),
            },
            "runtime": runtime,
            "initial_backbone": initial_backbone,
            "initial_local": initial_local,
            "final_backbone": tensor_hashes(backbone),
            "final_local": tensor_hashes(local),
            "local_gain": float(local.gain.detach()),
            "episode_exposures": specification["optimization"][
                "total_episode_exposures"
            ],
            "checkpoint": reference(checkpoint),
            "total_seconds": time.perf_counter() - start,
            **stats,
        }
        if metadata["stage_boundary_backbone"] != metadata["final_backbone"]:
            raise RuntimeError("the backbone moved during local adaptation")
        write_json_exclusive(directory / "config.json", metadata)
    return validate_training_run(directory, specification)


def train_cohort(cohort: str) -> dict:
    specification = load_specification()
    if cohort == "confirmation":
        development_result = (
            STUDIES_ROOT
            / "compact_global_local_model"
            / "records"
            / "results"
            / "compact_global_local_model_v1.development.json"
        )
        if not development_result.is_file():
            raise RuntimeError(
                "confirmation requires the registered development report"
            )
        if load_json(development_result)["outcome"] != "development_admitted":
            raise RuntimeError("development admission did not authorize confirmation")
    source = validate_source_lock()
    completed = []
    for seed in registered_seeds(specification, cohort):
        train_one(specification, seed, cohort, source)
        completed.append(seed)
        gc.collect()
        torch.cuda.empty_cache()
    return {"cohort": cohort, "completed": completed, "liu_evaluated": False}
