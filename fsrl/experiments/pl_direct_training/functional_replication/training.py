"""Final-step-only training for all fresh functional-replication seeds."""

from __future__ import annotations

import gc
import time
from dataclasses import asdict

import numpy as np
import torch

from fsrl.core.factorized_plastic_rnn import FactorizedPlasticRNNConfig
from fsrl.core.local_trace import PackedConjunctiveLocalTrace
from fsrl.core.model_config import RetroModelConfig
from fsrl.core.plastic_rnn import RetroModulRNN
from fsrl.infra.provenance import tensor_hashes, write_json_exclusive
from fsrl.infra.run_manifest import ProspectiveRun

from ..model import map_shadow_model
from ..task import make_task_generator
from ..training import _sequence, _train_steps
from .locks import reference, run_directory, validate_source_lock, validate_training_run
from .protocol import (
    CONDITION,
    PARENT_PROTOCOL_SHA256,
    PARENT_REPAIR_SHA256,
    PROTOCOL_SHA256,
    REPAIR_SHA256,
    candidate_specification,
    registered_seeds,
)


def train_one(seed: int, source_lock: dict) -> dict:
    specification = candidate_specification()
    directory = run_directory(seed)
    if directory.exists():
        return validate_training_run(directory)
    from ..execution import configure_execution

    runtime = configure_execution()
    torch.manual_seed(seed)
    architecture = specification["architecture"]["common"]
    shadow = RetroModulRNN(
        RetroModelConfig(
            input_size=37,
            hidden_size=architecture["hidden_size"],
            output_size=2,
            batch_size=specification["optimization"]["batch_size"],
        ),
        device="cuda",
    )
    initial_shadow = tensor_hashes(shadow)
    backbone = map_shadow_model(shadow, CONDITION)
    del shadow
    local = PackedConjunctiveLocalTrace(
        specification["task"]["cue_size"],
        initial_gain=specification["optimization"]["initial_local_gain"],
        device="cuda",
    )
    initial_backbone = tensor_hashes(backbone)
    initial_local = tensor_hashes(local)
    sequence = _sequence(backbone)
    task = make_task_generator(specification)
    rng = np.random.default_rng(100000 + seed)
    producer = {
        "module": __name__,
        "source_commit": source_lock["source_commit"],
        "protocol_sha256": PROTOCOL_SHA256,
        "repair_sha256": REPAIR_SHA256,
        "parent_protocol_sha256": PARENT_PROTOCOL_SHA256,
        "parent_repair_sha256": PARENT_REPAIR_SHA256,
    }
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with ProspectiveRun.start(
        directory,
        workflow_id="pl_functional_replication_v1",
        execution_id=f"training-{seed}-{CONDITION}",
        producer=producer,
        resolved_config={
            "seed": seed,
            "condition": CONDITION,
            "optimization": specification["optimization"],
            "runtime": runtime,
        },
    ):
        stats = _train_steps(
            specification,
            CONDITION,
            backbone,
            local,
            sequence,
            task,
            rng,
            directory,
        )
        checkpoint = directory / "model.pth"
        with checkpoint.open("xb") as handle:
            torch.save(
                {
                    "condition": CONDITION,
                    "model_config": asdict(
                        FactorizedPlasticRNNConfig(
                            cue_size=15,
                            hidden_size=architecture["hidden_size"],
                        )
                    ),
                    "backbone": backbone.state_dict(),
                    "local": local.state_dict(),
                },
                handle,
            )
        metadata = {
            "schema_version": 1,
            "protocol_sha256": PROTOCOL_SHA256,
            "repair_sha256": REPAIR_SHA256,
            "parent_protocol_sha256": PARENT_PROTOCOL_SHA256,
            "parent_repair_sha256": PARENT_REPAIR_SHA256,
            "source_commit": source_lock["source_commit"],
            "seed": seed,
            "condition": CONDITION,
            "runtime": runtime,
            "architecture": {
                "task_input_size": backbone.model_config.input_size,
                "hidden_size": backbone.model_config.hidden_size,
                "support_steps": architecture["support_trial_steps"],
                "query_steps": architecture["query_trial_steps"],
                "P_scalars": architecture["persistent_state_sizes"]["P"],
                "L_scalars": architecture["persistent_state_sizes"]["L"],
                "backbone_parameters": sum(
                    value.numel() for value in backbone.parameters()
                ),
                "time_parameters": 0,
                "compatibility_buffers": len(list(backbone.named_buffers())),
            },
            "optimization": specification["optimization"],
            "initial_shadow": initial_shadow,
            "initial_backbone": initial_backbone,
            "initial_local": initial_local,
            "final_backbone": tensor_hashes(backbone),
            "final_local": tensor_hashes(local),
            "raw_gain": float(local.raw_gain.detach()),
            "local_gain": float(local.gain.detach()),
            "episode_exposures": (
                specification["optimization"]["batch_size"]
                * specification["optimization"]["total_steps"]
            ),
            "checkpoint": reference(checkpoint),
            "cost": {
                "total_seconds": time.perf_counter() - started,
                "backbone_parameters": sum(
                    value.numel() for value in backbone.parameters()
                ),
                "local_parameters": local.raw_gain.numel(),
                "persistent_P_scalars_per_episode": architecture[
                    "persistent_state_sizes"
                ]["P"],
                "persistent_L_scalars_per_episode": architecture[
                    "persistent_state_sizes"
                ]["L"],
            },
            **stats,
        }
        write_json_exclusive(directory / "config.json", metadata)
    return validate_training_run(directory)


def train_cohort() -> dict:
    source = validate_source_lock()
    completed = {}
    for seed in registered_seeds():
        completed[str(seed)] = train_one(seed, source)
        gc.collect()
        torch.cuda.empty_cache()
    return {
        "completed": sorted(completed),
        "generic_or_liu_evaluated": False,
    }
