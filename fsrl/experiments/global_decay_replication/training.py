"""Paired fresh single-stage training for the unchanged replication."""

from __future__ import annotations

import gc

import torch

from fsrl.experiments.adaptive_plasticity.model import make_model
from fsrl.experiments.adaptive_plasticity.training import train_steps
from fsrl.experiments.minimal_learner.training import runtime
from fsrl.infra.provenance import tensor_hashes, write_json_exclusive
from fsrl.infra.run_manifest import ProspectiveRun

from .protocol import (
    CONDITIONS,
    DESIGN_HASH,
    EXECUTION_ORDER,
    SEEDS,
    resolved_specification,
    run_directory,
)


def train_all() -> dict:
    from .evidence import validate_source, validate_training

    source = validate_source()
    execution = runtime()
    spec = resolved_specification()
    complete = []
    for seed in SEEDS:
        for condition in EXECUTION_ORDER[seed]:
            directory = run_directory(seed, condition)
            if directory.exists():
                validate_training(seed, condition)
            else:
                scheduler = (
                    "global" if condition == "adaptive_eta_resampled" else "relation"
                )
                model = make_model(condition, spec, "cuda", scheduler=scheduler)
                initial = tensor_hashes(model)
                with ProspectiveRun.start(
                    directory,
                    workflow_id="global_decay_replication_v1",
                    execution_id=f"train-{seed}-{condition}",
                    producer={
                        "module": __name__,
                        "source_commit": source["source_commit"],
                    },
                    resolved_config={"specification": spec, "runtime": execution},
                ):
                    config = train_steps(model, directory, spec, seed, condition)
                    config.update(
                        {
                            "seed": seed,
                            "condition": condition,
                            "scheduler": scheduler,
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
        raise RuntimeError("training omitted a mandatory replication fit")
    return {"complete_runs": complete, "evaluation_performed": False}
