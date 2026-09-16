"""Post-lock four-cell evaluation for both clean single-P conditions."""

from __future__ import annotations

import gc

import numpy as np
import torch

from fsrl.experiments.duplicate_observation.inputs import observed
from fsrl.experiments.duplicate_observation.rollouts import generic_arrays
from fsrl.experiments.finite_state.liu import primary_analysis
from fsrl.experiments.finite_state.model import sequences
from fsrl.experiments.local_memory_removal.rollouts import primary_rollouts
from fsrl.experiments.memory_structure.inputs import size_protocol
from fsrl.experiments.training_strategy.batches import EpisodeBatch
from fsrl.experiments.training_strategy.evaluation import (
    flatten_arrays,
    json_ready,
    write_arrays,
)
from fsrl.experiments.training_strategy.locks import verify_reference
from fsrl.experiments.write_cost.evaluation import summarize_generic
from fsrl.experiments.write_cost.inputs import load_input
from fsrl.experiments.write_cost.locks import completed
from fsrl.infra.provenance import tensor_hashes, write_json_exclusive
from fsrl.infra.run_manifest import ProspectiveRun

from .adapter import evaluation_adapter
from .locks import reference, validate_model_lock
from .model import AffineSingleP, CleanSinglePConfig
from .protocol import EVALUATION_RUNS, MODEL_LOCK, inherited_recipe, specification

CELLS = {
    "A0": {"training": "clean", "observation": "clean"},
    "Ae": {"training": "clean", "observation": "noisy"},
    "C0": {"training": "noisy", "observation": "clean"},
    "Ce": {"training": "noisy", "observation": "noisy"},
}


def legacy_input_record(record: dict) -> dict:
    """Wrap one verified direct input reference for the inherited loader."""
    with np.load(verify_reference(record), allow_pickle=False) as raw:
        batch = EpisodeBatch({key: raw[key] for key in raw.files})
    return {"file": record, "fingerprint": batch.fingerprint()}


def load_model(seed: int, condition: str, arm: str, models: dict):
    row = models["runs"][f"{seed}/{condition}/{arm}"]
    checkpoint = row["files"]["model.pth"]
    payload = torch.load(
        verify_reference(checkpoint), map_location="cuda", weights_only=True
    )
    model = AffineSingleP(
        CleanSinglePConfig(**payload["config"]),
        retain_time=condition == "time_retained_control",
        device="cuda",
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    if tensor_hashes(model) != row["metadata"]["final_model"]:
        raise RuntimeError("clean single-P loaded tensors differ from model lock")
    model.requires_grad_(False).eval()
    return model


def evaluate_one(seed, panel, condition, cell, settings, source, models):
    directory = EVALUATION_RUNS / str(seed) / str(panel) / condition / cell
    if directory.exists():
        return completed(directory)
    spec = inherited_recipe(panel)
    # Select the locked training arm explicitly; collection otherwise remains paired.
    clean = load_model(seed, condition, settings["training"], models)
    adapter = evaluation_adapter(clean)
    seqs = sequences(adapter, None, compiled=True)
    before = tensor_hashes(adapter)
    with ProspectiveRun.start(
        directory,
        workflow_id="clean_single_p_v1",
        execution_id=f"evaluate-{seed}-{panel}-{condition}-{cell}",
        producer={
            "model_lock": reference(MODEL_LOCK),
            "seed": seed,
            "panel": panel,
            "condition": condition,
            "cell": cell,
        },
        resolved_config=spec["evaluation"],
    ):
        with torch.no_grad():
            panel_inputs = {
                "inputs": {
                    name: legacy_input_record(record)
                    for name, record in source["panels"][str(panel)]["inputs"].items()
                }
            }
            generic_raw = generic_arrays(
                adapter,
                None,
                seqs,
                panel_inputs,
                "test",
                settings["observation"],
                spec,
                keep_hint=False,
            )
            generic = summarize_generic(generic_raw, spec, seed + panel * 1000000)
            global_raw = {**generic_raw, "margins": generic_raw["global_margins"]}
            generic["global"] = summarize_generic(
                global_raw, spec, seed + panel * 1000000
            )
            generic_raw["global_ce"] = global_raw["ce"]
            cpu = observed(
                load_input(
                    legacy_input_record(source["panels"][str(panel)]["inputs"]["liu-8"])
                ),
                settings["observation"],
                spec,
            )
            task = size_protocol(spec, 8)
            raw = primary_rollouts(adapter, None, seqs, None, cpu, task, spec)
            liu, analysis, sampled = primary_analysis(
                raw, cpu, task, spec, seed + panel * 1000000, generic
            )
        result = {"generic": generic, "liu": liu}
        write_arrays(
            directory / "raw.npz",
            flatten_arrays(
                {
                    "generic": generic_raw,
                    "liu": flatten_arrays({**raw, **analysis}),
                }
            ),
        )
        write_json_exclusive(directory / "behavior.json", json_ready(sampled))
        write_json_exclusive(directory / "result.json", json_ready(result))
    if tensor_hashes(adapter) != before:
        raise RuntimeError("clean single-P evaluation changed adapter parameters")
    return result


def evaluate_all() -> dict:
    source, models = validate_model_lock()
    spec = specification()
    completed_units = []
    for seed in spec["design"]["network_seeds"]:
        for panel in spec["design"]["evaluation_panels"]:
            for condition in spec["design"]["architecture_conditions"]:
                for cell, settings in CELLS.items():
                    evaluate_one(seed, panel, condition, cell, settings, source, models)
                    completed_units.append(f"{seed}/{panel}/{condition}/{cell}")
                    gc.collect()
                    torch.cuda.empty_cache()
    return {"completed_evaluation_units": len(completed_units)}


__all__ = [
    "CELLS",
    "evaluate_all",
    "evaluate_one",
    "legacy_input_record",
    "load_model",
]
