"""Locked all-cohort evaluation of the three observation conditions."""

from __future__ import annotations

import gc

import numpy as np
import torch

from fsrl.experiments.finite_state.liu import primary_analysis
from fsrl.experiments.finite_state.model import sequences
from fsrl.experiments.local_memory_removal.rollouts import primary_rollouts
from fsrl.experiments.memory_structure.inputs import size_protocol
from fsrl.experiments.minimal_single_p.model import make_model
from fsrl.experiments.minimal_single_p_promotion.adapter import evaluation_adapter
from fsrl.experiments.minimal_single_p_promotion.decisions import panel_passed
from fsrl.experiments.minimal_single_p_promotion.protocol import inherited_recipe
from fsrl.experiments.training_strategy.evaluation import (
    flatten_arrays,
    json_ready,
    write_arrays,
)
from fsrl.experiments.write_cost.locks import completed
from fsrl.infra.provenance import load_json, tensor_hashes, write_json_exclusive
from fsrl.infra.run_manifest import ProspectiveRun

from .decisions import CONDITIONS
from .locks import load_condition, reference, validate_source_input_lock
from .protocol import (
    PROTOCOL_SHA256,
    SOURCE_INPUT_LOCK,
    evaluation_directory,
    specification,
)


def _load_model(seed: int, lock: dict):
    row = lock["models"][str(seed)]
    payload = torch.load(
        row["checkpoint"]["path"], map_location="cuda", weights_only=True
    )
    model = make_model("M2", seed, device="cuda")
    model.load_state_dict(payload["state_dict"], strict=True)
    if tensor_hashes(model) != row["tensor_hashes"]:
        raise RuntimeError("loaded M2 tensors differ from source/input lock")
    model.requires_grad_(False).eval()
    return model


def _evaluate_one(
    seed: int,
    panel: int,
    condition: str,
    *,
    adapter,
    seqs,
    lock: dict,
    generic_panel: dict,
) -> dict:
    directory = evaluation_directory(seed, panel, condition)
    if directory.exists():
        return completed(directory)
    recipe = inherited_recipe(panel)
    cpu = load_condition(lock, panel, condition)
    task = size_protocol(recipe, 8)
    identity = {"seed": seed, "panel": panel, "condition": condition}
    with (
        ProspectiveRun.start(
            directory,
            workflow_id="minimal_single_p_observation_uncertainty_v1",
            execution_id=f"{seed}-{panel}-{condition}",
            producer={
                "source_input_lock": reference(SOURCE_INPUT_LOCK),
                **identity,
            },
            resolved_config={
                "observation": specification()["observation"],
                "liu": recipe["evaluation"]["liu"],
            },
        ),
        torch.no_grad(),
    ):
        raw = primary_rollouts(adapter, None, seqs, None, cpu, task, recipe)
        raw["bundles"]["query_shuffle"] = {
            "logits": raw["bundles"]["intact"]["logits"].copy()
        }
        raw["query_shuffle_applicable"] = np.asarray(False)
        competent = panel_passed(generic_panel)
        generic_contract = {
            "competence": competent,
            "global": {"competence": competent},
        }
        liu, analysis, sampled = primary_analysis(
            raw,
            cpu,
            task,
            recipe,
            seed + panel * 1000000,
            generic_contract,
        )
        result = {
            "schema_version": 1,
            "protocol_sha256": PROTOCOL_SHA256,
            **identity,
            "query_shuffle_applicable": False,
            "liu": liu,
        }
        write_arrays(directory / "raw.npz", flatten_arrays({**raw, **analysis}))
        write_json_exclusive(directory / "behavior.json", json_ready(sampled))
        write_json_exclusive(directory / "result.json", json_ready(result))
    return result


def evaluate_all() -> dict:
    lock = validate_source_input_lock()
    generic = load_json(lock["parent_generic_result"]["path"])
    completed_units = 0
    for seed in specification()["design"]["network_seeds"]:
        model = _load_model(seed, lock)
        adapter = evaluation_adapter(model)
        seqs = sequences(adapter, None, compiled=True)
        before = tensor_hashes(adapter)
        for panel in specification()["design"]["evaluation_panels"]:
            generic_panel = generic["network_results"][str(seed)]["panels"][str(panel)]
            for condition in CONDITIONS:
                _evaluate_one(
                    seed,
                    panel,
                    condition,
                    adapter=adapter,
                    seqs=seqs,
                    lock=lock,
                    generic_panel=generic_panel,
                )
                completed_units += 1
        if tensor_hashes(adapter) != before:
            raise RuntimeError("M2 observation evaluation changed model tensors")
        del model, adapter, seqs
        gc.collect()
        torch.cuda.empty_cache()
    return {"completed_units": completed_units, "scientific_summary_exposed": False}


__all__ = ["evaluate_all"]
