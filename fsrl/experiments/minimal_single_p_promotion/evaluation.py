"""Locked generic-first and Liu evaluation for all promotion networks."""

from __future__ import annotations

import gc

import numpy as np
import torch

from fsrl.experiments.finite_state.liu import primary_analysis
from fsrl.experiments.finite_state.model import sequences
from fsrl.experiments.local_memory_removal.rollouts import primary_rollouts
from fsrl.experiments.memory_structure.inputs import size_protocol
from fsrl.experiments.minimal_single_p.evaluation import _group_arrays, _summarize
from fsrl.experiments.minimal_single_p.model import MinimalSinglePSequence, make_model
from fsrl.experiments.pl_direct_training.execution import PROFILE
from fsrl.experiments.training_strategy.evaluation import (
    flatten_arrays,
    json_ready,
    write_arrays,
)
from fsrl.experiments.write_cost.locks import completed
from fsrl.infra.provenance import tensor_hashes, write_json_exclusive
from fsrl.infra.run_manifest import ProspectiveRun
from fsrl.infra.runtime import compile_module

from .adapter import evaluation_adapter
from .decisions import panel_passed
from .locks import (
    load_generic,
    load_liu,
    reference,
    require_generic_freeze,
    validate_model_lock,
)
from .protocol import (
    MODEL_LOCK,
    PROTOCOL_SHA256,
    generic_directory,
    inherited_recipe,
    liu_directory,
    specification,
)


def load_model(seed: int, lock: dict):
    row = lock["runs"][str(seed)]
    payload = torch.load(
        row["files"]["model.pth"]["path"], map_location="cuda", weights_only=True
    )
    model = make_model("M2", seed, device="cuda")
    model.load_state_dict(payload["state_dict"], strict=True)
    if tensor_hashes(model) != row["metadata"]["final_model"]:
        raise RuntimeError("promotion loaded tensors differ from joint model lock")
    model.requires_grad_(False).eval()
    return model


def evaluate_generic_one(seed: int, panel: int, source: dict, lock: dict) -> dict:
    directory = generic_directory(seed, panel)
    if directory.exists():
        return completed(directory)
    model = load_model(seed, lock)
    sequence = compile_module(MinimalSinglePSequence(model), PROFILE)
    before = tensor_hashes(model)
    groups = []
    with (
        ProspectiveRun.start(
            directory,
            workflow_id="minimal_single_p_promotion_v1",
            execution_id=f"generic-{seed}-{panel}",
            producer={
                "model_lock": reference(MODEL_LOCK),
                "seed": seed,
                "panel": panel,
            },
            resolved_config={"generic": specification()["generic_evaluation"]},
        ),
        torch.no_grad(),
    ):
        for name in sorted(source["panels"][str(panel)]["generic"]):
            groups.append(
                _group_arrays(model, sequence, load_generic(source, panel, name))
            )
        arrays = {
            key: np.concatenate([group[key] for group in groups]) for key in groups[0]
        }
        order = np.argsort(arrays["episode_indices"])
        arrays = {key: value[order] for key, value in arrays.items()}
        summary = _summarize(arrays, 980000 + seed * 10 + panel)
        result = {
            "seed": seed,
            "panel": panel,
            "protocol_sha256": PROTOCOL_SHA256,
            **summary,
        }
        result["passed"] = panel_passed(result)
        write_arrays(directory / "raw.npz", arrays)
        write_json_exclusive(directory / "result.json", result)
    if tensor_hashes(model) != before:
        raise RuntimeError("generic evaluation changed promotion model")
    return result


def evaluate_generic_all() -> dict:
    source, lock = validate_model_lock()
    units = []
    for seed in specification()["design"]["network_seeds"]:
        for panel in specification()["design"]["evaluation_panels"]:
            evaluate_generic_one(seed, panel, source, lock)
            units.append(f"{seed}/{panel}")
        gc.collect()
        torch.cuda.empty_cache()
    return {"completed_generic_units": len(units)}


def evaluate_liu_one(seed: int, panel: int, source: dict, lock: dict) -> dict:
    directory = liu_directory(seed, panel)
    if directory.exists():
        return completed(directory)
    generic = completed(generic_directory(seed, panel))
    model = load_model(seed, lock)
    adapter = evaluation_adapter(model)
    seqs = sequences(adapter, None, compiled=True)
    before = tensor_hashes(adapter)
    recipe = inherited_recipe(panel)
    cpu = load_liu(source, panel)
    task = size_protocol(recipe, 8)
    with (
        ProspectiveRun.start(
            directory,
            workflow_id="minimal_single_p_promotion_v1",
            execution_id=f"liu-{seed}-{panel}",
            producer={
                "model_lock": reference(MODEL_LOCK),
                "seed": seed,
                "panel": panel,
            },
            resolved_config={"liu": recipe["evaluation"]["liu"]},
        ),
        torch.no_grad(),
    ):
        raw = primary_rollouts(adapter, None, seqs, None, cpu, task, recipe)
        # The historical query-key shuffle addresses L. With L absent, it is a
        # registered structural no-op and is retained explicitly as such.
        raw["bundles"]["query_shuffle"] = {
            "logits": raw["bundles"]["intact"]["logits"].copy()
        }
        raw["query_shuffle_applicable"] = np.asarray(False)
        generic_contract = {
            "competence": panel_passed(generic),
            "global": {"competence": panel_passed(generic)},
        }
        liu, analysis, sampled = primary_analysis(
            raw, cpu, task, recipe, seed + panel * 1000000, generic_contract
        )
        result = {
            "seed": seed,
            "panel": panel,
            "protocol_sha256": PROTOCOL_SHA256,
            "query_shuffle_applicable": False,
            "liu": liu,
        }
        write_arrays(
            directory / "raw.npz",
            flatten_arrays({**raw, **analysis}),
        )
        write_json_exclusive(directory / "behavior.json", json_ready(sampled))
        write_json_exclusive(directory / "result.json", json_ready(result))
    if tensor_hashes(adapter) != before:
        raise RuntimeError("Liu evaluation changed promotion adapter")
    return result


def evaluate_liu_all() -> dict:
    require_generic_freeze()
    source, lock = validate_model_lock()
    units = []
    for seed in specification()["design"]["network_seeds"]:
        for panel in specification()["design"]["evaluation_panels"]:
            evaluate_liu_one(seed, panel, source, lock)
            units.append(f"{seed}/{panel}")
        gc.collect()
        torch.cuda.empty_cache()
    return {"completed_liu_units": len(units)}


__all__ = [
    "evaluate_generic_all",
    "evaluate_generic_one",
    "evaluate_liu_all",
    "evaluate_liu_one",
    "load_model",
]
