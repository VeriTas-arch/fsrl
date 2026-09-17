"""Locked generic-first and clean/folded/noisy vector evaluation."""

from __future__ import annotations

import gc

import numpy as np
import torch

from fsrl.experiments.finite_state.liu import primary_analysis
from fsrl.experiments.finite_state.model import sequences
from fsrl.experiments.local_memory_removal.rollouts import primary_rollouts
from fsrl.experiments.memory_structure.inputs import size_protocol
from fsrl.experiments.minimal_single_p.evaluation import _group_arrays, _summarize
from fsrl.experiments.minimal_single_p.model import MinimalSinglePSequence
from fsrl.experiments.minimal_single_p_promotion.decisions import panel_passed
from fsrl.experiments.minimal_single_p_promotion.protocol import inherited_recipe
from fsrl.experiments.pl_direct_training.execution import PROFILE
from fsrl.experiments.q_only_aligned_score.evaluation import _morphology
from fsrl.experiments.training_strategy.evaluation import (
    flatten_arrays,
    json_ready,
    write_arrays,
)
from fsrl.experiments.write_cost.locks import completed
from fsrl.infra.provenance import load_json, tensor_hashes, write_json_exclusive
from fsrl.infra.run_manifest import ProspectiveRun
from fsrl.infra.runtime import compile_module

from .adapter import evaluation_adapter
from .diagnostics import geometry_records, summarize_geometry
from .locks import (
    load_generic,
    load_liu,
    reference,
    require_generic_freeze,
    validate_model_lock,
    verify_reference,
)
from .model import make_model
from .protocol import (
    MODEL_LOCK,
    PROTOCOL_SHA256,
    generic_directory,
    liu_directory,
    specification,
)


def load_model(seed: int, lock: dict):
    row = lock["runs"][str(seed)]
    payload = torch.load(
        verify_reference(row["files"]["model.pth"]),
        map_location="cuda",
        weights_only=True,
    )
    model = make_model(seed, device="cuda")
    model.load_state_dict(payload["state_dict"], strict=True)
    if tensor_hashes(model) != row["metadata"]["final_model"]:
        raise RuntimeError("loaded vector-modulation tensors differ")
    return model.requires_grad_(False).eval()


def _merge_records(groups: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {key: np.concatenate([group[key] for group in groups]) for key in groups[0]}


def evaluate_generic_one(seed: int, panel: int, source: dict, lock: dict) -> dict:
    directory = generic_directory(seed, panel)
    if directory.exists():
        return completed(directory)
    model = load_model(seed, lock)
    sequence = compile_module(MinimalSinglePSequence(model), PROFILE)
    before = tensor_hashes(model)
    groups = []
    writes = []
    with (
        ProspectiveRun.start(
            directory,
            workflow_id="minimal_single_p_vector_modulation_v1",
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
        for name in sorted(source["generic_inputs"][str(panel)]):
            cpu = load_generic(source, panel, name)
            groups.append(_group_arrays(model, sequence, cpu))
            batch, _ = cpu.to("cuda")
            writes.append(geometry_records(model, batch.support_inputs))
        arrays = {
            key: np.concatenate([group[key] for group in groups]) for key in groups[0]
        }
        order = np.argsort(arrays["episode_indices"])
        arrays = {key: value[order] for key, value in arrays.items()}
        write_records = _merge_records(writes)
        result = {
            "schema_version": 1,
            "seed": seed,
            "panel": panel,
            "protocol_sha256": PROTOCOL_SHA256,
            **_summarize(arrays, 980000 + seed * 10 + panel),
            "write_geometry": summarize_geometry(write_records),
        }
        parent = load_json(
            verify_reference(source["parents"]["promotion_generic_result"])
        )["network_results"][str(seed)]["panels"][str(panel)]
        result["paired_M2_competence_delta"] = {
            group: result["competence"][group]["mean"]
            - parent["competence"][group]["mean"]
            for group in ("learned", "nonlearned")
        }
        result["paired_preserved"] = all(
            value >= -0.02 for value in result["paired_M2_competence_delta"].values()
        )
        result["passed"] = panel_passed(result)
        write_arrays(directory / "raw.npz", arrays)
        write_arrays(directory / "writes.npz", write_records)
        write_json_exclusive(directory / "result.json", result)
    if tensor_hashes(model) != before:
        raise RuntimeError("generic evaluation changed vector model")
    return result


def evaluate_generic_all() -> dict:
    source, lock = validate_model_lock()
    count = 0
    for seed in specification()["design"]["network_seeds"]:
        for panel in specification()["design"]["generic_panels"]:
            evaluate_generic_one(seed, panel, source, lock)
            count += 1
        gc.collect()
        torch.cuda.empty_cache()
    return {"completed_generic_units": count}


def _narrow_support(cpu) -> torch.Tensor:
    support = cpu.to("cuda").support_inputs
    return torch.cat((support[..., :31], support[..., 37:38]), dim=-1)


def evaluate_liu_one(
    seed: int, panel: int, condition: str, source: dict, lock: dict
) -> dict:
    directory = liu_directory(seed, panel, condition)
    if directory.exists():
        return completed(directory)
    model = load_model(seed, lock)
    adapter = evaluation_adapter(model)
    seqs = sequences(adapter, None, compiled=True)
    before = tensor_hashes(adapter)
    recipe = inherited_recipe(panel)
    cpu = load_liu(source, panel, condition)
    task = size_protocol(recipe, 8)
    generic = completed(generic_directory(seed, panel))
    with (
        ProspectiveRun.start(
            directory,
            workflow_id="minimal_single_p_vector_modulation_v1",
            execution_id=f"liu-{seed}-{panel}-{condition}",
            producer={
                "model_lock": reference(MODEL_LOCK),
                "seed": seed,
                "panel": panel,
                "condition": condition,
            },
            resolved_config={
                "liu": recipe["evaluation"]["liu"],
                "condition": condition,
            },
        ),
        torch.no_grad(),
    ):
        raw = primary_rollouts(adapter, None, seqs, None, cpu, task, recipe)
        raw["bundles"]["query_shuffle"] = {
            "logits": raw["bundles"]["intact"]["logits"].copy()
        }
        raw["query_shuffle_applicable"] = np.asarray(False)
        competent = panel_passed(generic)
        contract = {"competence": competent, "global": {"competence": competent}}
        liu, analysis, sampled = primary_analysis(
            raw, cpu, task, recipe, seed + panel * 1000000, contract
        )
        morphology, pairs = _morphology(
            raw["bundles"]["intact"]["logits"],
            raw["removed"],
            sampled,
            task,
            recipe["evaluation"]["liu"],
            cpu.arrays,
        )
        write_records = geometry_records(model, _narrow_support(cpu))
        parent = load_json(
            verify_reference(
                source["parent_liu_results"][str(seed)][str(panel)][condition]
            )
        )
        exact = liu["summaries"]["intact"]["exact_decision"]
        parent_exact = parent["liu"]["summaries"]["intact"]["exact_decision"]
        exact_delta = {
            group: exact[group]["mean"] - parent_exact[group]["mean"]
            for group in ("learned", "nonlearned", "overall")
        }
        result = {
            "schema_version": 1,
            "protocol_sha256": PROTOCOL_SHA256,
            "seed": seed,
            "panel": panel,
            "condition": condition,
            "generic_passed": competent,
            "liu": liu,
            "morphology": morphology,
            "paired_M2_exact_delta": exact_delta,
            "paired_exact_preserved": all(
                value >= -0.02 for value in exact_delta.values()
            ),
            "write_geometry": summarize_geometry(write_records),
        }
        write_arrays(directory / "raw.npz", flatten_arrays({**raw, **analysis}))
        write_arrays(directory / "writes.npz", write_records)
        write_json_exclusive(directory / "pairs.json", json_ready(pairs))
        write_json_exclusive(directory / "behavior.json", json_ready(sampled))
        write_json_exclusive(directory / "result.json", json_ready(result))
    if tensor_hashes(adapter) != before:
        raise RuntimeError("Liu evaluation changed vector adapter")
    return result


def evaluate_liu_all() -> dict:
    require_generic_freeze()
    source, lock = validate_model_lock()
    count = 0
    for seed in specification()["design"]["network_seeds"]:
        for panel in specification()["design"]["liu_panels"]:
            for condition in specification()["design"]["liu_conditions"]:
                evaluate_liu_one(seed, panel, condition, source, lock)
                count += 1
        gc.collect()
        torch.cuda.empty_cache()
    return {"completed_liu_units": count}


__all__ = ["evaluate_generic_all", "evaluate_liu_all", "load_model"]
