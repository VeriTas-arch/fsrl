"""Post-lock historical B4 and nested B1--B7 evaluation."""

from __future__ import annotations

import gc

import numpy as np
import torch
import torch.nn.functional as F

from fsrl.experiments.clean_single_p.adapter import evaluation_adapter
from fsrl.experiments.clean_single_p.batches import SinglePEpisodeBatch
from fsrl.experiments.clean_single_p.model import (
    AffineSingleP,
    AffineSinglePSequence,
    CleanSinglePConfig,
)
from fsrl.experiments.duplicate_observation.inputs import observed
from fsrl.experiments.duplicate_observation.rollouts import generic_arrays
from fsrl.experiments.finite_state.liu import primary_analysis
from fsrl.experiments.finite_state.model import sequences
from fsrl.experiments.local_memory_removal.rollouts import primary_rollouts
from fsrl.experiments.memory_structure.inputs import size_protocol
from fsrl.experiments.pl_direct_training.execution import PROFILE
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
from fsrl.infra.runtime import compile_module

from .locks import reference, validate_model_lock
from .protocol import EVALUATION_RUNS, MODEL_LOCK, historical_recipe, specification
from .trajectory import nested_forward

CELLS = {
    "A0": {"training": "clean", "observation": "clean"},
    "Ae": {"training": "clean", "observation": "noisy"},
    "C0": {"training": "noisy", "observation": "clean"},
    "Ce": {"training": "noisy", "observation": "noisy"},
}


def legacy_input_record(record: dict) -> dict:
    with np.load(verify_reference(record), allow_pickle=False) as raw:
        batch = EpisodeBatch({key: raw[key] for key in raw.files})
    return {"file": record, "fingerprint": batch.fingerprint()}


def load_model(seed: int, recipe: str, arm: str, models: dict) -> AffineSingleP:
    row = models["runs"][f"{seed}/{recipe}/{arm}"]
    payload = torch.load(
        verify_reference(row["files"]["model.pth"]),
        map_location="cuda",
        weights_only=True,
    )
    model = AffineSingleP(
        CleanSinglePConfig(**payload["config"]), retain_time=False, device="cuda"
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    if tensor_hashes(model) != row["metadata"]["final_model"]:
        raise RuntimeError("loaded anytime tensors differ from model lock")
    model.requires_grad_(False).eval()
    return model


def evaluate_historical_one(
    seed: int,
    panel: int,
    recipe_name: str,
    cell: str,
    settings: dict,
    source: dict,
    models: dict,
) -> dict:
    directory = (
        EVALUATION_RUNS / "historical" / str(seed) / str(panel) / recipe_name / cell
    )
    if directory.exists():
        return completed(directory)
    recipe = historical_recipe(panel)
    model = load_model(seed, recipe_name, settings["training"], models)
    adapter = evaluation_adapter(model)
    seqs = sequences(adapter, None, compiled=True)
    before = tensor_hashes(adapter)
    with ProspectiveRun.start(
        directory,
        workflow_id="single_p_anytime_v1",
        execution_id=f"historical-{seed}-{panel}-{recipe_name}-{cell}",
        producer={
            "model_lock": reference(MODEL_LOCK),
            "seed": seed,
            "panel": panel,
            "recipe": recipe_name,
            "cell": cell,
        },
        resolved_config=recipe["evaluation"],
    ):
        with torch.no_grad():
            panel_inputs = {
                "inputs": {
                    name: legacy_input_record(record)
                    for name, record in source["historical_panels"][str(panel)][
                        "inputs"
                    ].items()
                }
            }
            generic_raw = generic_arrays(
                adapter,
                None,
                seqs,
                panel_inputs,
                "test",
                settings["observation"],
                recipe,
                keep_hint=False,
            )
            generic = summarize_generic(generic_raw, recipe, seed + panel * 1000000)
            global_raw = {**generic_raw, "margins": generic_raw["global_margins"]}
            generic["global"] = summarize_generic(
                global_raw, recipe, seed + panel * 1000000
            )
            generic_raw["global_ce"] = global_raw["ce"]
            cpu = observed(
                load_input(
                    legacy_input_record(
                        source["historical_panels"][str(panel)]["inputs"]["liu-8"]
                    )
                ),
                settings["observation"],
                recipe,
            )
            task = size_protocol(recipe, 8)
            raw = primary_rollouts(adapter, None, seqs, None, cpu, task, recipe)
            liu, analysis, sampled = primary_analysis(
                raw, cpu, task, recipe, seed + panel * 1000000, generic
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
        raise RuntimeError("historical anytime evaluation changed parameters")
    return result


def _load_anytime_batch(record: dict) -> SinglePEpisodeBatch:
    with np.load(verify_reference(record), allow_pickle=False) as raw:
        return SinglePEpisodeBatch({key: raw[key] for key in raw.files})


def evaluate_anytime_one(
    seed: int,
    panel: int,
    recipe_name: str,
    cell: str,
    settings: dict,
    source: dict,
    models: dict,
) -> dict:
    directory = (
        EVALUATION_RUNS / "anytime" / str(seed) / str(panel) / recipe_name / cell
    )
    if directory.exists():
        return completed(directory)
    model = load_model(seed, recipe_name, settings["training"], models)
    sequence = compile_module(AffineSinglePSequence(model), PROFILE)
    before = tensor_hashes(model)
    with ProspectiveRun.start(
        directory,
        workflow_id="single_p_anytime_v1",
        execution_id=f"anytime-{seed}-{panel}-{recipe_name}-{cell}",
        producer={
            "model_lock": reference(MODEL_LOCK),
            "seed": seed,
            "panel": panel,
            "recipe": recipe_name,
            "cell": cell,
        },
        resolved_config=specification()["evaluation"],
    ):
        margins, probabilities, losses, targets, learned, edge_counts = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        p_norm, effective_p_norm, clamp_fraction = [], [], []
        with torch.no_grad():
            for edge_count in (7, 8, 9, 10):
                record = source["anytime_panels"][str(panel)]["inputs"][
                    f"E{edge_count}-{settings['observation']}"
                ]
                cpu = _load_anytime_batch(record)
                batch, times = cpu.to("cuda")
                nested = nested_forward(
                    model, sequence, batch, times, edge_count=edge_count
                )
                query_count = cpu.arrays["query_pairs"].shape[0]
                subject_count = cpu.arrays["item_codes"].shape[0]
                margin = nested.margins.reshape(7, query_count, subject_count)
                target = batch.targets.reshape(query_count, subject_count)
                sign = 2.0 * target.to(margin.dtype) - 1.0
                margins.append(margin.cpu().numpy())
                probabilities.append(
                    torch.sigmoid(sign.unsqueeze(0) * margin).cpu().numpy()
                )
                losses.append(F.softplus(-sign.unsqueeze(0) * margin).cpu().numpy())
                targets.append(target.cpu().numpy())
                learned.append(np.asarray(cpu.arrays["learned"], dtype=bool))
                edge_counts.append(np.full(subject_count, edge_count, dtype=np.int16))
                p_norm.append(nested.p_norm.cpu().numpy())
                effective_p_norm.append(nested.effective_p_norm.cpu().numpy())
                clamp_fraction.append(nested.clamp_fraction.cpu().numpy())
        arrays = {
            "margins": np.concatenate(margins, axis=2),
            "correct_probability": np.concatenate(probabilities, axis=2),
            "query_ce": np.concatenate(losses, axis=2),
            "targets": np.concatenate(targets, axis=1),
            "learned": np.concatenate(learned, axis=1),
            "edge_count": np.concatenate(edge_counts),
            "P_frobenius_by_E": np.stack(p_norm, axis=1),
            "effective_P_frobenius_by_E": np.stack(effective_p_norm, axis=1),
            "clamp_fraction_by_E": np.stack(clamp_fraction, axis=1),
        }
        write_arrays(directory / "raw.npz", arrays)
        result = {
            "episodes": int(arrays["targets"].shape[1]),
            "queries_per_episode": int(arrays["targets"].shape[0]),
            "horizons": 7,
            "edge_counts": {
                str(value): int(np.count_nonzero(arrays["edge_count"] == value))
                for value in (7, 8, 9, 10)
            },
            "diagnostics": {
                "P_frobenius": arrays["P_frobenius_by_E"].mean(axis=1).tolist(),
                "effective_P_frobenius": arrays["effective_P_frobenius_by_E"]
                .mean(axis=1)
                .tolist(),
                "clamp_fraction": arrays["clamp_fraction_by_E"].mean(axis=1).tolist(),
            },
        }
        write_json_exclusive(directory / "result.json", json_ready(result))
    if tensor_hashes(model) != before:
        raise RuntimeError("nested anytime evaluation changed model parameters")
    return result


def evaluate_all() -> dict:
    source, models = validate_model_lock()
    spec = specification()
    historical = anytime = 0
    for seed in spec["design"]["network_seeds"]:
        for panel in spec["design"]["evaluation_panels"]:
            for recipe_name in spec["design"]["training_recipes"]:
                for cell, settings in CELLS.items():
                    evaluate_historical_one(
                        seed, panel, recipe_name, cell, settings, source, models
                    )
                    historical += 1
                    evaluate_anytime_one(
                        seed, panel, recipe_name, cell, settings, source, models
                    )
                    anytime += 1
                    gc.collect()
                    torch.cuda.empty_cache()
    return {
        "completed_historical_units": historical,
        "completed_anytime_units": anytime,
    }


__all__ = [
    "CELLS",
    "evaluate_all",
    "evaluate_anytime_one",
    "evaluate_historical_one",
    "legacy_input_record",
    "load_model",
]
