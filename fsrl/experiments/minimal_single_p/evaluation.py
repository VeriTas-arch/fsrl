"""Human-blind generic evaluation for one locked ladder level."""

from __future__ import annotations

import gc
from itertools import combinations

import numpy as np
import torch

from fsrl.analysis.hodge import CompleteGraphGeometry, gradient_energy_fraction
from fsrl.analysis.policy import exact_probability
from fsrl.analysis.statistics import summarize_subjects
from fsrl.experiments.pl_direct_training.execution import PROFILE
from fsrl.experiments.training_strategy.evaluation import write_arrays
from fsrl.infra.provenance import tensor_hashes, write_json_exclusive
from fsrl.infra.run_manifest import ProspectiveRun
from fsrl.infra.runtime import compile_module

from .decisions import panel_passed
from .locks import load_panel_input, reference, validate_model_lock
from .model import MinimalSinglePSequence, make_model
from .optimization import forward_batch, query_from_weights
from .protocol import (
    PROTOCOL_SHA256,
    evaluation_directory,
    model_lock_path,
    specification,
    validate_level,
)

BOOTSTRAP_SAMPLES = 2000


def _estimate(values: np.ndarray, seed: int) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("minimal single-P endpoint requires finite episode values")
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(
        len(values), np.full(len(values), 1.0 / len(values)), BOOTSTRAP_SAMPLES
    )
    return summarize_subjects(values, counts, interval=0.95)


def _canonical_fields(margins: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    subjects, queries = margins.shape
    canonical = tuple(combinations(range(8), 2))
    lookup = {pair: index for index, pair in enumerate(canonical)}
    fields = np.empty((subjects, len(canonical)), dtype=np.float64)
    filled = np.zeros_like(fields, dtype=bool)
    for query in range(queries):
        for subject in range(subjects):
            left, right = map(int, pairs[query, subject])
            pair = (min(left, right), max(left, right))
            index = lookup[pair]
            fields[subject, index] = (
                margins[subject, query] if left < right else -margins[subject, query]
            )
            filled[subject, index] = True
    if not bool(filled.all()):
        raise RuntimeError("generic queries do not cover the complete graph")
    return fields


def _geometry() -> CompleteGraphGeometry:
    pairs = tuple(combinations(range(8), 2))
    incidence = np.zeros((len(pairs), 8), dtype=np.float64)
    for index, (first, second) in enumerate(pairs):
        incidence[index, first] = 1.0
        incidence[index, second] = -1.0
    score_operator = np.linalg.pinv(incidence)
    return CompleteGraphGeometry(
        pairs=pairs,
        incidence=incidence,
        projection=incidence @ score_operator,
        score_operator=score_operator,
        true_sign=np.zeros(len(pairs), dtype=np.float64),
        true_potential=np.zeros(8, dtype=np.float64),
    )


GENERIC_GEOMETRY = _geometry()


def _group_arrays(model, sequence, cpu) -> dict[str, np.ndarray]:
    batch, _ = cpu.to("cuda")
    subjects = batch.support_inputs.shape[2]
    queries = batch.targets.numel() // subjects
    intact = forward_batch(model, sequence, batch, penalty=0.0)
    intact_margins = intact.margins[:, 0].reshape(queries, subjects).T
    zero = model.initial_fast_weights(queries * subjects)
    p_off_margins = (
        query_from_weights(model, sequence, batch.query_inputs, zero)[:, 0]
        .reshape(queries, subjects)
        .T
    )
    signs = (2 * batch.targets - 1).reshape(queries, subjects).T.cpu().numpy()
    learned = np.asarray(cpu.arrays["learned"], dtype=bool)
    if learned.shape != intact_margins.shape:
        raise RuntimeError("generic learned mask has the wrong shape")
    intact_probability = exact_probability(
        intact_margins.detach().cpu().numpy() * signs, 1.0
    )
    p_off_probability = exact_probability(
        p_off_margins.detach().cpu().numpy() * signs, 1.0
    )
    fields = _canonical_fields(
        intact_margins.detach().cpu().numpy(), cpu.arrays["query_pairs"]
    )
    return {
        "intact_margins": intact_margins.detach().cpu().numpy(),
        "P_off_margins": p_off_margins.detach().cpu().numpy(),
        "signs": signs,
        "learned": learned,
        "competence_learned": np.divide(
            np.where(learned, intact_probability, 0.0).sum(1), learned.sum(1)
        ),
        "competence_nonlearned": np.divide(
            np.where(~learned, intact_probability, 0.0).sum(1), (~learned).sum(1)
        ),
        "P_dependence_learned": np.divide(
            np.where(learned, intact_probability - p_off_probability, 0.0).sum(1),
            learned.sum(1),
        ),
        "P_dependence_nonlearned": np.divide(
            np.where(~learned, intact_probability - p_off_probability, 0.0).sum(1),
            (~learned).sum(1),
        ),
        "coherence": gradient_energy_fraction(fields, GENERIC_GEOMETRY),
        "episode_indices": np.asarray(cpu.arrays["episode_indices"]),
    }


def _summarize(arrays: dict[str, np.ndarray], seed: int) -> dict:
    return {
        "competence": {
            group: _estimate(arrays[f"competence_{group}"], seed + offset)
            for group, offset in (("learned", 1), ("nonlearned", 2))
        },
        "P_dependence": {
            group: _estimate(arrays[f"P_dependence_{group}"], seed + offset)
            for group, offset in (("learned", 3), ("nonlearned", 4))
        },
        "coherence": _estimate(arrays["coherence"], seed + 5),
    }


def load_model(seed: int, level: str, lock: dict):
    row = lock["runs"][str(seed)]
    payload = torch.load(
        row["files"]["model.pth"]["path"], map_location="cuda", weights_only=True
    )
    model = make_model(level, seed, device="cuda")
    model.load_state_dict(payload["state_dict"], strict=True)
    if tensor_hashes(model) != row["metadata"]["final_model"]:
        raise RuntimeError("minimal single-P loaded tensors differ from model lock")
    model.requires_grad_(False).eval()
    return model


def evaluate_one(seed: int, panel: int, level: str, source: dict, lock: dict) -> dict:
    directory = evaluation_directory(seed, panel, level)
    if directory.exists():
        from fsrl.experiments.write_cost.locks import completed

        return completed(directory)
    model = load_model(seed, level, lock)
    sequence = compile_module(MinimalSinglePSequence(model), PROFILE)
    before = tensor_hashes(model)
    groups = []
    with (
        ProspectiveRun.start(
            directory,
            workflow_id="minimal_single_p_v1",
            execution_id=f"evaluate-{level}-{seed}-{panel}",
            producer={
                "model_lock": reference(model_lock_path(level)),
                "seed": seed,
                "panel": panel,
                "level": level,
            },
            resolved_config={"human_blind_generic": specification()["evaluation"]},
        ),
        torch.no_grad(),
    ):
        for name in sorted(source["panels"][str(panel)]["inputs"]):
            groups.append(
                _group_arrays(model, sequence, load_panel_input(source, panel, name))
            )
        arrays = {
            key: np.concatenate([group[key] for group in groups]) for key in groups[0]
        }
        order = np.argsort(arrays["episode_indices"])
        arrays = {key: value[order] for key, value in arrays.items()}
        summary = _summarize(arrays, 930000 + seed * 10 + panel)
        result = {
            "seed": seed,
            "panel": panel,
            "level": level,
            "protocol_sha256": PROTOCOL_SHA256,
            **summary,
        }
        result["passed"] = panel_passed(result)
        write_arrays(directory / "raw.npz", arrays)
        write_json_exclusive(directory / "result.json", result)
    if tensor_hashes(model) != before:
        raise RuntimeError("minimal single-P evaluation changed model parameters")
    return result


def evaluate_all(level: str) -> dict:
    level = validate_level(level)
    source, lock = validate_model_lock(level)
    units = []
    for seed in specification()["design"]["network_seeds"]:
        for panel in specification()["design"]["evaluation_panels"]:
            evaluate_one(seed, panel, level, source, lock)
            units.append(f"{seed}/{panel}")
            gc.collect()
            torch.cuda.empty_cache()
    return {"level": level, "completed_evaluation_units": units}


__all__ = [
    "BOOTSTRAP_SAMPLES",
    "evaluate_all",
    "evaluate_one",
    "load_model",
]
