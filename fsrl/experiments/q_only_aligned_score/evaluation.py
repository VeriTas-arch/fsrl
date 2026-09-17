"""Locked generic and Liu evaluation for the aligned q-only comparator."""

from __future__ import annotations

import gc
from itertools import combinations

import numpy as np
import torch

from fsrl.analysis.hodge import build_complete_graph_geometry, gradient_energy_fraction
from fsrl.analysis.policy import exact_probability
from fsrl.analysis.statistics import summarize_subjects
from fsrl.experiments.finite_state.liu import primary_analysis
from fsrl.experiments.memory_structure.inputs import size_protocol
from fsrl.experiments.memory_structure.interventions import (
    remove_relation,
    shuffle_evidence,
)
from fsrl.experiments.minimal_single_p_pair_morphology.analysis import (
    _analysis_mask,
    _latent_probability,
    _pair_rows,
    _replay_route,
)
from fsrl.experiments.minimal_single_p_pair_morphology.methods import (
    hodge_components,
    panel_stage,
)
from fsrl.experiments.minimal_single_p_promotion.protocol import inherited_recipe
from fsrl.experiments.training_strategy.evaluation import (
    flatten_arrays,
    json_ready,
    write_arrays,
)
from fsrl.experiments.write_cost.locks import completed
from fsrl.infra.provenance import load_json, tensor_hashes, write_json_exclusive
from fsrl.infra.run_manifest import ProspectiveRun
from fsrl.tasks.protocol import ordered_pairs

from .algebra import recurrence_and_kernel
from .data import margin_signs, visible_batch
from .decisions import generic_panel_passed
from .locks import (
    load_npz,
    reference,
    require_committed,
    validate_model_lock,
    verify_reference,
)
from .model import QOnlyScore, physical_parameters
from .protocol import (
    GENERIC_RESULT,
    MODEL_LOCK,
    PROTOCOL_SHA256,
    generic_directory,
    liu_directory,
    specification,
)

BOOTSTRAP_SAMPLES = 2000


def _bundle(margins: np.ndarray) -> dict[str, np.ndarray]:
    return {"logits": np.asarray(margins, dtype=np.float64)}


def _estimate(values: np.ndarray, seed: int) -> dict:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(
        len(values), np.full(len(values), 1.0 / len(values)), BOOTSTRAP_SAMPLES
    )
    return summarize_subjects(values, counts, interval=0.95)


def _load_model(seed: int, lock: dict) -> QOnlyScore:
    row = lock["runs"][str(seed)]
    model = QOnlyScore(device="cuda")
    state = torch.load(
        verify_reference(row["model"]), map_location="cuda", weights_only=True
    )
    model.load_state_dict(state, strict=True)
    if tensor_hashes(model) != row["metadata"]["final_parameters"]:
        raise RuntimeError("aligned-score loaded tensors differ from model lock")
    return model.requires_grad_(False).eval()


def _canonical_fields(margins: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    canonical = tuple(combinations(range(8), 2))
    lookup = {pair: index for index, pair in enumerate(canonical)}
    fields = np.empty((len(margins), len(canonical)), dtype=np.float64)
    filled = np.zeros_like(fields, dtype=bool)
    for query in range(margins.shape[1]):
        for subject in range(len(margins)):
            left, right = map(int, pairs[query, subject])
            pair = (min(left, right), max(left, right))
            index = lookup[pair]
            fields[subject, index] = (
                margins[subject, query] if left < right else -margins[subject, query]
            )
            filled[subject, index] = True
    if not filled.all():
        raise RuntimeError("generic query field is incomplete")
    return fields


def _generic_group(model, arrays: dict, shuffle_seed: int) -> dict[str, np.ndarray]:
    batch = visible_batch(arrays)
    support, q, query, _ = batch.tensors("cuda")
    with torch.no_grad():
        intact, _ = model(support, q, query)
    margins = intact.cpu().numpy()
    signs = margin_signs(batch)
    probability = exact_probability(margins * signs, 1.0)
    state_off = np.full_like(probability, 0.5)

    from fsrl.evaluation.local_access import apply_blockwise_route
    from fsrl.experiments.local_fidelity.evidence_access_pilot import (
        blockwise_derangements,
    )

    trials, subjects = batch.realized_q.shape
    route = blockwise_derangements(subjects, 4, trials // 4, shuffle_seed)
    shuffled_q = apply_blockwise_route(batch.realized_q.T, route).T.copy()
    with torch.no_grad():
        shuffled, _ = model(
            support,
            torch.as_tensor(shuffled_q, device="cuda"),
            query,
        )
    shuffled_probability = exact_probability(shuffled.cpu().numpy() * signs, 1.0)
    learned = np.asarray(arrays["learned"], dtype=bool)
    fields = _canonical_fields(margins, arrays["query_pairs"])

    def grouped(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return np.divide(np.where(mask, values, 0.0).sum(1), mask.sum(1))

    return {
        "intact_margins": margins,
        "signs": signs,
        "learned": learned,
        "competence_learned": grouped(probability, learned),
        "competence_nonlearned": grouped(probability, ~learned),
        "state_dependence_learned": grouped(probability - state_off, learned),
        "state_dependence_nonlearned": grouped(probability - state_off, ~learned),
        "evidence_binding_learned": grouped(
            probability - shuffled_probability, learned
        ),
        "evidence_binding_nonlearned": grouped(
            probability - shuffled_probability, ~learned
        ),
        "coherence": gradient_energy_fraction(
            fields, build_complete_graph_geometry(size_protocol(inherited_recipe(1), 8))
        ),
        "episode_indices": np.asarray(arrays["episode_indices"]),
    }


def evaluate_generic_one(seed: int, panel: int, source: dict, lock: dict) -> dict:
    directory = generic_directory(seed, panel)
    if directory.exists():
        return completed(directory)
    model = _load_model(seed, lock)
    parent_source = load_json(
        verify_reference(source["parents"]["promotion_source_lock"])
    )
    shuffle_seed = parent_source["panels"][str(panel)]["liu_seeds"][
        "evidence_shuffle_seed"
    ]
    groups = []
    with ProspectiveRun.start(
        directory,
        workflow_id="q_only_aligned_score_comparator_v1",
        execution_id=f"generic-{seed}-{panel}",
        producer={"model_lock": reference(MODEL_LOCK), "seed": seed, "panel": panel},
        resolved_config={"generic": specification()["generic_evaluation"]},
    ):
        for name, row in sorted(
            source["parents"]["generic_inputs"][str(panel)].items()
        ):
            groups.append(_generic_group(model, load_npz(row), shuffle_seed))
        arrays = {
            key: np.concatenate([group[key] for group in groups]) for key in groups[0]
        }
        order = np.argsort(arrays["episode_indices"])
        arrays = {key: value[order] for key, value in arrays.items()}
        boot = 980000 + seed * 10 + panel
        result = {
            "schema_version": 1,
            "protocol_sha256": PROTOCOL_SHA256,
            "seed": seed,
            "panel": panel,
            "competence": {
                group: _estimate(arrays[f"competence_{group}"], boot + offset)
                for group, offset in (("learned", 1), ("nonlearned", 2))
            },
            "state_dependence": {
                group: _estimate(arrays[f"state_dependence_{group}"], boot + offset)
                for group, offset in (("learned", 3), ("nonlearned", 4))
            },
            "evidence_binding": {
                group: _estimate(arrays[f"evidence_binding_{group}"], boot + offset)
                for group, offset in (("learned", 5), ("nonlearned", 6))
            },
            "coherence": _estimate(arrays["coherence"], boot + 7),
        }
        result["passed"] = generic_panel_passed(result)
        write_arrays(directory / "raw.npz", arrays)
        write_json_exclusive(directory / "result.json", result)
    return result


def evaluate_generic_all() -> dict:
    source, lock = validate_model_lock()
    count = 0
    for seed in specification()["design"]["training_streams"]:
        for panel in specification()["design"]["generic_panels"]:
            evaluate_generic_one(seed, panel, source, lock)
            count += 1
        gc.collect()
        torch.cuda.empty_cache()
    return {"completed_generic_units": count}


def _score_rollout(model: QOnlyScore, arrays: dict) -> tuple[np.ndarray, np.ndarray]:
    batch = visible_batch(arrays)
    support, q, query, _ = batch.tensors("cuda")
    with torch.no_grad():
        margins, state = model(support, q, query)
    return margins.cpu().numpy(), state.cpu().numpy()


def _morphology(margins, removed, sampled, task, settings, arrays) -> tuple[dict, list]:
    geometry = build_complete_graph_geometry(task)
    full, accuracy, replay_error = _replay_route(
        margins, sampled["full"], task, settings["choice_seed"], settings["temperature"]
    )
    exact_p = _latent_probability(margins, geometry, settings["temperature"])
    field = 0.5 * (margins[:, 0::2] - margins[:, 1::2])
    gradient, residual = hodge_components(field, geometry)
    mask = _analysis_mask(full)
    pairs, counts = _pair_rows(
        exact_p,
        accuracy,
        full,
        mask,
        field,
        gradient,
        residual,
        removed,
        geometry,
        task,
    )
    positions = np.empty(task.n_items, dtype=int)
    for position, item in enumerate(task.true_order_high_to_low):
        positions[item] = position
    by_distance = {}
    for row, pair in zip(pairs, geometry.pairs, strict=True):
        distance = abs(int(positions[pair[0]] - positions[pair[1]]))
        row["pair"] = list(pair)
        row["symbolic_distance"] = distance
    for label, selected in (
        ("1", [row for row in pairs if row["symbolic_distance"] == 1]),
        ("2", [row for row in pairs if row["symbolic_distance"] == 2]),
        ("3", [row for row in pairs if row["symbolic_distance"] == 3]),
        ("4+", [row for row in pairs if row["symbolic_distance"] >= 4]),
    ):
        by_distance[label] = {
            "pairs": len(selected),
            "latent_bimodal_fraction": float(
                np.mean([row["latent_class"] == 3 for row in selected])
            ),
            "sampled_bimodal_fraction": float(
                np.mean([row["sampled_class"] == 3 for row in selected])
            ),
            "fraction_strong_error": float(
                np.mean([row["fraction_strong_error"] for row in selected])
            ),
            "fraction_strong_correct": float(
                np.mean([row["fraction_strong_correct"] for row in selected])
            ),
        }
    counts["stage"] = panel_stage(
        counts["direction_present_pairs"],
        counts["strong_two_sided_pairs"],
        counts["latent_bimodal_pairs"],
        counts["sampled_bimodal_pairs"],
        target=15,
    )
    counts["analysis_subjects"] = int(mask.sum())
    counts["choice_replay_max_abs_error"] = replay_error
    counts["by_distance"] = by_distance
    counts["gradient_energy_fraction"] = float(
        np.mean(gradient_energy_fraction(field[mask], geometry))
    )
    return counts, pairs


def evaluate_liu_one(
    seed: int, panel: int, condition: str, source: dict, lock: dict
) -> dict:
    directory = liu_directory(seed, panel, condition)
    if directory.exists():
        return completed(directory)
    model = _load_model(seed, lock)
    arrays = load_npz(source["parents"]["liu_inputs"][str(panel)][condition])
    recipe = inherited_recipe(panel)
    task = size_protocol(recipe, 8)
    from fsrl.experiments.training_strategy.batches import EpisodeBatch

    cpu = EpisodeBatch(arrays)
    settings = recipe["evaluation"]["liu"]
    with ProspectiveRun.start(
        directory,
        workflow_id="q_only_aligned_score_comparator_v1",
        execution_id=f"liu-{seed}-{panel}-{condition}",
        producer={
            "model_lock": reference(MODEL_LOCK),
            "seed": seed,
            "panel": panel,
            "condition": condition,
        },
        resolved_config={"liu": settings, "model": specification()["model"]},
    ):
        intact, state = _score_rollout(model, arrays)
        shuffled_cpu, evidence_route = shuffle_evidence(
            cpu, task.support_blocks, settings["evidence_shuffle_seed"]
        )
        shuffled, _ = _score_rollout(model, shuffled_cpu.arrays)
        removed = []
        for relation in task.support_pairs_higher_lower:
            changed, _ = _score_rollout(model, remove_relation(cpu, relation).arrays)
            removed.append(changed)
        removed_array = np.stack(removed)
        zero = np.zeros_like(intact)
        raw = {
            "bundles": {
                "intact": _bundle(intact),
                "local_off": _bundle(intact),
                "P_off": _bundle(zero),
                "evidence_shuffle": _bundle(shuffled),
                "query_shuffle": _bundle(intact),
            },
            "removed": removed_array,
            "removed_global": removed_array.copy(),
            "evidence_route": evidence_route,
            "query_route": np.tile(np.arange(len(ordered_pairs(8))), (len(intact), 1)),
            "cost": np.zeros(len(intact), dtype=np.float64),
            "total_write": np.linalg.norm(state, axis=1),
            "storage": {"w_l2": np.linalg.norm(state, axis=1)},
        }
        generic = completed(generic_directory(seed, panel))
        contract = {
            "competence": bool(generic["passed"]),
            "global": {"competence": bool(generic["passed"])},
        }
        liu, analysis, sampled = primary_analysis(
            raw, cpu, task, recipe, seed + panel * 1000000, contract
        )
        morphology, pairs = _morphology(
            intact, removed_array, sampled, task, settings, arrays
        )
        visible = visible_batch(arrays)
        float64_margin, kernel, reconstructed = recurrence_and_kernel(
            visible.support_cues,
            visible.realized_q,
            visible.query_cues,
            eta=physical_parameters(model)["eta"],
            gamma=physical_parameters(model)["gamma"],
            epsilon=1e-8,
        )
        float32_error = float(np.max(np.abs(intact - float64_margin)))
        kernel_error = float(np.max(np.abs(float64_margin - reconstructed)))
        if not np.allclose(intact, float64_margin, atol=1e-5, rtol=1e-5):
            raise RuntimeError("float32 rollout differs from float64 recurrence")
        if not np.allclose(float64_margin, reconstructed, atol=1e-10, rtol=1e-10):
            raise RuntimeError("float64 kernel reconstruction differs")
        result = {
            "schema_version": 1,
            "protocol_sha256": PROTOCOL_SHA256,
            "seed": seed,
            "panel": panel,
            "condition": condition,
            "generic_passed": bool(generic["passed"]),
            "parameters": physical_parameters(model),
            "liu": liu,
            "morphology": morphology,
            "analytic": {
                "float32_float64_max_abs_error": float32_error,
                "kernel_reconstruction_max_abs_error": kernel_error,
                "kernel_rms": float(np.sqrt(np.mean(kernel * kernel))),
                "conditional_gaussian_covariance_applicable": condition == "noisy",
            },
        }
        write_arrays(directory / "raw.npz", flatten_arrays({**raw, **analysis}))
        write_json_exclusive(directory / "pairs.json", json_ready(pairs))
        write_json_exclusive(directory / "behavior.json", json_ready(sampled))
        write_json_exclusive(directory / "result.json", json_ready(result))
    return result


def evaluate_liu_all() -> dict:
    require_committed(GENERIC_RESULT)
    source, lock = validate_model_lock()
    count = 0
    for seed in specification()["design"]["training_streams"]:
        for panel in specification()["design"]["liu_panels"]:
            for condition in specification()["design"]["liu_conditions"]:
                evaluate_liu_one(seed, panel, condition, source, lock)
                count += 1
        gc.collect()
        torch.cuda.empty_cache()
    return {"completed_liu_units": count}


__all__ = [
    "evaluate_generic_all",
    "evaluate_generic_one",
    "evaluate_liu_all",
    "evaluate_liu_one",
]
