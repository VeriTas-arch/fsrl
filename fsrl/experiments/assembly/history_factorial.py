"""Registered history baseline-by-factor closure for frozen pilot networks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from fsrl.analysis.hodge import build_complete_graph_geometry, hodge_potentials
from fsrl.analysis.statistics import (
    bootstrap_counts,
    json_values,
    masked_column_mean,
    summarize_subjects,
)
from fsrl.core.config import DEVICE
from fsrl.experiments.assembly.factor_swap import (
    EpisodeFactors,
    compose_factors,
    readout_effective_margin_fields_batched,
    trace_natural_episode,
)
from fsrl.experiments.assembly.trajectory import load_frozen_evaluator
from fsrl.experiments.assembly.write_localization import (
    replay_without_relation_history,
    row_cosine,
    trace_support_trial,
)
from fsrl.infra.provenance import file_sha256, load_json
from fsrl.infra.study_registry import (
    resolve_record,
    validate_registered_file,
)
from fsrl.infra.study_registry import resolve_registered_path as resolve_path
from fsrl.paths import REPO_ROOT
from fsrl.tasks.registered_protocol import load_ranking_protocol

ROOT = REPO_ROOT
DEFAULT_SPECIFICATION_PATH = resolve_record(
    "benchmarks/history_state_factorial_v1.json"
)
DEFAULT_OUTPUT_PATH = resolve_record("results/history_state_factorial_v1.json")
CELL_NAMES = ("NN", "NH", "HN", "HH")


def validate_registered_sources(specification: dict) -> dict:
    sources = specification["registered_sources"]
    names = (
        "pilot_specification",
        "protocol",
        "factor_swap_specification",
        "factor_swap_result",
        "factor_swap_implementation",
        "model_equation_source",
        "frozen_evaluator_source",
    )
    validated = {name: validate_registered_file(sources[name]) for name in names}
    artifacts = []
    for registration in sources["pilot_artifacts"]:
        row = {"seed": int(registration["seed"])}
        for prefix in ("checkpoint", "config", "behavior"):
            path = resolve_path(registration[f"{prefix}_path"])
            observed = file_sha256(path)
            if observed != registration[f"{prefix}_sha256"]:
                raise RuntimeError(f"registered SHA-256 mismatch: {path}")
            row[prefix] = {
                "path": registration[f"{prefix}_path"],
                "sha256": observed,
            }
        artifacts.append(row)
    validated["pilot_artifacts"] = artifacts
    return validated


def scalar_factorial(cells: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    nn, nh, hn, hh = (np.asarray(cells[name]) for name in CELL_NAMES)
    interaction = hh - hn - nh + nn
    return {
        **{
            f"cell_{name}": values
            for name, values in zip(CELL_NAMES, (nn, nh, hn, hh), strict=True)
        },
        "factor_generation_effect": 0.5 * ((nh - nn) + (hh - hn)),
        "baseline_expression_effect": 0.5 * ((hn - nn) + (hh - nh)),
        "interaction": interaction,
        "matched_history_contrast": 0.5 * interaction,
        "actual_restoration": hh - nn,
    }


def vector_factorial(cells: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    nn, nh, hn, hh = (np.asarray(cells[name]) for name in CELL_NAMES)
    factor = 0.5 * ((nh - nn) + (hh - hn))
    baseline = 0.5 * ((hn - nn) + (hh - nh))
    interaction = hh - hn - nh + nn
    return {
        "factor_vector_norm": np.linalg.norm(factor, axis=-1),
        "baseline_vector_norm": np.linalg.norm(baseline, axis=-1),
        "interaction_vector_norm": np.linalg.norm(interaction, axis=-1),
    }


def first_exposure_target_indices(factors: EpisodeFactors) -> np.ndarray:
    fourth = np.flatnonzero(np.all(factors.exposure == 4, axis=1))
    targets = np.full((len(fourth), factors.retained.shape[1]), -1, dtype=np.int64)
    for row, trial_index in enumerate(fourth):
        for subject in range(factors.retained.shape[1]):
            relation = factors.relations[trial_index, subject]
            matches = np.flatnonzero(
                (factors.exposure[:, subject] == 1)
                & np.all(factors.relations[:, subject] == relation, axis=1)
            )
            if len(matches) != 1:
                raise RuntimeError("same-relation first-exposure target is not unique")
            targets[row, subject] = int(matches[0])
    return targets


def _gather_first_targets(
    factors: EpisodeFactors, target_indices: np.ndarray
) -> np.ndarray:
    subjects = np.broadcast_to(np.arange(target_indices.shape[1]), target_indices.shape)
    return factors.natural_potential_updates[target_indices, subjects]


def _history_factors(
    evaluator,
    factors: EpisodeFactors,
    trial_index: int,
    effective_steps: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    relations = tuple(
        tuple(values) for values in factors.relations[trial_index].tolist()
    )
    history = replay_without_relation_history(evaluator, trial_index, relations)
    history_plus = trace_support_trial(evaluator, history, trial_index)
    history_zero = trace_support_trial(
        evaluator,
        history,
        trial_index,
        evidence_scales=np.zeros(evaluator.config.bs, dtype=np.float32),
    )
    step_index = torch.as_tensor(effective_steps, device=history.device)
    da_plus = torch.index_select(history_plus.da, 1, step_index)
    da_zero = torch.index_select(history_zero.da, 1, step_index)
    e_plus = torch.index_select(history_plus.eligibility_before, 1, step_index)
    e_zero = torch.index_select(history_zero.eligibility_before, 1, step_index)
    factor = compose_factors(
        0.5 * (da_plus + da_zero),
        da_plus - da_zero,
        0.5 * (e_plus + e_zero),
        e_plus - e_zero,
    )
    direct = torch.sum(
        torch.index_select(
            history_plus.intended_increment - history_zero.intended_increment,
            1,
            step_index,
        ),
        dim=1,
    )
    return (
        history_zero.final_fast_weights,
        history_plus.final_fast_weights,
        factor,
        {
            "history_factor_identity_max_abs_error": float(
                torch.max(torch.abs(factor - direct)).item()
            )
        },
    )


def collect_cells(
    evaluator,
    factors: EpisodeFactors,
    geometry,
    effective_steps: tuple[int, ...],
    tolerance: float,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict]:
    fourth_trials = np.flatnonzero(np.all(factors.exposure == 4, axis=1))
    alpha = evaluator.net.alpha.detach()
    groups = {
        "baseline_N": [],
        "baseline_H": [],
        "NN": [],
        "NH": [],
        "HN": [],
        "HH": [],
        "actual_H_plus": [],
    }
    validation = {
        "history_factor_identity_max_abs_error": 0.0,
        "natural_factor_identity_max_abs_error": 0.0,
        "natural_baseline_replay_max_abs_error": 0.0,
        "natural_cell_replay_max_abs_error": 0.0,
        "history_cell_replay_max_abs_error": 0.0,
        "stable_omitted_factor_max_abs": 0.0,
    }

    natural_factor = compose_factors(
        factors.da_mean,
        factors.da_difference,
        factors.eligibility_mean,
        factors.eligibility_difference,
    )
    for trial_index in fourth_trials:
        print(
            f"[history-closure] trial {trial_index + 1}/"
            f"{evaluator.protocol.support_trials}",
            file=sys.stderr,
        )
        history_zero, history_plus, history_factor, factor_validation = (
            _history_factors(evaluator, factors, int(trial_index), effective_steps)
        )
        validation["history_factor_identity_max_abs_error"] = max(
            validation["history_factor_identity_max_abs_error"],
            factor_validation["history_factor_identity_max_abs_error"],
        )
        baseline_n = factors.baseline_modulation[trial_index]
        baseline_h = alpha * history_zero
        factor_n = alpha * natural_factor[trial_index]
        factor_h = alpha * history_factor
        groups["baseline_N"].append(baseline_n)
        groups["baseline_H"].append(baseline_h)
        groups["NN"].append(baseline_n + factor_n)
        groups["NH"].append(baseline_n + factor_h)
        groups["HN"].append(baseline_h + factor_n)
        groups["HH"].append(baseline_h + factor_h)
        groups["actual_H_plus"].append(alpha * history_plus)
        omitted = ~factors.retained[trial_index]
        if np.any(omitted):
            omitted_t = torch.as_tensor(omitted, device=factor_n.device)
            validation["stable_omitted_factor_max_abs"] = max(
                validation["stable_omitted_factor_max_abs"],
                float(torch.max(torch.abs(factor_n[omitted_t])).item()),
                float(torch.max(torch.abs(factor_h[omitted_t])).item()),
            )

    stacked = {name: torch.stack(rows) for name, rows in groups.items()}
    order = tuple(groups)
    fields = readout_effective_margin_fields_batched(
        evaluator,
        torch.cat([stacked[name] for name in order], dim=0),
        geometry,
    )
    count = len(fourth_trials)
    field_groups = {
        name: fields[index * count : (index + 1) * count]
        for index, name in enumerate(order)
    }
    validation["natural_baseline_replay_max_abs_error"] = float(
        np.max(
            np.abs(field_groups["baseline_N"] - factors.baseline_fields[fourth_trials])
        )
    )
    validation["natural_cell_replay_max_abs_error"] = float(
        np.max(np.abs(field_groups["NN"] - factors.plus_fields[fourth_trials]))
    )
    validation["history_cell_replay_max_abs_error"] = float(
        np.max(np.abs(field_groups["HH"] - field_groups["actual_H_plus"]))
    )
    cell_potentials = {
        "NN": hodge_potentials(
            field_groups["NN"] - field_groups["baseline_N"], geometry
        ),
        "NH": hodge_potentials(
            field_groups["NH"] - field_groups["baseline_N"], geometry
        ),
        "HN": hodge_potentials(
            field_groups["HN"] - field_groups["baseline_H"], geometry
        ),
        "HH": hodge_potentials(
            field_groups["HH"] - field_groups["baseline_H"], geometry
        ),
    }
    omitted = ~factors.retained[fourth_trials]
    if np.any(omitted):
        omitted_update = max(
            float(np.max(np.abs(values[omitted])))
            for values in cell_potentials.values()
        )
        validation["stable_omitted_update_max_abs"] = omitted_update
    else:
        validation["stable_omitted_update_max_abs"] = 0.0

    targets = _gather_first_targets(factors, first_exposure_target_indices(factors))
    alignments = {
        name: row_cosine(values, targets, tolerance)
        for name, values in cell_potentials.items()
    }
    return (
        cell_potentials,
        np.asarray(factors.retained[fourth_trials]),
        {
            **validation,
            "target_alignments": alignments,
        },
    )


def cell_metrics(
    cells: dict[str, np.ndarray],
    target_alignments: dict[str, np.ndarray],
    tolerance: float,
) -> dict[str, dict[str, np.ndarray]]:
    norms = {name: np.linalg.norm(values, axis=-1) for name, values in cells.items()}
    directions = {
        "natural_factor_across_baselines_cosine": row_cosine(
            cells["NN"], cells["HN"], tolerance
        ),
        "history_factor_across_baselines_cosine": row_cosine(
            cells["NH"], cells["HH"], tolerance
        ),
        "factor_source_at_natural_baseline_cosine": row_cosine(
            cells["NN"], cells["NH"], tolerance
        ),
        "factor_source_at_history_baseline_cosine": row_cosine(
            cells["HN"], cells["HH"], tolerance
        ),
    }
    return {
        "potential_norm": scalar_factorial(norms),
        "potential_vector": vector_factorial(cells),
        "first_exposure_relation_alignment": scalar_factorial(target_alignments),
        "direction_cosines": directions,
    }


def summarize_metric_group(
    metrics: dict[str, np.ndarray],
    mask: np.ndarray,
    counts: np.ndarray,
    interval: float,
) -> tuple[dict, dict]:
    subjects = {
        name: masked_column_mean(values, mask) for name, values in metrics.items()
    }
    return (
        {
            name: summarize_subjects(values, counts, interval=interval)
            for name, values in subjects.items()
        },
        {name: json_values(values) for name, values in subjects.items()},
    )


def _signed_classification(summary: dict) -> str:
    lower = summary["bootstrap"]["lower"]
    upper = summary["bootstrap"]["upper"]
    if lower > 0.0:
        return "positive"
    if upper < 0.0:
        return "suppressive"
    return "unresolved"


def summarize_seed(
    metrics: dict[str, dict[str, np.ndarray]],
    mask: np.ndarray,
    validation: dict,
    counts: np.ndarray,
    *,
    interval: float,
    tolerance: float,
) -> tuple[dict, dict]:
    summaries = {}
    raw = {}
    for group, values in metrics.items():
        summaries[group], raw[group] = summarize_metric_group(
            values, mask, counts, interval
        )
    scalar = summaries["potential_norm"]
    alignment = summaries["first_exposure_relation_alignment"]
    reproduction_tolerance = float(max(tolerance, 32.0 * np.finfo(np.float32).eps))
    validation["implementation_reproduction_tolerance"] = reproduction_tolerance
    for name, value in validation.items():
        if (
            name.endswith(("max_abs_error", "max_abs"))
            and value > reproduction_tolerance
        ):
            raise RuntimeError(f"implementation validation failed: {name}")
    matched_dependency_error = abs(
        scalar["matched_history_contrast"]["mean"] - 0.5 * scalar["interaction"]["mean"]
    )
    alignment_dependency_error = abs(
        alignment["matched_history_contrast"]["mean"]
        - 0.5 * alignment["interaction"]["mean"]
    )
    if max(matched_dependency_error, alignment_dependency_error) > tolerance:
        raise RuntimeError("matched-history contrast is not interaction / 2")
    validation["matched_interaction_dependency_max_abs_error"] = max(
        matched_dependency_error, alignment_dependency_error
    )
    classifications = {
        "magnitude_factor_generation": _signed_classification(
            scalar["factor_generation_effect"]
        ),
        "magnitude_baseline_expression": _signed_classification(
            scalar["baseline_expression_effect"]
        ),
        "magnitude_interaction": _signed_classification(scalar["interaction"]),
        "alignment_factor_generation": _signed_classification(
            alignment["factor_generation_effect"]
        ),
        "alignment_baseline_expression": _signed_classification(
            alignment["baseline_expression_effect"]
        ),
        "alignment_interaction": _signed_classification(alignment["interaction"]),
    }
    locus_resolved = any(
        value != "unresolved"
        for name, value in classifications.items()
        if name.startswith("magnitude_")
    )
    return {
        **summaries,
        "registered_classification": classifications,
        "history_locus_resolved": locus_resolved,
        "validation": validation,
    }, raw


def run_history_state_factorial(
    specification_path: Path = DEFAULT_SPECIFICATION_PATH,
) -> dict:
    specification = load_json(specification_path)
    artifact_validation = validate_registered_sources(specification)
    sources = specification["registered_sources"]
    pilot_specification = load_json(
        resolve_path(sources["pilot_specification"]["path"])
    )
    protocol = load_ranking_protocol(resolve_path(sources["protocol"]["path"]))
    geometry = build_complete_graph_geometry(protocol)
    contract = specification["execution_contract"]
    effective_steps = tuple(int(value) for value in contract["effective_support_steps"])
    tolerance = float(contract["floating_zero_tolerance"])
    bootstrap = specification["bootstrap"]
    interval = float(bootstrap["interval"])
    rng = np.random.default_rng(int(bootstrap["seed"]))
    per_seed = {}

    for registration in sources["pilot_artifacts"]:
        seed = int(registration["seed"])
        print(f"[history-closure] loading frozen seed {seed}", file=sys.stderr)
        evaluator, behavior = load_frozen_evaluator(
            registration, pilot_specification, protocol
        )
        counts = bootstrap_counts(rng, int(bootstrap["samples"]), evaluator.config.bs)
        factors = trace_natural_episode(evaluator, geometry, effective_steps)
        cells, mask, cell_validation = collect_cells(
            evaluator, factors, geometry, effective_steps, tolerance
        )
        target_alignments = cell_validation.pop("target_alignments")
        metrics = cell_metrics(cells, target_alignments, tolerance)
        summary, raw = summarize_seed(
            metrics,
            mask,
            {**factors.validation, **cell_validation},
            counts,
            interval=interval,
            tolerance=tolerance,
        )
        per_seed[str(seed)] = {
            "seed": seed,
            "subjects": evaluator.config.bs,
            "checkpoint": behavior["checkpoint"],
            **summary,
            "raw_subject_level": raw,
        }

    classification_names = tuple(
        next(iter(per_seed.values()))["registered_classification"]
    )
    replicated = {}
    for name in classification_names:
        values = [row["registered_classification"][name] for row in per_seed.values()]
        replicated[name] = values[0] if len(set(values)) == 1 else "seed_dependent"
    magnitude_resolved = any(
        replicated[name] in ("positive", "suppressive")
        for name in (
            "magnitude_factor_generation",
            "magnitude_baseline_expression",
            "magnitude_interaction",
        )
    )
    all_replays_valid = all(
        row["validation"]["natural_cell_replay_max_abs_error"]
        <= row["validation"]["implementation_reproduction_tolerance"]
        and row["validation"]["history_cell_replay_max_abs_error"]
        <= row["validation"]["implementation_reproduction_tolerance"]
        for row in per_seed.values()
    )
    pilot_stop_rule_met = bool(magnitude_resolved and all_replays_valid)
    return {
        "schema_version": 1,
        "diagnostic_id": specification["diagnostic_id"],
        "registration_status": specification["registration_status"],
        "registration_parent_commit": specification["registration_parent_commit"],
        "claim_boundary": specification["claim_boundary"],
        "working_theory": specification["working_theory"],
        "device": {"neural_replay": DEVICE, "summaries": "cpu_numpy"},
        "artifact_validation": artifact_validation,
        "execution_contract": contract,
        "state_and_factor_contract": specification["state_and_factor_contract"],
        "bootstrap": bootstrap,
        "pilot_seeds": per_seed,
        "overall_diagnosis": {
            "replicated_classification": replicated,
            "pilot_stop_rule_met": pilot_stop_rule_met,
            "formal_confirmation_status": "deferred; formal seeds remain untouched",
            "confirmation_rule": specification["decision_logic"]["confirmation_rule"],
        },
    }


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Run the registered history state-by-factor closure."
    )
    parser.add_argument(
        "--specification", type=Path, default=DEFAULT_SPECIFICATION_PATH
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args(args)


def main(args=None):
    parsed = parse_args(args)
    result = run_history_state_factorial(parsed.specification)
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    parsed.output.parent.mkdir(parents=True, exist_ok=True)
    with parsed.output.open("w", encoding="utf-8") as handle:
        handle.write(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
