"""Audit learned-relation residuals in frozen response hidden states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fsrl.core.config import DEVICE, NUMRESPONSESTEP
from fsrl.evaluation.frozen_fast_weight import FastWeightIntervention
from fsrl.experiments.assembly.diagnostics import file_sha256, load_json, resolve_path
from fsrl.experiments.assembly.trajectory import (
    CompleteGraphGeometry,
    bootstrap_counts,
    build_complete_graph_geometry,
    gradient_energy_fraction,
    load_frozen_evaluator,
    ordered_query_schedule,
    summarize_subjects,
    vector_gradient_energy_fraction,
)
from fsrl.infrastructure.formal_runtime import configure_formal_runtime
from fsrl.infrastructure.study_registry import registered_file_sha256, resolve_record
from fsrl.paths import REPO_ROOT
from fsrl.tasks.registered_protocol import RankingProtocol, load_ranking_protocol

ROOT = REPO_ROOT
DEFAULT_SPECIFICATION_PATH = resolve_record("benchmarks/hidden_residual_audit_v1.json")
DEFAULT_OUTPUT_PATH = resolve_record("results/hidden_residual_audit_v1.json")


def validate_registered_sources(specification: dict) -> dict:
    sources = specification["registered_sources"]
    validated = {}
    for name, registration in sources.items():
        if name == "pilot_artifacts":
            continue
        path = resolve_path(registration["path"])
        observed = registered_file_sha256(
            registration["path"], registration["sha256"], resolved_path=path
        )
        if observed != registration["sha256"]:
            raise RuntimeError(f"registered SHA-256 mismatch: {path}")
        validated[name] = {"path": registration["path"], "sha256": observed}

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


def vector_hodge_components(
    fields: np.ndarray, geometry: CompleteGraphGeometry
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(fields, dtype=np.float64)
    if values.shape[-2] != len(geometry.pairs):
        raise ValueError("vector field does not match the complete-graph edge order")
    gradient = np.einsum("ef,...fd->...ed", geometry.projection, values)
    return gradient, values - gradient


def _hidden_and_margin_fields(
    evaluator, fast_weights, geometry: CompleteGraphGeometry
) -> tuple[np.ndarray, np.ndarray, float]:
    schedules = ordered_query_schedule(geometry, evaluator.config.bs)
    hidden_rows, logit_rows = evaluator.readout_hidden_and_logit_trajectories(
        fast_weights, schedules
    )
    hidden = np.empty(
        (evaluator.config.bs, len(geometry.pairs), evaluator.config.hs),
        dtype=np.float64,
    )
    margins = np.empty((evaluator.config.bs, len(geometry.pairs)), dtype=np.float64)
    for subject in range(evaluator.config.bs):
        for edge, pair in enumerate(geometry.pairs):
            reverse = (pair[1], pair[0])
            hidden[subject, edge] = 0.5 * (
                hidden_rows[subject][pair][NUMRESPONSESTEP]
                - hidden_rows[subject][reverse][NUMRESPONSESTEP]
            )
            margins[subject, edge] = 0.5 * (
                logit_rows[subject][pair][NUMRESPONSESTEP]
                - logit_rows[subject][reverse][NUMRESPONSESTEP]
            )

    output_direction = (
        (evaluator.net.h2o.weight[1] - evaluator.net.h2o.weight[0])
        .detach()
        .cpu()
        .numpy()
    )
    analytic = np.einsum("beh,h->be", hidden, output_direction)
    return hidden, margins, float(np.max(np.abs(analytic - margins)))


def _relation_geometry(
    protocol: RankingProtocol, geometry: CompleteGraphGeometry
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, ...]]:
    edge_lookup = {pair: index for index, pair in enumerate(geometry.pairs)}
    direct_edges = []
    signs = []
    remote_masks = []
    for relation in protocol.support_pairs_higher_lower:
        canonical = tuple(sorted(relation))
        edge = edge_lookup[canonical]
        direct_edges.append(edge)
        signs.append(geometry.true_sign[edge])
        endpoints = set(relation)
        remote_masks.append(
            np.asarray(
                [not endpoints.intersection(pair) for pair in geometry.pairs],
                dtype=bool,
            )
        )
    return (
        np.asarray(direct_edges, dtype=np.int64),
        np.asarray(signs, dtype=np.float64),
        tuple(remote_masks),
    )


def cross_validated_local_direction(
    residual_influence: np.ndarray,
    retained: np.ndarray,
    direct_edges: np.ndarray,
    correctness_signs: np.ndarray,
    remote_masks: tuple[np.ndarray, ...],
    output_direction: np.ndarray,
    *,
    subject_folds: int,
) -> dict[str, np.ndarray]:
    values = np.asarray(residual_influence, dtype=np.float64)
    relation_count, subjects, _edges, hidden_size = values.shape
    if retained.shape != (relation_count, subjects):
        raise ValueError("retained mask does not match relation and subject axes")
    if relation_count < 2 or subjects < subject_folds:
        raise ValueError("cross-validation needs multiple relations and subjects")

    output = np.asarray(output_direction, dtype=np.float64)
    output_norm = np.linalg.norm(output)
    if output.shape != (hidden_size,) or output_norm == 0.0:
        raise ValueError("output direction must be a nonzero hidden-state vector")
    output /= output_norm

    local_direct = np.full((relation_count, subjects), np.nan)
    local_remote = np.full((relation_count, subjects), np.nan)
    output_direct = np.full((relation_count, subjects), np.nan)
    output_remote = np.full((relation_count, subjects), np.nan)
    direction_cosine = np.full((relation_count, subjects), np.nan)
    subject_fold = np.arange(subjects) % subject_folds

    for held_relation in range(relation_count):
        for held_fold in range(subject_folds):
            aligned_training = []
            for relation in range(relation_count):
                if relation == held_relation:
                    continue
                mask = (subject_fold != held_fold) & retained[relation]
                if np.any(mask):
                    aligned_training.append(
                        correctness_signs[relation]
                        * values[relation, mask, direct_edges[relation]]
                    )
            if not aligned_training:
                raise RuntimeError("cross-validation fold has no training examples")
            local = np.mean(np.concatenate(aligned_training, axis=0), axis=0)
            local_norm = np.linalg.norm(local)
            if local_norm == 0.0:
                raise RuntimeError("cross-validated local direction is zero")
            local /= local_norm

            test_subjects = np.flatnonzero(subject_fold == held_fold)
            direct_vectors = values[
                held_relation, test_subjects, direct_edges[held_relation]
            ]
            remote_vectors = values[held_relation, test_subjects][
                :, remote_masks[held_relation]
            ]
            local_direct[held_relation, test_subjects] = correctness_signs[
                held_relation
            ] * (direct_vectors @ local)
            local_remote[held_relation, test_subjects] = np.mean(
                np.abs(remote_vectors @ local), axis=1
            )
            output_direct[held_relation, test_subjects] = correctness_signs[
                held_relation
            ] * (direct_vectors @ output)
            output_remote[held_relation, test_subjects] = np.mean(
                np.abs(remote_vectors @ output), axis=1
            )
            direction_cosine[held_relation, test_subjects] = float(local @ output)

    return {
        "local_direct_correctness": local_direct,
        "local_remote_absolute": local_remote,
        "output_direct_correctness": output_direct,
        "output_remote_absolute": output_remote,
        "local_output_direction_cosine": direction_cosine,
    }


def _retained_subject_mean(values: np.ndarray, retained: np.ndarray) -> np.ndarray:
    rows = np.where(retained, np.asarray(values, dtype=np.float64), np.nan)
    if np.any(np.sum(retained, axis=0) == 0):
        raise RuntimeError("every subject must retain at least one support relation")
    return np.nanmean(rows, axis=0)


def _json_values(values: np.ndarray) -> list:
    return [None if not np.isfinite(value) else float(value) for value in values]


def _summarize_seed(
    *,
    seed: int,
    evaluator,
    geometry: CompleteGraphGeometry,
    intact_hidden: np.ndarray,
    intact_margins: np.ndarray,
    residual_influence: np.ndarray,
    full_influence: np.ndarray,
    retained: np.ndarray,
    direct_edges: np.ndarray,
    correctness_signs: np.ndarray,
    remote_masks: tuple[np.ndarray, ...],
    output_direction: np.ndarray,
    counts: np.ndarray,
    interval: float,
    subject_folds: int,
    tolerance: float,
    reproduction_error: float,
) -> dict:
    relation_count, subjects, _edges, _hidden = residual_influence.shape
    residual_norm = np.linalg.norm(residual_influence, axis=-1)
    direct_norm = np.stack(
        [residual_norm[r, :, direct_edges[r]] for r in range(relation_count)]
    )
    remote_norm = np.stack(
        [
            np.mean(residual_norm[r][:, remote_masks[r]], axis=1)
            for r in range(relation_count)
        ]
    )
    residual_energy = np.sum(residual_influence * residual_influence, axis=(-2, -1))
    total_energy = np.sum(full_influence * full_influence, axis=(-2, -1))
    residual_fraction = np.divide(
        residual_energy,
        total_energy,
        out=np.full_like(residual_energy, np.nan),
        where=total_energy > 0.0,
    )

    projections = cross_validated_local_direction(
        residual_influence,
        retained,
        direct_edges,
        correctness_signs,
        remote_masks,
        output_direction,
        subject_folds=subject_folds,
    )
    relation_metrics = {
        "direct_residual_norm": direct_norm,
        "remote_residual_norm": remote_norm,
        "direct_minus_remote_residual_norm": direct_norm - remote_norm,
        "residual_energy_fraction": residual_fraction,
        **projections,
        "local_direct_minus_remote_absolute": (
            projections["local_direct_correctness"]
            - projections["local_remote_absolute"]
        ),
        "output_direct_minus_remote_absolute": (
            projections["output_direct_correctness"]
            - projections["output_remote_absolute"]
        ),
        "local_minus_output_direct_correctness": (
            projections["local_direct_correctness"]
            - projections["output_direct_correctness"]
        ),
    }
    subject_metrics = {
        name: _retained_subject_mean(values, retained)
        for name, values in relation_metrics.items()
    }
    summaries = {
        name: summarize_subjects(values, counts, interval=interval)
        for name, values in subject_metrics.items()
    }

    hidden_gradient = vector_gradient_energy_fraction(intact_hidden, geometry)
    output_gradient = gradient_energy_fraction(intact_margins, geometry)
    intact_summary = {
        "hidden_vector_gradient_energy_fraction": summarize_subjects(
            hidden_gradient, counts, interval=interval
        ),
        "output_projected_gradient_energy_fraction": summarize_subjects(
            output_gradient, counts, interval=interval
        ),
        "output_minus_hidden_gradient_energy_fraction": summarize_subjects(
            output_gradient - hidden_gradient, counts, interval=interval
        ),
    }

    omitted_selector = np.broadcast_to(
        (~retained)[..., None, None], full_influence.shape
    )
    omitted_max = (
        float(np.max(np.abs(full_influence[omitted_selector])))
        if np.any(~retained)
        else 0.0
    )
    diagnosis = {
        "causal_local_residual": (
            summaries["direct_minus_remote_residual_norm"]["bootstrap"]["lower"] > 0.0
            and omitted_max <= tolerance
        ),
        "linearly_accessible_local_direction": (
            summaries["local_direct_correctness"]["bootstrap"]["lower"] > 0.0
            and summaries["local_direct_minus_remote_absolute"]["bootstrap"]["lower"]
            > 0.0
        ),
        "fixed_readout_suppression": (
            summaries["local_minus_output_direct_correctness"]["bootstrap"]["lower"]
            > 0.0
        ),
    }
    diagnosis["hidden_local_trace_supported"] = all(diagnosis.values())

    per_relation = []
    for relation_index, relation in enumerate(
        evaluator.protocol.support_pairs_higher_lower
    ):
        mask = retained[relation_index]
        per_relation.append(
            {
                "relation": [int(relation[0]), int(relation[1])],
                "retained_subjects": int(np.sum(mask)),
                "stable_omitted_subjects": int(np.sum(~mask)),
                "retained": {
                    name: summarize_subjects(
                        np.where(mask, values[relation_index], np.nan),
                        counts,
                        interval=interval,
                    )
                    for name, values in relation_metrics.items()
                },
            }
        )

    return {
        "seed": seed,
        "subjects": subjects,
        "intact_field": intact_summary,
        "loo_residual": summaries,
        "per_relation": per_relation,
        "validation": {
            "hidden_to_logit_projection_max_abs_error": reproduction_error,
            "stable_omitted_hidden_influence_max_abs": omitted_max,
            "floating_reproduction_tolerance": tolerance,
        },
        "directional_diagnosis": diagnosis,
        "raw_subject_level": {
            name: _json_values(values) for name, values in subject_metrics.items()
        },
    }


def _run_seed(
    registration: dict,
    pilot_specification: dict,
    protocol: RankingProtocol,
    geometry: CompleteGraphGeometry,
    counts: np.ndarray,
    specification: dict,
) -> dict:
    evaluator, _behavior = load_frozen_evaluator(
        registration, pilot_specification, protocol
    )
    intact_weights = evaluator.learn_fast_weights(FastWeightIntervention.INTACT)
    intact_hidden, intact_margins, reproduction_error = _hidden_and_margin_fields(
        evaluator, intact_weights, geometry
    )
    relations = tuple(protocol.support_pairs_higher_lower)
    hidden_rows = []
    retained_rows = []
    for relation in relations:
        weights = evaluator.initialize_fast_weights()
        for trial_index in range(protocol.support_trials):
            weights = evaluator.advance_support_trial(
                weights,
                trial_index,
                zero_relations=frozenset((relation,)),
            )
        loo_hidden, _loo_margins, error = _hidden_and_margin_fields(
            evaluator, weights, geometry
        )
        reproduction_error = max(reproduction_error, error)
        hidden_rows.append(intact_hidden - loo_hidden)
        if evaluator.subject_relation_gains is None:
            retained_rows.append(np.ones(evaluator.config.bs, dtype=bool))
        else:
            retained_rows.append(
                np.asarray(
                    [
                        evaluator.subject_relation_gains[subject][relation] > 0.0
                        for subject in range(evaluator.config.bs)
                    ],
                    dtype=bool,
                )
            )

    full_influence = np.asarray(hidden_rows)
    _gradient, residual_influence = vector_hodge_components(full_influence, geometry)
    retained = np.asarray(retained_rows)
    direct_edges, correctness_signs, remote_masks = _relation_geometry(
        protocol, geometry
    )
    output_direction = (
        (evaluator.net.h2o.weight[1] - evaluator.net.h2o.weight[0])
        .detach()
        .cpu()
        .numpy()
    )
    contract = specification["execution_contract"]
    result = _summarize_seed(
        seed=int(registration["seed"]),
        evaluator=evaluator,
        geometry=geometry,
        intact_hidden=intact_hidden,
        intact_margins=intact_margins,
        residual_influence=residual_influence,
        full_influence=full_influence,
        retained=retained,
        direct_edges=direct_edges,
        correctness_signs=correctness_signs,
        remote_masks=remote_masks,
        output_direction=output_direction,
        counts=counts,
        interval=float(specification["bootstrap"]["interval"]),
        subject_folds=int(
            specification["cross_validated_local_direction"]["subject_folds"]
        ),
        tolerance=float(contract["floating_reproduction_tolerance"]),
        reproduction_error=reproduction_error,
    )
    result["checkpoint"] = {
        "path": registration["checkpoint_path"],
        "sha256": registration["checkpoint_sha256"],
    }
    return result


def _overall_diagnosis(seed_results: dict[str, dict]) -> dict:
    keys = (
        "causal_local_residual",
        "linearly_accessible_local_direction",
        "fixed_readout_suppression",
        "hidden_local_trace_supported",
    )
    replicated = {
        f"{key}_replicated_across_development_seeds": all(
            row["directional_diagnosis"][key] for row in seed_results.values()
        )
        for key in keys
    }
    if replicated["hidden_local_trace_supported_replicated_across_development_seeds"]:
        next_step = (
            "test a selectively causal local readout or objective on new v2 "
            "development seeds before adding a memory store"
        )
    elif replicated["causal_local_residual_replicated_across_development_seeds"]:
        next_step = (
            "localize high-dimensional or relation-specific residual content; "
            "do not claim a common readout channel"
        )
    else:
        next_step = (
            "audit earlier support-state representations before proposing a new "
            "persistent local mechanism"
        )
    return {**replicated, "next_step": next_step}


def run_hidden_residual_audit(
    specification_path: Path = DEFAULT_SPECIFICATION_PATH,
) -> dict:
    specification_path = resolve_path(str(specification_path))
    specification = load_json(specification_path)
    validation = validate_registered_sources(specification)
    runtime = configure_formal_runtime()
    if not runtime["cuda_available"] or DEVICE != "cuda":
        raise RuntimeError("hidden residual audit requires a visible CUDA device")

    sources = specification["registered_sources"]
    pilot_specification = load_json(
        resolve_path(sources["pilot_specification"]["path"])
    )
    protocol = load_ranking_protocol(resolve_path(sources["protocol"]["path"]))
    geometry = build_complete_graph_geometry(protocol)
    bootstrap = specification["bootstrap"]
    rng = np.random.default_rng(int(bootstrap["seed"]))
    counts = bootstrap_counts(
        rng,
        int(bootstrap["samples"]),
        int(pilot_specification["evaluation"]["batch_size"]),
    )
    seed_results = {
        str(registration["seed"]): _run_seed(
            registration,
            pilot_specification,
            protocol,
            geometry,
            counts,
            specification,
        )
        for registration in sources["pilot_artifacts"]
    }
    declared = {str(seed) for seed in specification["execution_contract"]["seeds"]}
    if set(seed_results) != declared:
        raise RuntimeError("hidden residual audit did not report every declared seed")

    return {
        "schema_version": 1,
        "diagnostic_id": specification["diagnostic_id"],
        "registration_status": specification["registration_status"],
        "claim_boundary": specification["claim_boundary"],
        "specification": {
            "path": str(Path(specification_path).relative_to(ROOT)),
            "sha256": file_sha256(specification_path),
        },
        "implementation": {
            "path": "fsrl/hidden_residual_audit.py",
            "sha256": file_sha256(Path(__file__)),
        },
        "execution_runtime": runtime,
        "artifact_validation": validation,
        "estimands": {
            "representation": specification["representation_contract"],
            "cross_validated_local_direction": specification[
                "cross_validated_local_direction"
            ],
            "primary": specification["primary_estimands"],
            "secondary": specification["secondary_estimands"],
        },
        "bootstrap": bootstrap,
        "seed_results": seed_results,
        "overall_diagnosis": _overall_diagnosis(seed_results),
        "formal_seed_access": False,
    }


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Audit hidden learned-relation residuals in frozen pilots."
    )
    parser.add_argument(
        "--specification", type=Path, default=DEFAULT_SPECIFICATION_PATH
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args(args)


def main(args=None):
    parsed = parse_args(args)
    result = run_hidden_residual_audit(parsed.specification)
    parsed.output.parent.mkdir(parents=True, exist_ok=True)
    with parsed.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
