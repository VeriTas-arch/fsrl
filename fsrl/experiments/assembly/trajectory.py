"""Registered support-prefix and relation-LOO diagnostics for frozen pilots."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fsrl.analysis.hodge import (
    CompleteGraphGeometry,
    build_complete_graph_geometry,
    gradient_energy_fraction,
    hodge_potentials,
    normalize_potentials,
    potential_alignment,
    vector_gradient_energy_fraction,
)
from fsrl.analysis.posterior import ExactRankingPosterior
from fsrl.analysis.statistics import (
    bootstrap_counts,
    json_values,
    masked_column_mean,
    summarize_difference,
    summarize_subjects,
)
from fsrl.core.config import DEVICE, NUMRESPONSESTEP
from fsrl.evaluation.fields import ordered_query_schedule, readout_margin_fields
from fsrl.evaluation.frozen_fast_weight import (
    FastWeightIntervention,
    FrozenFastWeightEvaluator,
    load_retro_checkpoint,
)
from fsrl.experiments.confirmation.behavioral import validate_checkpoint
from fsrl.infra.provenance import file_sha256, load_json
from fsrl.infra.study_registry import (
    resolve_record,
    validate_registered_file,
)
from fsrl.infra.study_registry import resolve_registered_path as resolve_path
from fsrl.paths import REPO_ROOT
from fsrl.tasks.registered_protocol import RankingProtocol, load_ranking_protocol

ROOT = REPO_ROOT
DEFAULT_SPECIFICATION_PATH = resolve_record("benchmarks/assembly_trajectory_v1.json")
DEFAULT_OUTPUT_PATH = resolve_record("results/assembly_trajectory_v1.json")


@dataclass(frozen=True)
class ExactTrajectory:
    fields: np.ndarray
    distributional_potentials: np.ndarray
    expected_rank_potentials: np.ndarray
    map_potentials: np.ndarray
    expected_rank_equivalence_error: np.ndarray


def exact_prefix_trajectory(
    evaluator: FrozenFastWeightEvaluator,
    protocol: RankingProtocol,
    geometry: CompleteGraphGeometry,
    *,
    temperature: float,
) -> ExactTrajectory:
    """Compute all exact posterior prefixes with incrementally updated energies."""

    exact = ExactRankingPosterior(protocol.n_items, temperature=temperature)
    pair_masks = np.stack(
        [
            exact.positions[:, first] < exact.positions[:, second]
            for first, second in geometry.pairs
        ],
        axis=1,
    ).astype(np.float64)
    energies = np.zeros((evaluator.config.bs, exact.n_hypotheses), dtype=np.float64)
    fields = []
    distributional = []
    expected_rank = []
    map_potentials = []
    equivalence_error = []

    for prefix in range(protocol.support_trials + 1):
        if prefix > 0:
            trial_index = prefix - 1
            for subject, schedule in enumerate(evaluator.support_schedules):
                trial = schedule[trial_index]
                predicted = (
                    exact.positions[:, trial.lower_item]
                    - exact.positions[:, trial.higher_item]
                ) / float(protocol.n_items - 1)
                residual = predicted - abs(trial.signed_magnitude)
                energies[subject] += (
                    evaluator._encoding_reliability(subject, trial_index)
                    * residual
                    * residual
                )

        log_weights = -energies / exact.temperature
        log_weights -= np.max(log_weights, axis=1, keepdims=True)
        probabilities = np.exp(log_weights)
        probabilities /= np.sum(probabilities, axis=1, keepdims=True)
        prefix_fields = 2.0 * (probabilities @ pair_masks) - 1.0
        prefix_distributional = normalize_potentials(
            hodge_potentials(prefix_fields, geometry)
        )
        prefix_expected = normalize_potentials(
            -(probabilities @ exact.positions.astype(np.float64))
        )
        map_indices = np.argmin(energies, axis=1)
        prefix_map = normalize_potentials(
            -exact.positions[map_indices].astype(np.float64)
        )
        fields.append(prefix_fields)
        distributional.append(prefix_distributional)
        expected_rank.append(prefix_expected)
        map_potentials.append(prefix_map)
        equivalence_error.append(
            np.max(np.abs(prefix_distributional - prefix_expected), axis=1)
        )

    return ExactTrajectory(
        fields=np.asarray(fields),
        distributional_potentials=np.asarray(distributional),
        expected_rank_potentials=np.asarray(expected_rank),
        map_potentials=np.asarray(map_potentials),
        expected_rank_equivalence_error=np.asarray(equivalence_error),
    )


def validate_registered_sources(specification: dict) -> dict:
    sources = specification["registered_sources"]
    validated = {
        name: validate_registered_file(sources[name])
        for name in (
            "pilot_specification",
            "protocol",
            "assembly_diagnostic_specification",
            "assembly_diagnostic_result",
        )
    }
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


def load_frozen_evaluator(
    registration: dict,
    pilot_specification: dict,
    protocol: RankingProtocol,
) -> tuple[FrozenFastWeightEvaluator, dict]:
    seed = int(registration["seed"])
    checkpoint = resolve_path(registration["checkpoint_path"])
    validate_checkpoint(checkpoint, pilot_specification, seed)
    behavior = load_json(resolve_path(registration["behavior_path"]))
    net, config, checkpoint_info = load_retro_checkpoint(
        checkpoint, len(behavior["subjects"])
    )
    if behavior["checkpoint"]["sha256"] != checkpoint_info.sha256:
        raise RuntimeError(f"seed {seed} behavior and checkpoint do not match")
    evaluator = FrozenFastWeightEvaluator(
        net,
        config,
        protocol,
        cue_seed=int(behavior["cue_seed"]),
        support_seed=int(behavior["support_seed"]),
        cue_mode="permuted_shared",
        subject_encoding_mode=behavior["subject_encoding_mode"],
        subject_encoding_seed=int(behavior["subject_encoding_seed"]),
    )
    return evaluator, behavior


def _effect_class_masks(
    relation: tuple[int, int], geometry: CompleteGraphGeometry
) -> dict[str, np.ndarray]:
    endpoints = set(relation)
    overlaps = np.asarray(
        [len(endpoints.intersection(pair)) for pair in geometry.pairs]
    )
    return {
        "direct": overlaps == 2,
        "endpoint_sharing": overlaps == 1,
        "remote": overlaps == 0,
    }


def classified_effects(
    fields: np.ndarray,
    relations: tuple[tuple[int, int], ...],
    geometry: CompleteGraphGeometry,
) -> dict[str, dict[str, np.ndarray]]:
    values = np.asarray(fields, dtype=np.float64)
    if values.shape != (len(relations), len(geometry.pairs)):
        raise ValueError("one complete pair field is required per relation")
    absolute = {name: [] for name in ("direct", "endpoint_sharing", "remote")}
    aligned = {name: [] for name in ("direct", "endpoint_sharing", "remote")}
    for row, relation in zip(values, relations, strict=True):
        masks = _effect_class_masks(relation, geometry)
        for name, mask in masks.items():
            absolute[name].append(float(np.mean(np.abs(row[mask]))))
            aligned[name].append(float(np.mean(row[mask] * geometry.true_sign[mask])))
    return {
        "mean_absolute": {name: np.asarray(rows) for name, rows in absolute.items()},
        "mean_correctness_aligned": {
            name: np.asarray(rows) for name, rows in aligned.items()
        },
    }


def run_prefix_branches(
    evaluator: FrozenFastWeightEvaluator,
    geometry: CompleteGraphGeometry,
    *,
    tolerance: float,
) -> tuple[object, np.ndarray, dict]:
    """Run P_0..P_T and each matched zero-evidence one-step branch."""

    fast_weights = evaluator.initialize_fast_weights()
    prefix_fields = [readout_margin_fields(evaluator, fast_weights, geometry)]
    effect_rows = {
        measure: {name: [] for name in ("direct", "endpoint_sharing", "remote")}
        for measure in ("mean_absolute", "mean_correctness_aligned")
    }
    retained_rows = []
    relation_rows = []

    for trial_index in range(evaluator.protocol.support_trials):
        zero_weights = evaluator.advance_support_trial(
            fast_weights, trial_index, zero_evidence=True
        )
        natural_weights = evaluator.advance_support_trial(fast_weights, trial_index)
        natural_field = readout_margin_fields(evaluator, natural_weights, geometry)
        zero_field = readout_margin_fields(evaluator, zero_weights, geometry)
        delta = natural_field - zero_field
        relations = tuple(
            (trial.higher_item, trial.lower_item)
            for trial in (
                schedule[trial_index] for schedule in evaluator.support_schedules
            )
        )
        effects = classified_effects(delta, relations, geometry)
        for measure, classes in effects.items():
            for name, values in classes.items():
                effect_rows[measure][name].append(values)
        retained_rows.append(
            np.asarray(
                [
                    evaluator._encoding_reliability(subject, trial_index) > 0.0
                    for subject in range(evaluator.config.bs)
                ],
                dtype=bool,
            )
        )
        relation_rows.append(relations)
        prefix_fields.append(natural_field)
        fast_weights = natural_weights

    reference = evaluator.learn_fast_weights(FastWeightIntervention.INTACT)
    endpoint_error = float(np.max(np.abs((fast_weights - reference).cpu().numpy())))
    if endpoint_error > tolerance:
        raise RuntimeError(
            "incremental support endpoint does not match intact evaluator"
        )

    return (
        fast_weights,
        np.asarray(prefix_fields),
        {
            "effects": {
                measure: {name: np.asarray(values) for name, values in classes.items()}
                for measure, classes in effect_rows.items()
            },
            "retained": np.asarray(retained_rows),
            "relations": relation_rows,
            "endpoint_max_abs_error": endpoint_error,
        },
    )


def summarize_prefixes(
    prefix_fields: np.ndarray,
    exact: ExactTrajectory,
    geometry: CompleteGraphGeometry,
    counts: np.ndarray,
    *,
    interval: float,
) -> tuple[dict, dict]:
    potentials = hodge_potentials(prefix_fields, geometry)
    normalized = normalize_potentials(potentials)
    gradient_fraction = gradient_energy_fraction(prefix_fields, geometry)
    commitment = np.linalg.norm(potentials, axis=-1)
    final_potentials = np.broadcast_to(normalized[-1], normalized.shape)
    true_potentials = np.broadcast_to(geometry.true_potential, normalized.shape)
    targets = {
        "final_neural": final_potentials,
        "true_order": true_potentials,
        "exact_distributional": exact.distributional_potentials,
        "exact_expected_rank": exact.expected_rank_potentials,
        "exact_map": exact.map_potentials,
    }
    alignments = {
        name: potential_alignment(normalized, target)
        for name, target in targets.items()
    }
    summaries = []
    for prefix in range(prefix_fields.shape[0]):
        summaries.append(
            {
                "prefix": prefix,
                "gradient_energy_fraction": summarize_subjects(
                    gradient_fraction[prefix], counts, interval=interval
                ),
                "commitment_strength": summarize_subjects(
                    commitment[prefix], counts, interval=interval
                ),
                "alignment": {
                    target: {
                        metric: summarize_subjects(
                            values[prefix], counts, interval=interval
                        )
                        for metric, values in metrics.items()
                    }
                    for target, metrics in alignments.items()
                },
                "exact_expected_rank_equivalence_max_abs_error": (
                    summarize_subjects(
                        exact.expected_rank_equivalence_error[prefix],
                        counts,
                        interval=interval,
                    )
                ),
            }
        )
    raw = {
        "gradient_energy_fraction": json_values(gradient_fraction),
        "commitment_strength": json_values(commitment),
        "alignment": {
            target: {metric: json_values(values) for metric, values in metrics.items()}
            for target, metrics in alignments.items()
        },
        "exact_expected_rank_equivalence_max_abs_error": json_values(
            exact.expected_rank_equivalence_error
        ),
    }
    return {
        "prefixes": summaries,
        "final_distributional_minus_map_alignment": {
            metric: summarize_difference(
                alignments["exact_distributional"][metric][-1],
                alignments["exact_map"][metric][-1],
                counts,
                interval=interval,
            )
            for metric in ("cosine", "pearson", "kendall_tau")
        },
    }, raw


def summarize_matched_branches(
    branch: dict,
    counts: np.ndarray,
    *,
    interval: float,
    tolerance: float,
) -> tuple[dict, dict]:
    retained = branch["retained"]
    per_prefix = []
    raw = {
        "retained": retained.astype(int).tolist(),
        "relations": [
            [[int(first), int(second)] for first, second in rows]
            for rows in branch["relations"]
        ],
        "effects": {},
    }
    aggregate = {"retained": {}, "stable_omitted": {}}
    for measure, classes in branch["effects"].items():
        raw["effects"][measure] = {
            name: json_values(values) for name, values in classes.items()
        }
        for status, mask in (
            ("retained", retained),
            ("stable_omitted", ~retained),
        ):
            aggregate[status][measure] = {
                name: summarize_subjects(
                    masked_column_mean(values, mask), counts, interval=interval
                )
                for name, values in classes.items()
            }

    for trial_index in range(retained.shape[0]):
        row = {"prefix": trial_index + 1, "retained": {}, "stable_omitted": {}}
        for status, status_mask in (
            ("retained", retained[trial_index]),
            ("stable_omitted", ~retained[trial_index]),
        ):
            for measure, classes in branch["effects"].items():
                row[status][measure] = {
                    name: summarize_subjects(
                        np.where(status_mask, values[trial_index], np.nan),
                        counts,
                        interval=interval,
                    )
                    for name, values in classes.items()
                }
        per_prefix.append(row)

    omitted_max = max(
        float(np.max(np.abs(values[~retained]))) if np.any(~retained) else 0.0
        for classes in branch["effects"].values()
        for values in classes.values()
    )
    return {
        "incremental_endpoint_max_abs_error": branch["endpoint_max_abs_error"],
        "aggregate": aggregate,
        "per_prefix": per_prefix,
        "stable_omitted_max_abs_effect": omitted_max,
        "stable_omitted_within_tolerance": omitted_max <= tolerance,
    }, raw


def run_leave_one_relation_out(
    evaluator: FrozenFastWeightEvaluator,
    intact_fields: np.ndarray,
    geometry: CompleteGraphGeometry,
    counts: np.ndarray,
    *,
    interval: float,
    tolerance: float,
) -> tuple[dict, dict]:
    relations = tuple(evaluator.protocol.support_pairs_higher_lower)
    intact_potentials = hodge_potentials(intact_fields, geometry)
    retained_rows = []
    field_rows = []
    gradient_rows = []
    requested_third_rows = []
    relational_third_rows = []
    effect_rows = {
        measure: {name: [] for name in ("direct", "endpoint_sharing", "remote")}
        for measure in ("mean_absolute", "mean_correctness_aligned")
    }

    for relation in relations:
        fast_weights = evaluator.initialize_fast_weights()
        for trial_index in range(evaluator.protocol.support_trials):
            fast_weights = evaluator.advance_support_trial(
                fast_weights,
                trial_index,
                zero_relations=frozenset((relation,)),
            )
        loo_fields = readout_margin_fields(evaluator, fast_weights, geometry)
        influence = intact_fields - loo_fields
        field_rows.append(influence)
        gradient_rows.append(gradient_energy_fraction(influence, geometry))
        effects = classified_effects(
            influence, tuple(relation for _ in range(evaluator.config.bs)), geometry
        )
        for measure, classes in effects.items():
            for name, values in classes.items():
                effect_rows[measure][name].append(values)

        delta_potential = intact_potentials - hodge_potentials(loo_fields, geometry)
        denominator = np.sum(delta_potential * delta_potential, axis=1)
        third_items = np.asarray(
            [item for item in range(evaluator.protocol.n_items) if item not in relation]
        )
        third_delta = delta_potential[:, third_items]
        requested_numerator = np.sum(third_delta * third_delta, axis=1)
        relational_delta = third_delta - np.mean(third_delta, axis=1, keepdims=True)
        relational_numerator = np.sum(relational_delta * relational_delta, axis=1)
        valid = denominator > tolerance * tolerance
        requested_third_rows.append(
            np.divide(
                requested_numerator,
                denominator,
                out=np.full_like(denominator, np.nan),
                where=valid,
            )
        )
        relational_third_rows.append(
            np.divide(
                relational_numerator,
                denominator,
                out=np.full_like(denominator, np.nan),
                where=valid,
            )
        )
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

    retained = np.asarray(retained_rows)
    fields = np.asarray(field_rows)
    gradient = np.asarray(gradient_rows)
    requested_third = np.asarray(requested_third_rows)
    relational_third = np.asarray(relational_third_rows)
    effects_array = {
        measure: {name: np.asarray(values) for name, values in classes.items()}
        for measure, classes in effect_rows.items()
    }
    aggregate = {"retained": {}, "stable_omitted": {}}
    for status, mask in (
        ("retained", retained),
        ("stable_omitted", ~retained),
    ):
        aggregate[status]["influence_gradient_energy_fraction"] = summarize_subjects(
            masked_column_mean(gradient, mask), counts, interval=interval
        )
        aggregate[status]["requested_R_third"] = summarize_subjects(
            masked_column_mean(requested_third, mask), counts, interval=interval
        )
        aggregate[status]["gauge_invariant_R_third_rel"] = summarize_subjects(
            masked_column_mean(relational_third, mask), counts, interval=interval
        )
        for measure, classes in effects_array.items():
            aggregate[status][measure] = {
                name: summarize_subjects(
                    masked_column_mean(values, mask), counts, interval=interval
                )
                for name, values in classes.items()
            }

    per_relation = []
    for relation_index, relation in enumerate(relations):
        row = {
            "relation": [int(relation[0]), int(relation[1])],
            "retained": {},
            "stable_omitted": {},
        }
        for status, status_mask in (
            ("retained", retained[relation_index]),
            ("stable_omitted", ~retained[relation_index]),
        ):
            row[status] = {
                "influence_gradient_energy_fraction": summarize_subjects(
                    np.where(status_mask, gradient[relation_index], np.nan),
                    counts,
                    interval=interval,
                ),
                "requested_R_third": summarize_subjects(
                    np.where(status_mask, requested_third[relation_index], np.nan),
                    counts,
                    interval=interval,
                ),
                "gauge_invariant_R_third_rel": summarize_subjects(
                    np.where(status_mask, relational_third[relation_index], np.nan),
                    counts,
                    interval=interval,
                ),
                "mean_absolute": {
                    name: summarize_subjects(
                        np.where(status_mask, values[relation_index], np.nan),
                        counts,
                        interval=interval,
                    )
                    for name, values in effects_array["mean_absolute"].items()
                },
            }
        per_relation.append(row)

    omitted_max = (
        float(
            np.max(
                np.abs(fields[np.broadcast_to((~retained)[..., None], fields.shape)])
            )
        )
        if np.any(~retained)
        else 0.0
    )
    raw = {
        "relations": [[int(first), int(second)] for first, second in relations],
        "retained": retained.astype(int).tolist(),
        "influence_gradient_energy_fraction": json_values(gradient),
        "requested_R_third": json_values(requested_third),
        "gauge_invariant_R_third_rel": json_values(relational_third),
        "effects": {
            measure: {name: json_values(values) for name, values in classes.items()}
            for measure, classes in effects_array.items()
        },
    }
    return {
        "aggregate": aggregate,
        "per_relation": per_relation,
        "stable_omitted_max_abs_pair_influence": omitted_max,
        "stable_omitted_within_tolerance": omitted_max <= tolerance,
    }, raw


def summarize_baselines(
    evaluator: FrozenFastWeightEvaluator,
    p0_fields: np.ndarray,
    intact_fields: np.ndarray,
    exact_final_potential: np.ndarray,
    geometry: CompleteGraphGeometry,
    counts: np.ndarray,
    *,
    interval: float,
) -> tuple[dict, dict]:
    zero_weights = evaluator.net.initialZeroPlasticWeights(evaluator.config.bs)
    write_off_weights = evaluator.learn_fast_weights(FastWeightIntervention.WRITE_OFF)
    alpha_zero_weights = evaluator.learn_fast_weights(FastWeightIntervention.ALPHA_ZERO)
    fields = {
        "P0": p0_fields,
        "intact_P32": intact_fields,
        "reset_after_support": readout_margin_fields(evaluator, zero_weights, geometry),
        "write_off": readout_margin_fields(evaluator, write_off_weights, geometry),
        "alpha_zero": readout_margin_fields(
            evaluator, alpha_zero_weights, geometry, alpha_zero=True
        ),
    }
    intact_potential = normalize_potentials(hodge_potentials(intact_fields, geometry))
    summaries = {}
    raw = {}
    subject_metrics = {}
    for name, field in fields.items():
        potential = hodge_potentials(field, geometry)
        normalized = normalize_potentials(potential)
        subject_metrics[name] = {
            "gradient_energy_fraction": gradient_energy_fraction(field, geometry),
            "commitment_strength": np.linalg.norm(potential, axis=1),
            "alignment_to_intact_final": potential_alignment(
                normalized, intact_potential
            )["cosine"],
            "alignment_to_exact_distributional": potential_alignment(
                normalized, exact_final_potential
            )["cosine"],
        }
        summaries[name] = {
            metric: summarize_subjects(values, counts, interval=interval)
            for metric, values in subject_metrics[name].items()
        }
        raw[name] = {
            metric: json_values(values)
            for metric, values in subject_metrics[name].items()
        }
    summaries["intact_minus_controls"] = {
        control: {
            metric: summarize_difference(
                subject_metrics["intact_P32"][metric],
                subject_metrics[control][metric],
                counts,
                interval=interval,
            )
            for metric in (
                "commitment_strength",
                "alignment_to_intact_final",
                "alignment_to_exact_distributional",
            )
        }
        for control in ("P0", "reset_after_support", "write_off", "alpha_zero")
    }
    return summaries, raw


def query_time_localization(
    evaluator: FrozenFastWeightEvaluator,
    states: dict[str, object],
    response_fields: dict[str, np.ndarray],
    geometry: CompleteGraphGeometry,
    counts: np.ndarray,
    *,
    interval: float,
    tolerance: float,
) -> tuple[dict, dict]:
    schedules = ordered_query_schedule(geometry, evaluator.config.bs)
    reproduction_tolerance = float(max(tolerance, 32.0 * np.finfo(np.float32).eps))
    output_direction = (
        (evaluator.net.h2o.weight[1] - evaluator.net.h2o.weight[0])
        .detach()
        .cpu()
        .numpy()
    )
    summaries = {}
    raw = {}
    for state_name, fast_weights in states.items():
        trajectories, logit_trajectories = (
            evaluator.readout_hidden_and_logit_trajectories(fast_weights, schedules)
        )
        hidden_fields = np.empty(
            (
                evaluator.config.bs,
                evaluator.config.triallen,
                len(geometry.pairs),
                evaluator.config.hs,
            ),
            dtype=np.float64,
        )
        projected_fields = np.empty(
            (
                evaluator.config.bs,
                evaluator.config.triallen,
                len(geometry.pairs),
            ),
            dtype=np.float64,
        )
        for subject, subject_rows in enumerate(trajectories):
            for pair_index, pair in enumerate(geometry.pairs):
                hidden_fields[subject, :, pair_index] = 0.5 * (
                    subject_rows[pair] - subject_rows[(pair[1], pair[0])]
                )
                projected_fields[subject, :, pair_index] = 0.5 * (
                    logit_trajectories[subject][pair]
                    - logit_trajectories[subject][(pair[1], pair[0])]
                )
        analytic_projection = np.einsum("bkeh,h->bke", hidden_fields, output_direction)
        analytic_projection_error = float(
            np.max(np.abs(analytic_projection - projected_fields))
        )
        hidden_fraction = vector_gradient_energy_fraction(hidden_fields, geometry)
        output_fraction = gradient_energy_fraction(projected_fields, geometry)
        response_error = float(
            np.max(
                np.abs(
                    projected_fields[:, NUMRESPONSESTEP] - response_fields[state_name]
                )
            )
        )
        if response_error > reproduction_tolerance:
            raise RuntimeError(
                "query trajectory does not reproduce registered response margin: "
                f"{response_error:.9g} > {reproduction_tolerance:.9g}"
            )
        summaries[state_name] = {
            "steps": [
                {
                    "step": step,
                    "hidden_vector_gradient_energy_fraction": summarize_subjects(
                        hidden_fraction[:, step], counts, interval=interval
                    ),
                    "output_projected_gradient_energy_fraction": summarize_subjects(
                        output_fraction[:, step], counts, interval=interval
                    ),
                    "output_minus_hidden_gradient_fraction": summarize_difference(
                        output_fraction[:, step],
                        hidden_fraction[:, step],
                        counts,
                        interval=interval,
                    ),
                }
                for step in range(evaluator.config.triallen)
            ],
            "response_minus_first_hidden_gradient_fraction": summarize_difference(
                hidden_fraction[:, NUMRESPONSESTEP],
                hidden_fraction[:, 0],
                counts,
                interval=interval,
            ),
            "response_projection_max_abs_margin_error": response_error,
            "response_reproduction_tolerance": reproduction_tolerance,
            "scientific_zero_effect_tolerance": tolerance,
            "independent_numpy_projection_max_abs_error": (analytic_projection_error),
        }
        raw[state_name] = {
            "hidden_vector_gradient_energy_fraction": json_values(hidden_fraction),
            "output_projected_gradient_energy_fraction": json_values(output_fraction),
        }
    return summaries, raw


def seed_directional_diagnosis(
    prefix_summary: dict,
    branch_summary: dict,
    loo_summary: dict,
    baseline_summary: dict,
    query_summary: dict,
    *,
    tolerance: float,
) -> dict:
    distributional = prefix_summary["final_distributional_minus_map_alignment"][
        "cosine"
    ]
    branch_remote = branch_summary["aggregate"]["retained"]["mean_absolute"]["remote"]
    loo_remote = loo_summary["aggregate"]["retained"]["mean_absolute"]["remote"]
    loo_relational = loo_summary["aggregate"]["retained"]["gauge_invariant_R_third_rel"]
    content_controls = baseline_summary["intact_minus_controls"]
    content_pass = all(
        content_controls[control]["alignment_to_exact_distributional"]["bootstrap"][
            "lower"
        ]
        > 0.0
        for control in ("reset_after_support", "write_off", "alpha_zero")
    )
    query_final = query_summary["intact_P32"]
    recurrent = query_final["response_minus_first_hidden_gradient_fraction"]
    response_step = query_final["steps"][NUMRESPONSESTEP]
    readout = response_step["output_minus_hidden_gradient_fraction"]
    return {
        "distributional_projection_over_map": (
            distributional["bootstrap"]["lower"] > 0.0
        ),
        "immediate_remote_propagation": (
            branch_remote["bootstrap"]["lower"] > 0.0
            and branch_summary["stable_omitted_max_abs_effect"] <= tolerance
        ),
        "episode_specific_global_reassembly": (
            loo_remote["bootstrap"]["lower"] > 0.0
            and loo_relational["bootstrap"]["lower"] > 0.0
            and loo_summary["stable_omitted_max_abs_pair_influence"] <= tolerance
            and content_pass
        ),
        "recurrent_query_commitment": recurrent["bootstrap"]["lower"] > 0.0,
        "output_readout_selects_more_additive_component_at_response": (
            readout["bootstrap"]["lower"] > 0.0
        ),
        "floating_zero_tolerance": tolerance,
        "nonzero_strength_boundary": (
            "Positive intervals establish causal existence, not practical dominance; "
            "absolute effects and direct-to-remote contrasts remain part of the result."
        ),
    }


def run_assembly_trajectory(
    specification_path: Path = DEFAULT_SPECIFICATION_PATH,
) -> dict:
    specification = load_json(specification_path)
    validation = validate_registered_sources(specification)
    sources = specification["registered_sources"]
    pilot_specification = load_json(
        resolve_path(sources["pilot_specification"]["path"])
    )
    protocol = load_ranking_protocol(resolve_path(sources["protocol"]["path"]))
    geometry = build_complete_graph_geometry(protocol)
    bootstrap = specification["bootstrap"]
    interval = float(bootstrap["interval"])
    tolerance = float(specification["execution_contract"]["floating_zero_tolerance"])
    rng = np.random.default_rng(int(bootstrap["seed"]))

    per_seed = {}
    reference_evidence = None
    reference_exact = None
    counts = None
    for registration in sources["pilot_artifacts"]:
        seed = int(registration["seed"])
        print(f"[assembly-trajectory] loading frozen seed {seed}", file=sys.stderr)
        evaluator, behavior = load_frozen_evaluator(
            registration, pilot_specification, protocol
        )
        evidence = evaluator.realized_support_evidence()
        if reference_evidence is None:
            reference_evidence = evidence
            print(
                "[assembly-trajectory] computing exact prefix posteriors",
                file=sys.stderr,
            )
            reference_exact = exact_prefix_trajectory(
                evaluator,
                protocol,
                geometry,
                temperature=float(
                    pilot_specification["evaluation"]["posterior_temperature"]
                ),
            )
            counts = bootstrap_counts(
                rng,
                int(bootstrap["samples"]),
                evaluator.config.bs,
            )
        elif evidence != reference_evidence:
            raise RuntimeError("pilot seeds used different realized support evidence")
        assert reference_exact is not None
        assert counts is not None

        print(f"[assembly-trajectory] seed {seed}: prefix branches", file=sys.stderr)
        intact_fast_weights, prefix_fields, branch = run_prefix_branches(
            evaluator, geometry, tolerance=tolerance
        )
        prefix_summary, prefix_raw = summarize_prefixes(
            prefix_fields,
            reference_exact,
            geometry,
            counts,
            interval=interval,
        )
        branch_summary, branch_raw = summarize_matched_branches(
            branch,
            counts,
            interval=interval,
            tolerance=tolerance,
        )

        print(f"[assembly-trajectory] seed {seed}: relation LOO", file=sys.stderr)
        loo_summary, loo_raw = run_leave_one_relation_out(
            evaluator,
            prefix_fields[-1],
            geometry,
            counts,
            interval=interval,
            tolerance=tolerance,
        )
        print(f"[assembly-trajectory] seed {seed}: baselines", file=sys.stderr)
        baseline_summary, baseline_raw = summarize_baselines(
            evaluator,
            prefix_fields[0],
            prefix_fields[-1],
            reference_exact.distributional_potentials[-1],
            geometry,
            counts,
            interval=interval,
        )
        print(f"[assembly-trajectory] seed {seed}: query-time fields", file=sys.stderr)
        p0_fast_weights = evaluator.initialize_fast_weights()
        query_summary, query_raw = query_time_localization(
            evaluator,
            {"P0": p0_fast_weights, "intact_P32": intact_fast_weights},
            {"P0": prefix_fields[0], "intact_P32": prefix_fields[-1]},
            geometry,
            counts,
            interval=interval,
            tolerance=tolerance,
        )
        diagnosis = seed_directional_diagnosis(
            prefix_summary,
            branch_summary,
            loo_summary,
            baseline_summary,
            query_summary,
            tolerance=tolerance,
        )
        per_seed[str(seed)] = {
            "seed": seed,
            "subjects": evaluator.config.bs,
            "checkpoint": behavior["checkpoint"],
            "prefix_trajectory": prefix_summary,
            "matched_zero_evidence_branches": branch_summary,
            "leave_one_relation_out": loo_summary,
            "baselines": baseline_summary,
            "query_time_localization": query_summary,
            "directional_diagnosis": diagnosis,
            "raw_subject_level": {
                "prefix_trajectory": prefix_raw,
                "matched_zero_evidence_branches": branch_raw,
                "leave_one_relation_out": loo_raw,
                "baselines": baseline_raw,
                "query_time_localization": query_raw,
            },
        }

    diagnosis_names = tuple(
        name
        for name, value in next(iter(per_seed.values()))[
            "directional_diagnosis"
        ].items()
        if isinstance(value, bool)
    )
    overall = {
        f"{name}_replicated_across_pilot_seeds": all(
            row["directional_diagnosis"][name] for row in per_seed.values()
        )
        for name in diagnosis_names
    }
    overall["formal_confirmation_status"] = (
        "deferred; frozen formal contract and seeds remain untouched"
    )
    overall["next_step_rule"] = specification["reporting"]["next_step_rule"]

    return {
        "schema_version": 1,
        "diagnostic_id": specification["diagnostic_id"],
        "registration_status": specification["registration_status"],
        "registration_parent_commit": specification["registration_parent_commit"],
        "claim_boundary": specification["claim_boundary"],
        "device": {"neural_evaluation": DEVICE, "exact_posterior": "cpu_numpy"},
        "artifact_validation": validation,
        "execution_contract": specification["execution_contract"],
        "field_contract": specification["field_contract"],
        "bootstrap": bootstrap,
        "pilot_seeds": per_seed,
        "overall_diagnosis": overall,
    }


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Run registered frozen support-prefix and relation-LOO diagnostics."
    )
    parser.add_argument(
        "--specification", type=Path, default=DEFAULT_SPECIFICATION_PATH
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args(args)


def main(args=None):
    parsed = parse_args(args)
    result = run_assembly_trajectory(parsed.specification)
    parsed.output.parent.mkdir(parents=True, exist_ok=True)
    with parsed.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
