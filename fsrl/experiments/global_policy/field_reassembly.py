"""Frozen read-only global-policy field-reassembly diagnostic."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from fsrl.analysis.hodge import build_complete_graph_geometry, hodge_potentials
from fsrl.analysis.policy import exact_probability
from fsrl.analysis.statistics import bootstrap_counts
from fsrl.evaluation.fields import readout_margin_fields
from fsrl.evaluation.frozen_fast_weight import (
    FastWeightIntervention,
    FrozenFastWeightEvaluator,
    load_frozen_retro_checkpoint,
)
from fsrl.experiments.global_policy.amplitude_provenance import (
    NonInterpretableEstimate,
    interval_summary,
    posterior_descriptors,
)
from fsrl.experiments.global_policy.slope_localization import subject_slopes
from fsrl.experiments.local_fidelity.evidence_access_confirmation import (
    validate_artifacts,
)
from fsrl.infra.formal_runtime import require_formal_runtime
from fsrl.infra.provenance import load_json, tensor_hashes, write_json_exclusive
from fsrl.infra.study_registry import (
    registered_file_sha256,
    resolve_record,
    resolve_registered_path,
)
from fsrl.paths import REPO_ROOT
from fsrl.tasks.protocol import load_ranking_protocol, symbolic_distances

ROOT = REPO_ROOT
EstimandBundle = dict[str, Any]
DEFAULT_SPECIFICATION_PATH = resolve_record(
    "benchmarks/global_policy_field_reassembly_v1.json"
)
DEFAULT_IMPLEMENTATION_LOCK_PATH = resolve_record(
    "benchmarks/global_policy_field_reassembly_v1.lock.json"
)
DEFAULT_RESULT_PATH = resolve_record("results/global_policy_field_reassembly_v1.json")
CONFIRMATION_OUTPUT_ROOT = (
    ROOT / "artifacts" / "runs" / "dual-evidence-access-confirmation-v2-4"
)

PRIMARY_ESTIMANDS = (
    "S_NN",
    "S_PN",
    "S_NP",
    "S_PP",
    "D",
    "A",
    "R",
    "I",
    "Delta_A",
    "C_A",
    "Delta_R",
    "C_R",
    "S_tildePN",
    "Q_shape",
    "C_shape",
    "Q_amp",
)
DECISION_CONTRASTS = (
    "D",
    "A",
    "R",
    "I",
    "Delta_A",
    "C_A",
    "Delta_R",
    "C_R",
    "Q_shape",
    "C_shape",
    "Q_amp",
)
NORM_TOLERANCE = 1e-12


def _subject_max_abs(values: np.ndarray) -> np.ndarray:
    rows = np.asarray(values, dtype=np.float64)
    if rows.ndim < 2:
        raise ValueError(
            "a subject axis followed by one or more value axes is required"
        )
    return np.max(np.abs(rows), axis=tuple(range(1, rows.ndim)))


def decompose_field(field: np.ndarray, geometry) -> dict[str, np.ndarray]:
    """Return the zero-sum Hodge potential, gradient, and residual per subject."""

    values = np.asarray(field, dtype=np.float64)
    edges = len(geometry.pairs)
    if values.ndim != 2 or values.shape[1] != edges:
        raise ValueError("field must have shape (subjects, complete-graph edges)")
    if not np.all(np.isfinite(values)):
        raise NonInterpretableEstimate("field rows must all remain finite")
    potential = hodge_potentials(values, geometry)
    gradient = potential @ geometry.incidence.T
    residual = values - gradient
    return {
        "potential": potential,
        "gradient": gradient,
        "residual": residual,
        "reconstruction_error": _subject_max_abs(values - gradient - residual),
        "zero_sum_gauge_error": np.abs(np.sum(potential, axis=1)),
        "residual_orthogonality_error": _subject_max_abs(residual @ geometry.incidence),
    }


def _factorial_identity_errors(values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Evaluate every registered dependent contrast identity row by row."""

    return {
        "D_equals_A_plus_R": np.abs(values["D"] - values["A"] - values["R"]),
        "D_equals_Delta_A_plus_C_A": np.abs(
            values["D"] - values["Delta_A"] - values["C_A"]
        ),
        "D_equals_Delta_R_plus_C_R": np.abs(
            values["D"] - values["Delta_R"] - values["C_R"]
        ),
        "D_equals_Q_shape_plus_C_shape": np.abs(
            values["D"] - values["Q_shape"] - values["C_shape"]
        ),
        "I_equals_Delta_A_minus_C_R": np.abs(
            values["I"] - values["Delta_A"] + values["C_R"]
        ),
        "I_equals_Delta_R_minus_C_A": np.abs(
            values["I"] - values["Delta_R"] + values["C_A"]
        ),
        "Delta_A_equals_A_plus_half_I": np.abs(
            values["Delta_A"] - values["A"] - 0.5 * values["I"]
        ),
        "C_A_equals_R_minus_half_I": np.abs(
            values["C_A"] - values["R"] + 0.5 * values["I"]
        ),
        "Delta_R_equals_R_plus_half_I": np.abs(
            values["Delta_R"] - values["R"] - 0.5 * values["I"]
        ),
        "C_R_equals_A_minus_half_I": np.abs(
            values["C_R"] - values["A"] + 0.5 * values["I"]
        ),
        "Delta_A_equals_Q_shape_plus_Q_amp": np.abs(
            values["Delta_A"] - values["Q_shape"] - values["Q_amp"]
        ),
        "C_shape_equals_C_A_plus_Q_amp": np.abs(
            values["C_shape"] - values["C_A"] - values["Q_amp"]
        ),
    }


def field_reassembly_estimands(
    neural_margin: np.ndarray,
    posterior_margin: np.ndarray,
    geometry,
    distances: np.ndarray,
    nonlearned_mask: np.ndarray,
    choice_temperature: float = 0.25,
) -> EstimandBundle:
    """Construct the frozen factorial and norm-matched participant estimands."""

    neural = np.asarray(neural_margin, dtype=np.float64)
    posterior = np.asarray(posterior_margin, dtype=np.float64)
    edge_distances = np.asarray(distances, dtype=np.float64)
    selected = np.asarray(nonlearned_mask, dtype=bool)
    expected_shape = (
        (neural.shape[0], len(geometry.pairs)) if neural.ndim == 2 else None
    )
    if expected_shape is None or neural.shape != expected_shape:
        raise ValueError(
            "neural_margin must have shape (subjects, complete-graph edges)"
        )
    if posterior.shape != neural.shape:
        raise ValueError("posterior_margin must match neural_margin")
    if edge_distances.shape != (neural.shape[1],):
        raise ValueError("distances must have one value per complete-graph edge")
    if selected.shape != (neural.shape[1],):
        raise ValueError("nonlearned_mask must have one value per complete-graph edge")
    if choice_temperature <= 0.0:
        raise ValueError("choice_temperature must be positive")
    if not np.all(np.isfinite(neural)) or not np.all(np.isfinite(posterior)):
        raise NonInterpretableEstimate("margin rows must all remain finite")

    decomposition_n = decompose_field(neural, geometry)
    decomposition_p = decompose_field(posterior, geometry)
    g_n = decomposition_n["gradient"]
    c_n = decomposition_n["residual"]
    g_p = decomposition_p["gradient"]
    c_p = decomposition_p["residual"]
    norm_g_n = np.linalg.norm(g_n, axis=1)
    norm_g_p = np.linalg.norm(g_p, axis=1)
    zero_denominators = {
        "norm_g_N": np.flatnonzero(norm_g_n <= NORM_TOLERANCE).astype(int),
        "norm_g_P": np.flatnonzero(norm_g_p <= NORM_TOLERANCE).astype(int),
    }
    zero_denominators = {
        name: indices for name, indices in zero_denominators.items() if len(indices)
    }
    if zero_denominators:
        detail = {name: rows.tolist() for name, rows in zero_denominators.items()}
        raise NonInterpretableEstimate(
            f"registered additive norm at or below 1e-12: {detail}"
        )

    scale = norm_g_n / norm_g_p
    g_p_tilde = scale[:, None] * g_p
    decomposition_p_tilde = decompose_field(g_p_tilde, geometry)
    fields = {
        "NN": g_n + c_n,
        "PN": g_p + c_n,
        "NP": g_n + c_p,
        "PP": g_p + c_p,
        "tildePN": g_p_tilde + c_n,
    }
    correct_sign = np.asarray(geometry.true_sign, dtype=np.float64)[None, :]
    probabilities = {
        name: exact_probability(correct_sign * field, choice_temperature)
        for name, field in fields.items()
    }
    slopes = {
        name: subject_slopes(probability, edge_distances, selected)
        for name, probability in probabilities.items()
    }

    s_nn = slopes["NN"]
    s_pn = slopes["PN"]
    s_np = slopes["NP"]
    s_pp = slopes["PP"]
    s_tilde = slopes["tildePN"]
    delta_a = s_nn - s_pn
    c_a = s_pn - s_pp
    delta_r = s_nn - s_np
    c_r = s_np - s_pp
    estimands = {
        "S_NN": s_nn,
        "S_PN": s_pn,
        "S_NP": s_np,
        "S_PP": s_pp,
        "D": s_nn - s_pp,
        "A": 0.5 * (delta_a + c_r),
        "R": 0.5 * (delta_r + c_a),
        "I": delta_a - c_r,
        "Delta_A": delta_a,
        "C_A": c_a,
        "Delta_R": delta_r,
        "C_R": c_r,
        "S_tildePN": s_tilde,
        "Q_shape": s_nn - s_tilde,
        "C_shape": s_tilde - s_pp,
        "Q_amp": s_tilde - s_pn,
    }
    margin_slopes = {
        name: subject_slopes(correct_sign * field, edge_distances, selected)
        for name, field in fields.items()
        if name != "tildePN"
    }
    margin_i = (
        margin_slopes["NN"]
        - margin_slopes["PN"]
        - margin_slopes["NP"]
        + margin_slopes["PP"]
    )
    margin_field_i = fields["NN"] - fields["PN"] - fields["NP"] + fields["PP"]
    norm_g_p_tilde = np.linalg.norm(g_p_tilde, axis=1)
    energy_nn = np.sum(fields["NN"] * fields["NN"], axis=1)
    energy_tilde = np.sum(fields["tildePN"] * fields["tildePN"], axis=1)
    n_items = geometry.incidence.shape[1]

    diagnostics = {
        "norm_g_N": norm_g_n,
        "norm_g_P": norm_g_p,
        "posterior_to_neural_scale_k": scale,
        "norm_c_N": np.linalg.norm(c_n, axis=1),
        "norm_c_P": np.linalg.norm(c_p, axis=1),
        "norm_g_P_tilde": norm_g_p_tilde,
        "a_N_bridge": norm_g_n / np.sqrt(n_items),
        "a_post_bridge": norm_g_p / np.sqrt(n_items),
        "Y_margin": np.log(scale),
        "Y_bridge": np.log(scale),
        "neural_field_reconstruction_error": decomposition_n["reconstruction_error"],
        "posterior_field_reconstruction_error": decomposition_p["reconstruction_error"],
        "neural_zero_sum_gauge_error": decomposition_n["zero_sum_gauge_error"],
        "posterior_zero_sum_gauge_error": decomposition_p["zero_sum_gauge_error"],
        "neural_residual_orthogonality_error": decomposition_n[
            "residual_orthogonality_error"
        ],
        "posterior_residual_orthogonality_error": decomposition_p[
            "residual_orthogonality_error"
        ],
        "norm_match_norm_error": np.abs(norm_g_p_tilde - norm_g_n),
        "norm_match_scale_reconstruction_error": _subject_max_abs(
            g_p_tilde - scale[:, None] * g_p
        ),
        "norm_match_gradient_reconstruction_error": _subject_max_abs(
            decomposition_p_tilde["gradient"] - g_p_tilde
        ),
        "norm_match_zero_sum_gauge_error": decomposition_p_tilde[
            "zero_sum_gauge_error"
        ],
        "norm_match_gradient_residual_error": np.linalg.norm(
            decomposition_p_tilde["residual"], axis=1
        ),
        "norm_match_field_reconstruction_error": _subject_max_abs(
            fields["tildePN"] - g_p_tilde - c_n
        ),
        "norm_match_residual_orthogonality_error": _subject_max_abs(
            c_n @ geometry.incidence
        ),
        "norm_match_energy_error": np.abs(energy_tilde - energy_nn),
        "NN_probability_reconstruction_error": _subject_max_abs(
            probabilities["NN"]
            - exact_probability(correct_sign * neural, choice_temperature)
        ),
        "PP_probability_reconstruction_error": _subject_max_abs(
            probabilities["PP"]
            - exact_probability(correct_sign * posterior, choice_temperature)
        ),
        "margin_field_interaction_error": _subject_max_abs(margin_field_i),
        "margin_I": margin_i,
    }
    identity_errors = _factorial_identity_errors(estimands)
    return {
        **estimands,
        **diagnostics,
        "fields": fields,
        "probabilities": probabilities,
        "decompositions": {
            "N": decomposition_n,
            "P": decomposition_p,
            "P_tilde": decomposition_p_tilde,
        },
        "margin_slopes": margin_slopes,
        "factorial_identity_errors": identity_errors,
    }


def summarize_estimand(values: np.ndarray, counts: np.ndarray) -> dict:
    """Summarize one participant vector without dropping any nonfinite draw."""

    rows = np.asarray(values, dtype=np.float64)
    bootstrap = np.asarray(counts, dtype=np.float64)
    if rows.ndim != 1 or bootstrap.ndim != 2 or bootstrap.shape[1] != len(rows):
        raise ValueError(
            "one scalar per participant and a matching count matrix required"
        )
    if not np.all(np.isfinite(rows)):
        raise NonInterpretableEstimate("participant estimands must all remain finite")
    denominator = np.sum(bootstrap, axis=1)
    samples = np.divide(
        bootstrap @ rows,
        denominator,
        out=np.full(len(bootstrap), np.nan, dtype=np.float64),
        where=denominator > 0.0,
    )
    return interval_summary(float(np.mean(rows)), samples)


def classify_status(summary: dict, equivalence_margin: float = 0.005) -> str:
    """Apply the frozen mutually exclusive material/equivalence rule."""

    if equivalence_margin < 0.0:
        raise ValueError("equivalence_margin must be nonnegative")
    interval = summary["bootstrap"]
    if float(interval["lower95"]) > equivalence_margin:
        return "material_positive"
    if float(interval["upper95"]) < -equivalence_margin:
        return "material_negative"
    if (
        float(interval["lower90"]) >= -equivalence_margin
        and float(interval["upper90"]) <= equivalence_margin
    ):
        return "equivalent"
    return "unresolved"


def _descriptive_direction(summary: dict) -> str:
    interval = summary["bootstrap"]
    if float(interval["lower95"]) > 0.0:
        return "positive"
    if float(interval["upper95"]) < 0.0:
        return "negative"
    return "unresolved"


def _cross_network_status(seeds: dict[str, dict], name: str) -> dict:
    by_seed = {seed: row["statistics"]["statuses"][name] for seed, row in seeds.items()}
    statuses = set(by_seed.values())
    return {
        "status": next(iter(statuses))
        if len(statuses) == 1
        else "heterogeneous_or_unresolved",
        "by_seed": by_seed,
    }


def _gated_decision(outcome: str) -> dict:
    return {
        "outcome": outcome,
        "field_source_fingerprint": "not_evaluated",
        "main_effect_descriptions": "not_evaluated",
        "interaction_axis": "not_evaluated",
        "norm_matched_shape_axis": "not_evaluated",
        "network_population_inference": "not_performed",
    }


def cross_seed_decision(
    seeds: dict[str, dict], equivalence_margin: float = 0.005
) -> dict:
    """Apply all registered gates separately in each network, without pooling."""

    del equivalence_margin  # Statuses are frozen within each seed before this gate.
    rows = tuple(seeds.values())
    if set(seeds) != {"2104", "2105"} or not all(
        row["integrity"]["passed"] for row in rows
    ):
        return _gated_decision("noninterpretable_integrity_failure")
    if not all(
        float(row["statistics"]["summaries"]["D"]["bootstrap"]["lower95"]) > 0.0
        for row in rows
    ):
        return _gated_decision("premise_not_confirmed")

    def seed_rules(row: dict) -> dict[str, bool]:
        statuses = row["statistics"]["statuses"]
        additive = (
            statuses["Delta_A"] == "material_positive"
            and statuses["C_A"] == "equivalent"
        )
        residual = (
            statuses["Delta_R"] == "material_positive"
            and statuses["C_R"] == "equivalent"
        )
        both_material = all(
            statuses[name] == "material_positive"
            for name in ("Delta_A", "C_A", "Delta_R", "C_R")
        )
        return {
            "additive": additive,
            "residual": residual,
            "both_components_material": both_material,
        }

    rules_by_seed = {seed: seed_rules(row) for seed, row in seeds.items()}

    def seed_fingerprint(rules: dict[str, bool]) -> str:
        if rules["additive"] and rules["residual"]:
            return "both_replacements_sufficient"
        if rules["additive"]:
            return "additive_replacement_only"
        if rules["residual"]:
            return "residual_replacement_only"
        if rules["both_components_material"]:
            return "both_components_material"
        return "mixed_or_unresolved"

    fingerprint_by_seed = {
        seed: seed_fingerprint(rules) for seed, rules in rules_by_seed.items()
    }
    unique_fingerprints = set(fingerprint_by_seed.values())
    fingerprint = (
        next(iter(unique_fingerprints))
        if len(unique_fingerprints) == 1
        else "mixed_or_unresolved"
    )

    def replicated_status(name: str, required: str) -> bool:
        return all(row["statistics"]["statuses"][name] == required for row in rows)

    q_shape = _cross_network_status(seeds, "Q_shape")
    c_shape = _cross_network_status(seeds, "C_shape")
    q_amp = _cross_network_status(seeds, "Q_amp")
    shape_sufficient = replicated_status(
        "Q_shape", "material_positive"
    ) and replicated_status("C_shape", "equivalent")
    return {
        "outcome": fingerprint,
        "field_source_fingerprint": fingerprint,
        "field_source_fingerprint_by_seed": fingerprint_by_seed,
        "replacement_sufficiency": {
            "by_seed": rules_by_seed,
            "additive": all(rules["additive"] for rules in rules_by_seed.values()),
            "residual": all(rules["residual"] for rules in rules_by_seed.values()),
            "both_components_material": all(
                rules["both_components_material"] for rules in rules_by_seed.values()
            ),
        },
        "main_effect_descriptions": {
            "A": _cross_network_status(seeds, "A"),
            "R": _cross_network_status(seeds, "R"),
        },
        "interaction_axis": _cross_network_status(seeds, "I"),
        "norm_matched_shape_axis": {
            "outcome": (
                "norm_matched_shape_replacement_sufficient"
                if shape_sufficient
                else "mixed_or_unresolved"
            ),
            "Q_shape": q_shape,
            "C_shape": c_shape,
            "Q_amp": q_amp,
        },
        "network_population_inference": "not_performed",
    }


def _source_validation(
    specification: dict,
    specification_path: Path,
    implementation_lock: dict,
) -> dict:
    registrations = {
        **specification["registered_sources"],
        "diagnostic_specification": {
            "path": specification_path,
            "sha256": implementation_lock["diagnostic_specification_sha256"],
        },
        **implementation_lock["implementation_sources"],
    }
    checks = []
    for name, registration in registrations.items():
        path = resolve_registered_path(registration["path"])
        observed = registered_file_sha256(
            registration["path"], registration["sha256"], resolved_path=path
        )
        checks.append(
            {
                "name": name,
                "path": str(path.relative_to(ROOT)),
                "observed": observed,
                "expected": registration["sha256"],
                "passed": observed == registration["sha256"],
            }
        )
    if not all(row["passed"] for row in checks):
        raise RuntimeError(
            f"global-policy field-reassembly source lock failed: {checks}"
        )
    return {"passed": True, "checks": checks}


def _artifact_validation(specification: dict) -> dict:
    sources = specification["registered_sources"]
    confirmation_specification_path = resolve_registered_path(
        sources["v2_4_confirmation_specification"]["path"]
    )
    confirmation_specification = load_json(confirmation_specification_path)
    return validate_artifacts(
        confirmation_specification,
        confirmation_specification_path,
        resolve_registered_path(
            sources["v2_4_confirmation_implementation_lock"]["path"]
        ),
        resolve_registered_path(sources["v2_4_confirmation_artifact_lock"]["path"]),
        CONFIRMATION_OUTPUT_ROOT,
    )


def _bootstrap_identity_errors(
    estimands: EstimandBundle, counts: np.ndarray
) -> dict[str, float]:
    denominators = np.sum(counts, axis=1)
    samples = {
        name: counts @ np.asarray(estimands[name], dtype=np.float64) / denominators
        for name in PRIMARY_ESTIMANDS
    }
    return {
        name: float(np.max(error))
        for name, error in _factorial_identity_errors(samples).items()
    }


def _seed_statistics(specification: dict, seed: int, estimands: EstimandBundle) -> dict:
    contract = specification["statistical_estimands"]
    bootstrap = contract["bootstrap"]
    counts = bootstrap_counts(
        np.random.default_rng(int(bootstrap["seeds"][str(seed)])),
        int(bootstrap["samples"]),
        len(np.asarray(estimands["D"])),
    )
    summaries = {
        name: summarize_estimand(np.asarray(estimands[name]), counts)
        for name in PRIMARY_ESTIMANDS
    }
    margin = float(contract["equivalence_margin"])
    statuses = {
        name: classify_status(summaries[name], margin) for name in DECISION_CONTRASTS
    }
    descriptive = {
        name: _descriptive_direction(summaries[name]) for name in DECISION_CONTRASTS
    }
    subject_identity = {
        name: float(np.max(error))
        for name, error in estimands["factorial_identity_errors"].items()
    }
    bootstrap_identity = _bootstrap_identity_errors(estimands, counts)
    raw_names = (
        *PRIMARY_ESTIMANDS,
        "norm_g_N",
        "norm_g_P",
        "posterior_to_neural_scale_k",
        "norm_c_N",
        "norm_c_P",
        "norm_g_P_tilde",
        "a_N_bridge",
        "a_post_bridge",
        "Y_margin",
        "Y_bridge",
        "neural_field_reconstruction_error",
        "posterior_field_reconstruction_error",
        "neural_zero_sum_gauge_error",
        "posterior_zero_sum_gauge_error",
        "neural_residual_orthogonality_error",
        "posterior_residual_orthogonality_error",
        "norm_match_norm_error",
        "norm_match_scale_reconstruction_error",
        "norm_match_gradient_reconstruction_error",
        "norm_match_zero_sum_gauge_error",
        "norm_match_gradient_residual_error",
        "norm_match_field_reconstruction_error",
        "norm_match_residual_orthogonality_error",
        "norm_match_energy_error",
        "NN_probability_reconstruction_error",
        "PP_probability_reconstruction_error",
        "margin_field_interaction_error",
        "margin_I",
    )
    raw_subject_level = {
        name: np.asarray(estimands[name], dtype=np.float64).tolist()
        for name in raw_names
    }
    raw_subject_level.update(
        {
            f"factorial_{name}": np.asarray(error, dtype=np.float64).tolist()
            for name, error in estimands["factorial_identity_errors"].items()
        }
    )
    return {
        "summaries": summaries,
        "statuses": statuses,
        "descriptive_directions": descriptive,
        "raw_subject_level": raw_subject_level,
        "integrity": {
            "bootstrap_samples": int(counts.shape[0]),
            "bootstrap_subjects": int(counts.shape[1]),
            "all_bootstrap_estimates_finite": all(
                summary["bootstrap"]["finite_samples"] == counts.shape[0]
                for summary in summaries.values()
            ),
            "subject_factorial_identity_max_abs_errors": subject_identity,
            "bootstrap_factorial_identity_max_abs_errors": bootstrap_identity,
        },
    }


def analyze_seed(specification: dict, seed: int, artifact_validation: dict) -> dict:
    evaluation = specification["evaluation"]
    artifact = artifact_validation["lock"]["artifacts"][str(seed)]["checkpoint"]
    checkpoint_path = resolve_registered_path(artifact["path"])
    backbone, model_config, checkpoint = load_frozen_retro_checkpoint(
        checkpoint_path, int(evaluation["subjects"])
    )
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    before = tensor_hashes(backbone)
    protocol = load_ranking_protocol(
        resolve_registered_path(
            specification["registered_sources"]["liu_protocol"]["path"]
        )
    )
    evaluator = FrozenFastWeightEvaluator(
        backbone,
        model_config,
        protocol,
        cue_seed=int(evaluation["cue_seed"]),
        support_seed=int(evaluation["support_seed"]),
        cue_mode=str(evaluation["cue_mode"]),
        subject_encoding_mode=str(evaluation["subject_encoding_mode"]),
        subject_encoding_seed=int(evaluation["subject_encoding_seed"]),
    )
    geometry = build_complete_graph_geometry(protocol)
    distances = symbolic_distances(protocol, geometry.pairs)
    nonlearned = np.asarray(
        [pair not in protocol.learned_pairs for pair in geometry.pairs], dtype=bool
    )
    fast_weights = evaluator.learn_fast_weights(FastWeightIntervention.INTACT)
    neural_margin = readout_margin_fields(evaluator, fast_weights, geometry)
    posterior, posterior_integrity = posterior_descriptors(
        evaluator, geometry, specification
    )
    posterior_margin = posterior["fields"]["same_unit_margin"]
    temperature = float(specification["posterior_comparator"]["choice_temperature"])
    estimands = field_reassembly_estimands(
        neural_margin,
        posterior_margin,
        geometry,
        distances,
        nonlearned,
        temperature,
    )
    statistics = _seed_statistics(specification, seed, estimands)

    track_b = load_json(
        resolve_registered_path(
            specification["registered_sources"]["slope_localization_result"]["path"]
        )
    )["seeds"][str(seed)]["raw_subject_level"]
    amplitude = load_json(
        resolve_registered_path(
            specification["registered_sources"]["amplitude_provenance_result"]["path"]
        )
    )["seeds"][str(seed)]["statistics"]["raw_subject_level"]
    frozen_a_n_from_margin = np.asarray(amplitude["a_N_from_margin"], dtype=np.float64)
    frozen_a_post = np.asarray(amplitude["a_post"], dtype=np.float64)
    frozen_y_margin = np.log(frozen_a_n_from_margin / frozen_a_post)
    exact_references = {
        "S_NN": (
            np.asarray(estimands["S_NN"]),
            np.asarray(track_b["beta_p"], dtype=np.float64),
        ),
        "S_PP": (
            np.asarray(estimands["S_PP"]),
            np.asarray(track_b["beta_p_posterior"], dtype=np.float64),
        ),
        "D": (
            np.asarray(estimands["D"]),
            np.asarray(track_b["beta_p_minus_posterior"], dtype=np.float64),
        ),
        "Track_B_potential_amplitude": (
            np.asarray(estimands["a_N_bridge"]),
            np.asarray(track_b["potential_amplitude"], dtype=np.float64),
        ),
        "amplitude_a_N_from_margin": (
            np.asarray(estimands["a_N_bridge"]),
            frozen_a_n_from_margin,
        ),
        "amplitude_a_post": (
            np.asarray(estimands["a_post_bridge"]),
            frozen_a_post,
        ),
        "amplitude_Y_margin_internal": (
            np.asarray(estimands["Y_margin"]),
            frozen_y_margin,
        ),
    }
    anchor_errors = {
        f"{name}_reference_reproduction_max_abs_error": float(
            np.max(np.abs(observed - target))
        )
        for name, (observed, target) in exact_references.items()
    }
    historical_a_n_signed_difference = np.asarray(estimands["a_N_bridge"]) - np.asarray(
        amplitude["a_N"], dtype=np.float64
    )
    historical_y_signed_difference = np.asarray(estimands["Y_margin"]) - np.asarray(
        amplitude["Y"], dtype=np.float64
    )
    statistics["raw_subject_level"].update(
        {
            "historical_hidden_a_N_signed_difference": (
                historical_a_n_signed_difference.tolist()
            ),
            "historical_frozen_Y_signed_difference": (
                historical_y_signed_difference.tolist()
            ),
        }
    )
    historical_bridge_errors = {
        "historical_hidden_a_N_bridge_max_abs_error": float(
            np.max(np.abs(historical_a_n_signed_difference))
        ),
        "historical_frozen_Y_bridge_max_abs_error": float(
            np.max(np.abs(historical_y_signed_difference))
        ),
    }
    posterior_probability = 0.5 * (
        1.0
        + geometry.true_sign[None, :] * posterior["fields"]["pair_probability_field"]
    )
    posterior_probability_error = float(
        np.max(np.abs(estimands["probabilities"]["PP"] - posterior_probability))
    )
    after = tensor_hashes(backbone)

    float_tolerance = float(
        specification["integrity_gates"]["float64_field_and_probability_tolerance"]
    )
    track_tolerance = float(
        specification["integrity_gates"]["Track_B_raw_subject_reproduction_tolerance"]
    )
    algebra_error_names = (
        "neural_field_reconstruction_error",
        "posterior_field_reconstruction_error",
        "neural_zero_sum_gauge_error",
        "posterior_zero_sum_gauge_error",
        "neural_residual_orthogonality_error",
        "posterior_residual_orthogonality_error",
        "norm_match_norm_error",
        "norm_match_scale_reconstruction_error",
        "norm_match_gradient_reconstruction_error",
        "norm_match_zero_sum_gauge_error",
        "norm_match_gradient_residual_error",
        "norm_match_field_reconstruction_error",
        "norm_match_residual_orthogonality_error",
        "norm_match_energy_error",
        "NN_probability_reconstruction_error",
        "PP_probability_reconstruction_error",
        "margin_field_interaction_error",
        "margin_I",
    )
    algebra_errors = {
        f"{name}_max_abs_error": float(np.max(np.abs(estimands[name])))
        for name in algebra_error_names
    }
    identity_errors = {
        **statistics["integrity"]["subject_factorial_identity_max_abs_errors"],
        **{
            f"bootstrap_{name}": value
            for name, value in statistics["integrity"][
                "bootstrap_factorial_identity_max_abs_errors"
            ].items()
        },
    }
    integrity = {
        **posterior_integrity,
        **anchor_errors,
        **algebra_errors,
        **historical_bridge_errors,
        "posterior_choice_probability_reproduction_max_abs_error": (
            posterior_probability_error
        ),
        "factorial_identity_max_abs_errors": identity_errors,
        "minimum_neural_additive_norm": float(np.min(estimands["norm_g_N"])),
        "minimum_posterior_additive_norm": float(np.min(estimands["norm_g_P"])),
        "subjects": int(model_config.bs),
        "edges": len(geometry.pairs),
        "orientations": 2 * len(geometry.pairs),
        "nonlearned_pairs": int(np.sum(nonlearned)),
        "bootstrap_samples": statistics["integrity"]["bootstrap_samples"],
        "bootstrap_subjects": statistics["integrity"]["bootstrap_subjects"],
        "all_bootstrap_estimates_finite": statistics["integrity"][
            "all_bootstrap_estimates_finite"
        ],
        "backbone_tensor_hashes_unchanged": before == after,
    }
    exact_anchor_error_names = (
        "S_NN_reference_reproduction_max_abs_error",
        "S_PP_reference_reproduction_max_abs_error",
        "D_reference_reproduction_max_abs_error",
        "Track_B_potential_amplitude_reference_reproduction_max_abs_error",
        "amplitude_a_N_from_margin_reference_reproduction_max_abs_error",
        "amplitude_a_post_reference_reproduction_max_abs_error",
        "amplitude_Y_margin_internal_reference_reproduction_max_abs_error",
    )
    posterior_error_names = (
        "posterior_inverse_link_max_abs_error",
        "posterior_orientation_reversal_max_abs_error",
        "posterior_choice_probability_reproduction_max_abs_error",
    )
    gpu_tolerance = float(
        specification["integrity_gates"]["GPU_neural_margin_reproduction_tolerance"]
    )
    integrity["passed"] = bool(
        all(integrity[name] <= track_tolerance for name in exact_anchor_error_names)
        and all(value <= gpu_tolerance for value in historical_bridge_errors.values())
        and all(integrity[name] <= float_tolerance for name in algebra_errors)
        and all(value <= float_tolerance for value in identity_errors.values())
        and all(integrity[name] <= float_tolerance for name in posterior_error_names)
        and integrity["minimum_neural_additive_norm"] > NORM_TOLERANCE
        and integrity["minimum_posterior_additive_norm"] > NORM_TOLERANCE
        and integrity["subjects"] == int(evaluation["subjects"]) == 77
        and integrity["edges"] == 28
        and integrity["orientations"] == 56
        and integrity["nonlearned_pairs"] == 20
        and integrity["bootstrap_samples"]
        == int(specification["statistical_estimands"]["bootstrap"]["samples"])
        and integrity["bootstrap_subjects"] == 77
        and integrity["all_bootstrap_estimates_finite"]
        and integrity["backbone_tensor_hashes_unchanged"]
    )
    return {
        "seed": seed,
        "checkpoint": {"path": artifact["path"], "sha256": checkpoint.sha256},
        "condition": "pure_L_off_intact_P_T",
        "statistics": statistics,
        "integrity": integrity,
    }


def run_diagnostic(
    specification: dict,
    source_validation: dict,
    artifact_validation: dict,
    runtime: dict,
) -> dict:
    seeds = {
        str(seed): analyze_seed(specification, int(seed), artifact_validation)
        for seed in specification["network_contract"]["mandatory_frozen_seeds"]
    }
    return {
        "schema_version": 1,
        "diagnostic_id": specification["diagnostic_id"],
        "registration_status": specification["registration_status"],
        "claim_boundary": specification["claim_boundary"],
        "runtime": runtime,
        "source_validation": source_validation,
        "artifact_validation": artifact_validation,
        "primary_condition": "pure_L_off_intact_P_T",
        "seeds": seeds,
        "decision": cross_seed_decision(
            seeds,
            float(specification["statistical_estimands"]["equivalence_margin"]),
        ),
    }


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Run the frozen global-policy field-reassembly diagnostic."
    )
    parser.add_argument(
        "--specification", type=Path, default=DEFAULT_SPECIFICATION_PATH
    )
    parser.add_argument(
        "--implementation-lock", type=Path, default=DEFAULT_IMPLEMENTATION_LOCK_PATH
    )
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT_PATH)
    return parser.parse_args(args)


def main(args=None) -> int:
    parsed = parse_args(args)
    runtime = require_formal_runtime()
    specification = load_json(parsed.specification)
    implementation_lock = load_json(parsed.implementation_lock)
    source_validation = _source_validation(
        specification, parsed.specification, implementation_lock
    )
    artifact_validation = _artifact_validation(specification)
    result = run_diagnostic(
        specification, source_validation, artifact_validation, runtime
    )
    write_json_exclusive(parsed.result, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
