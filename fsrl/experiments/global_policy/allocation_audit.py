"""Read-only localization of the replicated global-policy allocation fingerprint."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from fsrl.analysis.hodge import build_complete_graph_geometry
from fsrl.analysis.policy import exact_probability
from fsrl.evaluation.fields import readout_margin_fields
from fsrl.evaluation.frozen_fast_weight import (
    FastWeightIntervention,
    FrozenFastWeightEvaluator,
    load_retro_checkpoint,
)
from fsrl.experiments.global_policy.amplitude_provenance import (
    NonInterpretableEstimate,
    interval_summary,
    posterior_descriptors,
)
from fsrl.experiments.global_policy.field_reassembly import field_reassembly_estimands
from fsrl.experiments.global_policy.field_replication import (
    ALGEBRA_ERROR_ARRAYS,
    mandatory_seeds,
    require_pushed_freeze,
    validate_artifacts,
)
from fsrl.infra.formal_runtime import require_formal_runtime
from fsrl.infra.provenance import load_json, tensor_hashes, write_json_exclusive
from fsrl.infra.study_registry import canonical_file_sha256 as file_sha256
from fsrl.infra.study_registry import (
    legacy_identifier,
    registered_file_sha256,
    resolve_record,
    resolve_registered_path,
)
from fsrl.paths import REPO_ROOT
from fsrl.tasks.protocol import symbolic_distances
from fsrl.tasks.registered_protocol import load_ranking_protocol

ROOT = REPO_ROOT
DEFAULT_SPECIFICATION_PATH = resolve_record(
    "benchmarks/global_policy_allocation_audit_v1.json"
)
INITIAL_IMPLEMENTATION_LOCK_PATH = resolve_record(
    "benchmarks/global_policy_allocation_audit_v1.lock.json"
)
DEFAULT_IMPLEMENTATION_LOCK_PATH = resolve_record(
    "benchmarks/global_policy_allocation_audit_v1.repair1.lock.json"
)
DEFAULT_RESULT_PATH = resolve_record("results/global_policy_allocation_audit_v1.json")
NONINTERPRETABLE_ATTEMPT_PATH = resolve_record(
    "results/global_policy_allocation_audit_v1_attempt1_noninterpretable.json"
)
UPSTREAM_SPECIFICATION_PATH = resolve_record(
    "benchmarks/global_policy_field_fingerprint_replication_v1.json"
)
UPSTREAM_IMPLEMENTATION_LOCK_PATH = resolve_record(
    "benchmarks/global_policy_field_fingerprint_replication_v1.lock.json"
)
UPSTREAM_ARTIFACT_LOCK_PATH = resolve_record(
    "benchmarks/global_policy_field_fingerprint_replication_v1.artifact_lock.json"
)
UPSTREAM_OUTPUT_ROOT = (
    ROOT / "artifacts" / "runs" / "global-policy-field-fingerprint-replication-v1"
)
UPSTREAM_RESULT_PATH = resolve_record(
    "results/global_policy_field_fingerprint_replication_v1.json"
)

TOLERANCE = 1e-10
RANK_TOLERANCE = 1e-10
REQUIRED_IMPLEMENTATION_SOURCE_PATHS = {
    "audit_runner": "fsrl/global_policy_allocation_audit.py",
    "audit_tests": "tests/test_global_policy_allocation_audit.py",
    "formal_runtime": "fsrl/formal_runtime.py",
    "formal_runtime_tests": "tests/test_formal_runtime.py",
}
REQUIRED_REUSED_SOURCE_PATHS = {
    "fingerprint_runner": "fsrl/global_policy_field_fingerprint_replication.py",
    "field_reassembly_estimator": "fsrl/global_policy_field_reassembly.py",
    "posterior_descriptor": "fsrl/global_policy_amplitude_provenance.py",
    "slope_estimator": "fsrl/global_policy_slope_localization.py",
    "assembly_source": "fsrl/assembly_trajectory.py",
    "frozen_evaluator": "fsrl/liu_eval.py",
    "exact_probability": "fsrl/local_behavior_attribution.py",
    "exact_posterior": "fsrl/constructive.py",
    "ranking_protocol": "fsrl/ranking_protocol.py",
    "configuration": "fsrl/config.py",
    "hash_validation": "fsrl/confirmation.py",
    "tensor_hashes": "fsrl/curvature_gate_pilot.py",
    "v1_training_source": "fsrl/meta_train.py",
    "v1_model_source": "fsrl/model.py",
    "generic_task_source": "fsrl/meta_tasks.py",
    "subject_encoding_source": "fsrl/subject_encoding.py",
    "qualification_source": "fsrl/qualification.py",
}
UPSTREAM_ONLY_REUSED_SOURCES = {
    "v1_training_source",
    "v1_model_source",
    "generic_task_source",
    "subject_encoding_source",
    "qualification_source",
}


def canonical_paths(parsed: argparse.Namespace) -> None:
    expected = {
        "specification": DEFAULT_SPECIFICATION_PATH,
        "implementation_lock": DEFAULT_IMPLEMENTATION_LOCK_PATH,
        "output_root": UPSTREAM_OUTPUT_ROOT,
    }
    for name, canonical in expected.items():
        if getattr(parsed, name).resolve() != canonical.resolve():
            raise RuntimeError(f"formal workflow requires canonical {name}")
    result = parsed.result.resolve()
    if result != DEFAULT_RESULT_PATH.resolve():
        try:
            relative = result.relative_to(Path("/tmp").resolve())
        except ValueError as error:
            raise RuntimeError(
                "formal replay result must be a file below /tmp"
            ) from error
        if not relative.parts:
            raise RuntimeError("formal replay result must be a file below /tmp")
    if parsed.result.exists() or parsed.result.is_symlink():
        raise RuntimeError("allocation result already exists or is a symlink")
    if parsed.result.parent.is_symlink():
        raise RuntimeError("allocation result parent may not be a symlink")


def required_freeze_paths(
    specification_path: Path, implementation_lock_path: Path
) -> tuple[Path, ...]:
    return (
        specification_path,
        implementation_lock_path,
        INITIAL_IMPLEMENTATION_LOCK_PATH,
        NONINTERPRETABLE_ATTEMPT_PATH,
    )


def validate_sources(
    specification_path: Path = DEFAULT_SPECIFICATION_PATH,
    implementation_lock_path: Path = DEFAULT_IMPLEMENTATION_LOCK_PATH,
) -> dict:
    """Validate the new implementation and every reused frozen source."""

    specification = load_json(specification_path)
    lock = load_json(implementation_lock_path)
    if (
        lock.get("schema_version") != 1
        or lock.get("audit_id") != specification.get("audit_id")
        or lock.get("freeze_status")
        != "repair1_frozen_after_noninterpretable_attempt1_and_before_reexecution"
        or lock.get("audit_specification_sha256") != file_sha256(specification_path)
    ):
        raise RuntimeError("allocation implementation lock identity mismatch")
    expected_supersedes = {
        "path": legacy_identifier(INITIAL_IMPLEMENTATION_LOCK_PATH),
        "sha256": file_sha256(INITIAL_IMPLEMENTATION_LOCK_PATH),
    }
    expected_attempt = {
        "path": legacy_identifier(NONINTERPRETABLE_ATTEMPT_PATH),
        "sha256": file_sha256(NONINTERPRETABLE_ATTEMPT_PATH),
    }
    if lock.get("supersedes") != expected_supersedes:
        raise RuntimeError("allocation repair lock does not supersede the initial lock")
    if lock.get("noninterpretable_attempt") != expected_attempt:
        raise RuntimeError("allocation repair lock does not bind attempt 1")
    attempt = load_json(NONINTERPRETABLE_ATTEMPT_PATH)
    if (
        attempt.get("audit_id") != specification.get("audit_id")
        or attempt.get("status") != "noninterpretable_execution_failure"
        or not lock.get("repair_scope")
    ):
        raise RuntimeError("allocation repair provenance mismatch")
    groups = (
        (lock.get("implementation_sources", {}), REQUIRED_IMPLEMENTATION_SOURCE_PATHS),
        (lock.get("reused_frozen_sources", {}), REQUIRED_REUSED_SOURCE_PATHS),
    )
    for registrations, required in groups:
        if set(registrations) != set(required):
            raise RuntimeError("allocation implementation lock source set mismatch")
        for name, path in required.items():
            if Path(registrations[name].get("path", "")) != Path(path):
                raise RuntimeError(f"allocation source path mismatch: {name}")
    upstream_specification = load_json(UPSTREAM_SPECIFICATION_PATH)
    for name in REQUIRED_REUSED_SOURCE_PATHS:
        registered = (
            upstream_specification["registered_sources"][name]
            if name in UPSTREAM_ONLY_REUSED_SOURCES
            else specification["registered_sources"][name]
        )
        if lock["reused_frozen_sources"][name] != registered:
            raise RuntimeError(
                f"allocation reused source differs from the registered source: {name}"
            )
    upstream = lock.get("upstream_fingerprint", {})
    required_upstream = {
        "specification": specification["registered_sources"][
            "fingerprint_specification"
        ],
        "implementation_lock": specification["registered_sources"][
            "fingerprint_implementation_lock"
        ],
        "artifact_lock": specification["registered_sources"][
            "fingerprint_artifact_lock"
        ],
        "result": specification["registered_sources"]["fingerprint_result"],
        "report": specification["registered_sources"]["fingerprint_report"],
        "output_root": str(UPSTREAM_OUTPUT_ROOT.relative_to(ROOT)),
        "checkpoint_sha256": specification["artifact_contract"]["checkpoints"],
    }
    if upstream != required_upstream:
        raise RuntimeError("allocation upstream fingerprint binding mismatch")
    registrations = {
        **specification["registered_sources"],
        "audit_specification": {
            "path": legacy_identifier(specification_path),
            "sha256": lock["audit_specification_sha256"],
        },
        "superseded_implementation_lock": expected_supersedes,
        "noninterpretable_attempt": expected_attempt,
        **lock["implementation_sources"],
        **lock["reused_frozen_sources"],
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
    if not all(check["passed"] for check in checks):
        raise RuntimeError(f"allocation source lock failed: {checks}")
    return {"passed": True, "checks": checks, "lock": lock}


def q_shape_rows_complete(values: np.ndarray, subjects: int) -> bool:
    return bool(values.shape == (subjects,) and np.all(np.isfinite(values)))


def validate_upstream(specification: dict) -> tuple[dict, dict, dict[str, np.ndarray]]:
    """Validate original artifacts and the frozen replication prerequisite."""

    upstream_specification = load_json(UPSTREAM_SPECIFICATION_PATH)
    artifact_validation = validate_artifacts(
        upstream_specification,
        UPSTREAM_SPECIFICATION_PATH,
        UPSTREAM_IMPLEMENTATION_LOCK_PATH,
        UPSTREAM_ARTIFACT_LOCK_PATH,
        UPSTREAM_OUTPUT_ROOT,
    )
    prior = load_json(UPSTREAM_RESULT_PATH)
    seeds = tuple(str(seed) for seed in mandatory_seeds(upstream_specification))
    checks = {
        "result_outcome": prior.get("decision", {}).get("outcome")
        == "replicated_field_fingerprint",
        "source_validation": prior.get("source_validation", {}).get("passed") is True,
        "artifact_validation": prior.get("artifact_validation", {}).get("passed")
        is True,
        "network_population_inference": prior.get("decision", {}).get(
            "network_population_inference"
        )
        == "not_performed",
        "seed_set": set(prior.get("seeds", {})) == set(seeds),
        "seed_gates": all(
            prior["seeds"][seed].get("qualification", {}).get("passed") is True
            and prior["seeds"][seed].get("integrity", {}).get("passed") is True
            for seed in seeds
        ),
        "checkpoint_contract": all(
            artifact_validation["lock"]["artifacts"][seed]["checkpoint"]["sha256"]
            == specification["artifact_contract"]["checkpoints"][seed]
            for seed in seeds
        ),
    }
    prior_q_shape = {}
    for seed in seeds:
        values = np.asarray(
            prior["seeds"][seed]["statistics"]["raw_subject_level"]["Q_shape"],
            dtype=np.float64,
        )
        prior_q_shape[seed] = values
        checks[f"seed_{seed}_q_shape_rows"] = q_shape_rows_complete(
            values, int(specification["evaluation"]["subjects"])
        )
    passed = bool(all(checks.values()))
    if not passed:
        raise RuntimeError(f"frozen fingerprint prerequisite failed: {checks}")
    prerequisite = {
        "passed": True,
        "checks": checks,
        "outcome": prior["decision"]["outcome"],
        "network_population_inference": prior["decision"][
            "network_population_inference"
        ],
        "result": {
            "path": legacy_identifier(UPSTREAM_RESULT_PATH),
            "sha256": file_sha256(UPSTREAM_RESULT_PATH),
        },
    }
    return artifact_validation, prerequisite, prior_q_shape


def edge_metadata(specification: dict, protocol, geometry) -> dict:
    """Return and validate the prospectively frozen edge design."""

    labels = tuple(protocol.item_labels)
    pair_labels = tuple(
        f"{labels[first]}-{labels[second]}" for first, second in geometry.pairs
    )
    distances = symbolic_distances(protocol, geometry.pairs)
    nonlearned = np.asarray(
        [pair not in protocol.learned_pairs for pair in geometry.pairs], dtype=bool
    )
    selected_labels = tuple(
        label
        for label, selected in zip(pair_labels, nonlearned, strict=True)
        if selected
    )
    selected_distances = distances[nonlearned]
    expected_labels = tuple(specification["field_contract"]["nonlearned_pair_labels"])
    levels = np.asarray(specification["field_contract"]["distance_levels"], dtype=int)
    counts = np.asarray(
        specification["field_contract"]["distance_level_pair_counts"], dtype=int
    )
    observed_counts = np.asarray(
        [np.sum(selected_distances == level) for level in levels], dtype=int
    )
    centered = selected_distances - np.mean(selected_distances)
    denominator = float(centered @ centered)
    if not (
        len(geometry.pairs) == 28
        and int(np.sum(nonlearned)) == 20
        and selected_labels == expected_labels
        and np.array_equal(observed_counts, counts)
        and abs(float(np.mean(selected_distances)) - 2.8) <= TOLERANCE
        and abs(denominator - 57.2) <= TOLERANCE
    ):
        raise RuntimeError("frozen edge design does not match the allocation contract")
    return {
        "pair_labels": pair_labels,
        "nonlearned_pair_labels": selected_labels,
        "correct_sign": np.asarray(geometry.true_sign, dtype=np.float64),
        "distances": distances,
        "nonlearned": nonlearned,
        "selected_distances": selected_distances,
        "distance_levels": levels,
        "distance_level_pair_counts": observed_counts,
        "distance_weights": centered / denominator,
        "distance_mean": float(np.mean(selected_distances)),
        "distance_denominator": denominator,
    }


def allocation_tensor_view(
    estimands: dict,
    posterior: dict,
    metadata: dict,
    temperature: float,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Construct Delta-g and its exact fixed-sigmoid slope bridge."""

    decomposition_n = estimands["decompositions"]["N"]
    decomposition_p = estimands["decompositions"]["P"]
    g_n = np.asarray(decomposition_n["gradient"], dtype=np.float64)
    c_n = np.asarray(decomposition_n["residual"], dtype=np.float64)
    g_p = np.asarray(decomposition_p["gradient"], dtype=np.float64)
    scale = np.asarray(estimands["posterior_to_neural_scale_k"], dtype=np.float64)
    g_p_tilde = scale[:, None] * g_p
    delta_g = g_n - g_p_tilde
    sign = metadata["correct_sign"][None, :]
    nonlearned = metadata["nonlearned"]
    delta = sign * delta_g
    probability_nn = exact_probability(sign * (g_n + c_n), temperature)
    probability_tilde = exact_probability(sign * (g_p_tilde + c_n), temperature)
    delta_p_full = probability_nn - probability_tilde
    delta_selected = delta[:, nonlearned]
    delta_p = delta_p_full[:, nonlearned]
    q = delta_p * metadata["distance_weights"][None, :]
    q_by_distance = np.stack(
        [
            np.sum(q[:, metadata["selected_distances"] == level], axis=1)
            for level in metadata["distance_levels"]
        ],
        axis=1,
    )
    q_shape = np.asarray(estimands["Q_shape"], dtype=np.float64)
    uncertainty = np.asarray(posterior["arrays"]["posterior_entropy"], dtype=np.float64)
    coverage = np.asarray(posterior["arrays"]["coverage"], dtype=np.float64)
    arrays = {
        "delta_g_canonical": delta_g[:, nonlearned],
        "delta": delta_selected,
        "delta_p": delta_p,
        "q": q,
        "q_by_distance": q_by_distance,
        "delta_distance_slope": delta_selected @ metadata["distance_weights"],
        "q_shape": q_shape,
        "uncertainty": uncertainty,
        "coverage": coverage,
    }
    integrity = {
        "delta_p_probability_reconstruction_max_abs_error": float(
            np.max(
                np.abs(
                    delta_p_full
                    - (
                        np.asarray(estimands["probabilities"]["NN"])
                        - np.asarray(estimands["probabilities"]["tildePN"])
                    )
                )
            )
        ),
        "g_p_tilde_projection_max_abs_error": float(
            np.max(
                np.abs(
                    g_p_tilde
                    - np.asarray(
                        estimands["decompositions"]["P_tilde"]["gradient"],
                        dtype=np.float64,
                    )
                )
            )
        ),
        "q_sum_equals_q_shape_max_abs_error": float(
            np.max(np.abs(np.sum(q, axis=1) - q_shape))
        ),
        "q_by_distance_sum_equals_q_shape_max_abs_error": float(
            np.max(np.abs(np.sum(q_by_distance, axis=1) - q_shape))
        ),
        "all_allocation_arrays_finite": bool(
            all(np.all(np.isfinite(values)) for values in arrays.values())
        ),
    }
    return arrays, integrity


def bootstrap_counts(specification: dict, seed: int, subjects: int) -> np.ndarray:
    samples = int(specification["statistical_contract"]["bootstrap_samples"])
    bootstrap_seed = int(
        specification["statistical_contract"]["bootstrap_seeds"][str(seed)]
    )
    return (
        np.random.default_rng(bootstrap_seed)
        .multinomial(
            subjects,
            np.full(subjects, 1.0 / subjects),
            size=samples,
        )
        .astype(np.float64)
    )


def classify_direction(summary: dict) -> str:
    interval = summary["bootstrap"]
    if float(interval["lower95"]) > 0.0:
        return "resolved_positive"
    if float(interval["upper95"]) < 0.0:
        return "resolved_negative"
    return "unresolved"


def _scalar_summary(values: np.ndarray, counts: np.ndarray) -> dict:
    rows = np.asarray(values, dtype=np.float64)
    denominator = np.sum(counts, axis=1)
    samples = counts @ rows / denominator
    return interval_summary(float(np.mean(rows)), samples)


def _distance_residual_maker(distances: np.ndarray) -> np.ndarray:
    design = np.column_stack((np.ones(len(distances)), distances))
    return np.eye(len(distances)) - design @ np.linalg.solve(
        design.T @ design, design.T
    )


def pair_fingerprint_vectors(
    delta: np.ndarray,
    delta_p: np.ndarray,
    q: np.ndarray,
    distances: np.ndarray,
    distance_weights: np.ndarray,
    counts: np.ndarray,
) -> dict:
    """Construct point and bootstrap pair vectors after linear-distance removal."""

    denominator = np.sum(counts, axis=1)
    residual_maker = _distance_residual_maker(distances)
    point_delta = np.mean(delta, axis=0)
    point_delta_p = np.mean(delta_p, axis=0)
    point_q = np.mean(q, axis=0)
    bootstrap_delta = counts @ delta / denominator[:, None]
    bootstrap_delta_p = counts @ delta_p / denominator[:, None]
    bootstrap_q = counts @ q / denominator[:, None]
    r_delta = point_delta @ residual_maker
    r_delta_p = point_delta_p @ residual_maker
    r_q = r_delta_p * distance_weights
    bootstrap_r_delta = bootstrap_delta @ residual_maker
    bootstrap_r_delta_p = bootstrap_delta_p @ residual_maker
    bootstrap_r_q = bootstrap_r_delta_p * distance_weights[None, :]
    predicted_q = (point_delta_p - r_delta_p) * distance_weights
    bootstrap_predicted_q = (
        bootstrap_delta_p - bootstrap_r_delta_p
    ) * distance_weights[None, :]
    return {
        "point": {
            "mu_delta": point_delta,
            "mu_delta_p": point_delta_p,
            "mu_q": point_q,
            "r_delta": r_delta,
            "r_delta_p": r_delta_p,
            "r_q": r_q,
            "distance_predicted_q": predicted_q,
        },
        "bootstrap": {
            "mu_delta": bootstrap_delta,
            "mu_delta_p": bootstrap_delta_p,
            "mu_q": bootstrap_q,
            "r_delta": bootstrap_r_delta,
            "r_delta_p": bootstrap_r_delta_p,
            "r_q": bootstrap_r_q,
            "distance_predicted_q": bootstrap_predicted_q,
        },
        "integrity": {
            "r_delta_orthogonal_intercept_max_abs_error": float(abs(np.sum(r_delta))),
            "r_delta_orthogonal_distance_max_abs_error": float(
                abs(r_delta @ distances)
            ),
            "r_delta_p_orthogonal_intercept_max_abs_error": float(
                abs(np.sum(r_delta_p))
            ),
            "r_delta_p_orthogonal_distance_max_abs_error": float(
                abs(r_delta_p @ distances)
            ),
            "bootstrap_r_delta_orthogonal_intercept_max_abs_error": float(
                np.max(np.abs(np.sum(bootstrap_r_delta, axis=1)))
            ),
            "bootstrap_r_delta_orthogonal_distance_max_abs_error": float(
                np.max(np.abs(bootstrap_r_delta @ distances))
            ),
            "bootstrap_r_delta_p_orthogonal_intercept_max_abs_error": float(
                np.max(np.abs(np.sum(bootstrap_r_delta_p, axis=1)))
            ),
            "bootstrap_r_delta_p_orthogonal_distance_max_abs_error": float(
                np.max(np.abs(bootstrap_r_delta_p @ distances))
            ),
            "r_q_formula_max_abs_error": float(
                np.max(np.abs(r_q - r_delta_p * distance_weights))
            ),
            "mu_q_formula_max_abs_error": float(
                np.max(np.abs(point_q - point_delta_p * distance_weights))
            ),
            "r_q_zero_sum_max_abs_error": float(abs(np.sum(r_q))),
            "bootstrap_r_q_zero_sum_max_abs_error": float(
                np.max(np.abs(np.sum(bootstrap_r_q, axis=1)))
            ),
            "q_ledger_reconstruction_max_abs_error": float(
                np.max(np.abs(point_q - predicted_q - r_q))
            ),
            "bootstrap_q_ledger_reconstruction_max_abs_error": float(
                np.max(np.abs(bootstrap_q - bootstrap_predicted_q - bootstrap_r_q))
            ),
        },
    }


def _balanced_joint_coefficients(
    response: np.ndarray,
    uncertainty_z: np.ndarray,
    coverage_z: np.ndarray,
    counts: np.ndarray,
) -> dict:
    """Fit pair fixed effects through their exact balanced-design reduction."""

    rows = np.asarray(response, dtype=np.float64)
    if rows.ndim != 2:
        raise ValueError("joint-model response must be participant by pair")
    subjects, pairs = rows.shape
    design = np.column_stack((np.ones(subjects), uncertainty_z, coverage_z))
    point_rank = int(np.linalg.matrix_rank(design, tol=RANK_TOLERANCE))
    if point_rank != 3:
        raise NonInterpretableEstimate("joint U/C point design does not have rank 3")
    response_mean = np.mean(rows, axis=1)
    point = np.linalg.solve(design.T @ design, design.T @ response_mean)

    normal = np.einsum("bs,si,sj->bij", counts, design, design)
    ranks = np.linalg.matrix_rank(normal, tol=RANK_TOLERANCE)
    if not np.all(ranks == 3):
        raise NonInterpretableEstimate("joint U/C bootstrap design lost rank")
    rhs = np.einsum("bs,si,s->bi", counts, design, response_mean)
    bootstrap = np.linalg.solve(normal, rhs[..., None])[..., 0]

    pair_dummies = np.tile(np.eye(pairs), (subjects, 1))
    repeated_predictors = np.repeat(
        np.column_stack((uncertainty_z, coverage_z)), pairs, axis=0
    )
    full_design = np.column_stack((pair_dummies, repeated_predictors))
    full_rank = int(np.linalg.matrix_rank(full_design, tol=RANK_TOLERANCE))
    if full_rank != pairs + 2:
        raise NonInterpretableEstimate("explicit pair-FE point design lost rank")
    full_point = np.linalg.solve(
        full_design.T @ full_design,
        full_design.T @ rows.reshape(-1),
    )
    explicit_error = float(np.max(np.abs(full_point[-2:] - point[1:])))
    return {
        "point": point,
        "bootstrap": bootstrap,
        "point_reduced_rank": point_rank,
        "minimum_bootstrap_reduced_rank": int(np.min(ranks)),
        "point_full_rank": full_rank,
        "explicit_pair_fe_coefficient_max_abs_error": explicit_error,
    }


def joint_model_statistics(
    delta: np.ndarray,
    q: np.ndarray,
    q_shape: np.ndarray,
    uncertainty: np.ndarray,
    coverage: np.ndarray,
    counts: np.ndarray,
) -> tuple[dict, dict]:
    """Fit the frozen simultaneous U/C models and their exact q identity."""

    if not all(
        np.all(np.isfinite(values))
        for values in (delta, q, q_shape, uncertainty, coverage, counts)
    ):
        raise NonInterpretableEstimate("joint U/C model input contains nonfinite data")

    uncertainty_sd = float(np.sqrt(np.mean((uncertainty - np.mean(uncertainty)) ** 2)))
    coverage_sd = float(np.sqrt(np.mean((coverage - np.mean(coverage)) ** 2)))
    if uncertainty_sd <= 0.0 or coverage_sd <= 0.0:
        raise NonInterpretableEstimate("frozen U/C predictor has zero variance")
    uncertainty_z = (uncertainty - np.mean(uncertainty)) / uncertainty_sd
    coverage_z = (coverage - np.mean(coverage)) / coverage_sd
    delta_fit = _balanced_joint_coefficients(delta, uncertainty_z, coverage_z, counts)
    q_fit = _balanced_joint_coefficients(q, uncertainty_z, coverage_z, counts)
    q_shape_fit = _balanced_joint_coefficients(
        np.asarray(q_shape)[:, None], uncertainty_z, coverage_z, counts
    )
    coefficient_names = ("U", "C")
    summaries = {}
    statuses = {}
    for prefix, fit in (("beta_delta", delta_fit), ("beta_q", q_fit)):
        for offset, name in enumerate(coefficient_names, start=1):
            key = f"{prefix}_{name}"
            summary = interval_summary(
                float(fit["point"][offset]), fit["bootstrap"][:, offset]
            )
            summaries[key] = summary
            statuses[key] = classify_direction(summary)
    q_identity_point = np.max(
        np.abs(20.0 * q_fit["point"][1:] - q_shape_fit["point"][1:])
    )
    q_identity_bootstrap = np.max(
        np.abs(20.0 * q_fit["bootstrap"][:, 1:] - q_shape_fit["bootstrap"][:, 1:])
    )
    integrity = {
        "uncertainty_mean": float(np.mean(uncertainty)),
        "uncertainty_ddof0_sd": uncertainty_sd,
        "coverage_mean": float(np.mean(coverage)),
        "coverage_ddof0_sd": coverage_sd,
        "predictor_correlation": float(np.corrcoef(uncertainty_z, coverage_z)[0, 1]),
        "delta_point_reduced_rank": delta_fit["point_reduced_rank"],
        "delta_minimum_bootstrap_reduced_rank": delta_fit[
            "minimum_bootstrap_reduced_rank"
        ],
        "delta_point_full_rank": delta_fit["point_full_rank"],
        "q_point_reduced_rank": q_fit["point_reduced_rank"],
        "q_minimum_bootstrap_reduced_rank": q_fit["minimum_bootstrap_reduced_rank"],
        "q_point_full_rank": q_fit["point_full_rank"],
        "delta_explicit_pair_fe_coefficient_max_abs_error": delta_fit[
            "explicit_pair_fe_coefficient_max_abs_error"
        ],
        "q_explicit_pair_fe_coefficient_max_abs_error": q_fit[
            "explicit_pair_fe_coefficient_max_abs_error"
        ],
        "q_coefficient_identity_point_max_abs_error": float(q_identity_point),
        "q_coefficient_identity_bootstrap_max_abs_error": float(q_identity_bootstrap),
    }
    internal = {
        "uncertainty_z": uncertainty_z,
        "coverage_z": coverage_z,
        "delta_fit": delta_fit,
        "q_fit": q_fit,
        "q_shape_fit": q_shape_fit,
    }
    return {
        "summaries": summaries,
        "statuses": statuses,
        "integrity": integrity,
    }, internal


def _pearson_rows(
    first: np.ndarray, second: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("correlation rows must have matching two-dimensional shapes")
    input_finite = np.all(np.isfinite(a), axis=1) & np.all(np.isfinite(b), axis=1)
    a_centered = a - np.mean(a, axis=1, keepdims=True)
    b_centered = b - np.mean(b, axis=1, keepdims=True)
    denominator = np.sqrt(
        np.sum(a_centered * a_centered, axis=1)
        * np.sum(b_centered * b_centered, axis=1)
    )
    nonfinite = ~input_finite | ~np.isfinite(denominator)
    defined = ~nonfinite & (denominator > 0.0)
    values = np.full(len(a), np.nan, dtype=np.float64)
    values[defined] = (
        np.sum(a_centered[defined] * b_centered[defined], axis=1)
        / (denominator[defined])
    )
    nonfinite |= defined & ~np.isfinite(values)
    defined &= ~nonfinite
    return values, defined, nonfinite


def correlation_summary(
    first_point: np.ndarray,
    second_point: np.ndarray,
    first_bootstrap: np.ndarray,
    second_bootstrap: np.ndarray,
) -> dict:
    """Summarize a frozen cross-network vector correlation without filtering."""

    point_values, point_defined, point_nonfinite = _pearson_rows(
        np.asarray(first_point)[None, :], np.asarray(second_point)[None, :]
    )
    bootstrap_values, bootstrap_defined, bootstrap_nonfinite = _pearson_rows(
        first_bootstrap, second_bootstrap
    )
    degenerate_draws = int(np.sum(~bootstrap_defined))
    if not bool(point_defined[0]) or degenerate_draws:
        return {
            "point": float(point_values[0]) if point_defined[0] else None,
            "point_nonfinite": bool(point_nonfinite[0]),
            "bootstrap": {
                "samples": len(bootstrap_values),
                "degenerate_draws": degenerate_draws,
                "nonfinite_draws": int(np.sum(bootstrap_nonfinite)),
                "zero_norm_draws": int(
                    np.sum(~bootstrap_defined & ~bootstrap_nonfinite)
                ),
                "intervals": None,
            },
            "status": "unresolved_degenerate",
        }
    summary = interval_summary(float(point_values[0]), bootstrap_values)
    summary["bootstrap"]["degenerate_draws"] = 0
    summary["bootstrap"]["nonfinite_draws"] = 0
    summary["bootstrap"]["zero_norm_draws"] = 0
    summary["status"] = classify_direction(summary)
    return summary


def seed_statistics(
    specification: dict,
    seed: int,
    arrays: dict[str, np.ndarray],
    metadata: dict,
) -> tuple[dict, dict]:
    """Compute all registered within-network and cross-network inputs."""

    subjects = len(arrays["q_shape"])
    counts = bootstrap_counts(specification, seed, subjects)
    delta_distance = _scalar_summary(arrays["delta_distance_slope"], counts)
    joint, joint_internal = joint_model_statistics(
        arrays["delta"],
        arrays["q"],
        arrays["q_shape"],
        arrays["uncertainty"],
        arrays["coverage"],
        counts,
    )
    pair_vectors = pair_fingerprint_vectors(
        arrays["delta"],
        arrays["delta_p"],
        arrays["q"],
        metadata["selected_distances"],
        metadata["distance_weights"],
        counts,
    )
    denominator = np.sum(counts, axis=1)
    q_distance_point = np.mean(arrays["q_by_distance"], axis=0)
    q_distance_bootstrap = counts @ arrays["q_by_distance"] / denominator[:, None]
    q_shape_bootstrap = counts @ arrays["q_shape"] / denominator
    q_sum_bootstrap = counts @ np.sum(arrays["q"], axis=1) / denominator
    q_distance_sum_bootstrap = np.sum(q_distance_bootstrap, axis=1)
    q_distance_summaries = {
        str(int(level)): interval_summary(
            float(q_distance_point[index]), q_distance_bootstrap[:, index]
        )
        for index, level in enumerate(metadata["distance_levels"])
    }
    public = {
        "summaries": {
            "beta_delta_distance": delta_distance,
            **joint["summaries"],
        },
        "statuses": {
            "beta_delta_distance": classify_direction(delta_distance),
            **joint["statuses"],
        },
        "pair_vectors": {
            name: values.tolist() for name, values in pair_vectors["point"].items()
        },
        "q_by_distance_mean": q_distance_point.tolist(),
        "q_by_distance_summaries": q_distance_summaries,
        "joint_model": {
            "predictor_order": ["posterior_uncertainty_U", "coverage_C"],
            "integrity": joint["integrity"],
        },
        "raw_subject_level": {
            "delta_g_canonical_nonlearned": arrays["delta_g_canonical"].tolist(),
            "delta_correct_signed": arrays["delta"].tolist(),
            "delta_p": arrays["delta_p"].tolist(),
            "q": arrays["q"].tolist(),
            "q_by_distance": arrays["q_by_distance"].tolist(),
            "posterior_uncertainty_U": arrays["uncertainty"].tolist(),
            "coverage_C": arrays["coverage"].tolist(),
            "beta_delta_distance": arrays["delta_distance_slope"].tolist(),
            "q_shape_recomputed": arrays["q_shape"].tolist(),
        },
        "integrity": {
            "bootstrap_samples": int(counts.shape[0]),
            "bootstrap_subjects": int(counts.shape[1]),
            "q_bootstrap_sum_equals_q_shape_max_abs_error": float(
                np.max(np.abs(q_sum_bootstrap - q_shape_bootstrap))
            ),
            "q_by_distance_bootstrap_sum_equals_q_shape_max_abs_error": float(
                np.max(np.abs(q_distance_sum_bootstrap - q_shape_bootstrap))
            ),
            **pair_vectors["integrity"],
            **joint["integrity"],
        },
    }
    internal = {
        "counts": counts,
        "pair_point": pair_vectors["point"],
        "pair_bootstrap": pair_vectors["bootstrap"],
        "q_distance_point": q_distance_point,
        "q_distance_bootstrap": q_distance_bootstrap,
        "joint": joint_internal,
    }
    return public, internal


def analyze_seed(
    specification: dict,
    seed: int,
    artifact_validation: dict,
    prior_q_shape: np.ndarray,
) -> tuple[dict, dict]:
    """Replay one frozen backbone and compute the registered allocation audit."""

    evaluation = specification["evaluation"]
    artifact = artifact_validation["lock"]["artifacts"][str(seed)]["checkpoint"]
    checkpoint_path = resolve_registered_path(artifact["path"])
    backbone, model_config, checkpoint = load_retro_checkpoint(
        checkpoint_path, int(evaluation["subjects"])
    )
    backbone.eval()
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
    metadata = edge_metadata(specification, protocol, geometry)
    fast_weights = evaluator.learn_fast_weights(FastWeightIntervention.INTACT)
    neural_margin = readout_margin_fields(evaluator, fast_weights, geometry)
    posterior, posterior_integrity = posterior_descriptors(
        evaluator,
        geometry,
        {
            "posterior_comparator": {
                "posterior_temperature": evaluation["posterior_temperature"],
                "choice_temperature": evaluation["choice_temperature"],
            }
        },
    )
    estimands = field_reassembly_estimands(
        neural_margin,
        posterior["fields"]["same_unit_margin"],
        geometry,
        metadata["distances"],
        metadata["nonlearned"],
        float(evaluation["choice_temperature"]),
    )
    arrays, allocation_integrity = allocation_tensor_view(
        estimands,
        posterior,
        metadata,
        float(evaluation["choice_temperature"]),
    )
    statistics, internal = seed_statistics(specification, seed, arrays, metadata)
    after = tensor_hashes(backbone)

    field_errors = {
        f"{name}_max_abs_error": float(np.max(np.abs(estimands[name])))
        for name in ALGEBRA_ERROR_ARRAYS
    }
    prior_error = float(np.max(np.abs(arrays["q_shape"] - prior_q_shape)))
    statistics["raw_subject_level"]["q_shape_prior_anchor"] = prior_q_shape.tolist()
    posterior_error_names = (
        "posterior_inverse_link_max_abs_error",
        "posterior_orientation_reversal_max_abs_error",
        "posterior_expected_rank_Hodge_max_abs_error",
        "coverage_binary_max_abs_error",
        "coverage_relation_reuse_max_abs_error",
        "coverage_unique_fraction_max_abs_error",
    )
    integrity = {
        **posterior_integrity,
        **field_errors,
        **allocation_integrity,
        **statistics["integrity"],
        "q_shape_prior_replay_max_abs_error": prior_error,
        "checkpoint_sha_matches_artifact_lock": checkpoint.sha256 == artifact["sha256"],
        "backbone_tensor_hashes_unchanged": before == after,
        "subjects": int(model_config.bs),
        "edges": len(geometry.pairs),
        "nonlearned_pairs": int(np.sum(metadata["nonlearned"])),
        "distance_levels": metadata["distance_levels"].astype(int).tolist(),
        "distance_level_pair_counts": metadata["distance_level_pair_counts"]
        .astype(int)
        .tolist(),
        "distance_mean": metadata["distance_mean"],
        "distance_denominator": metadata["distance_denominator"],
    }
    tolerance_errors = {
        **{name: integrity[name] for name in posterior_error_names},
        **field_errors,
        **allocation_integrity,
        "q_shape_prior_replay_max_abs_error": prior_error,
        **{
            name: value
            for name, value in statistics["integrity"].items()
            if name.endswith("_max_abs_error")
        },
    }
    tolerance_errors.pop("all_allocation_arrays_finite", None)
    integrity["passed"] = bool(
        all(float(value) <= TOLERANCE for value in tolerance_errors.values())
        and integrity["all_allocation_arrays_finite"]
        and integrity["checkpoint_sha_matches_artifact_lock"]
        and integrity["backbone_tensor_hashes_unchanged"]
        and 0.0 <= integrity["posterior_entropy_min"]
        and integrity["posterior_entropy_max"] <= 1.0
        and 0.0 <= integrity["coverage_min"]
        and integrity["coverage_max"] <= 1.0
        and integrity["coverage_unique_relations_min"] == 8
        and integrity["coverage_unique_relations_max"] == 8
        and integrity["subjects"] == int(evaluation["subjects"]) == 77
        and integrity["edges"] == 28
        and integrity["nonlearned_pairs"] == 20
        and integrity["distance_levels"]
        == specification["field_contract"]["distance_levels"]
        and integrity["distance_level_pair_counts"]
        == specification["field_contract"]["distance_level_pair_counts"]
        and integrity["bootstrap_samples"]
        == int(specification["statistical_contract"]["bootstrap_samples"])
        and integrity["bootstrap_subjects"] == 77
        and integrity["delta_point_reduced_rank"] == 3
        and integrity["delta_minimum_bootstrap_reduced_rank"] == 3
        and integrity["delta_point_full_rank"] == 22
        and integrity["q_point_reduced_rank"] == 3
        and integrity["q_minimum_bootstrap_reduced_rank"] == 3
        and integrity["q_point_full_rank"] == 22
    )
    public = {
        "seed": seed,
        "checkpoint": {"path": artifact["path"], "sha256": checkpoint.sha256},
        "condition": evaluation["condition"],
        "statistics": statistics,
        "integrity": integrity,
    }
    internal.update(
        {
            "uncertainty": arrays["uncertainty"],
            "coverage": arrays["coverage"],
        }
    )
    return public, internal


def _stable_direction(statuses: dict[str, str]) -> str | None:
    values = tuple(statuses.values())
    if len(set(values)) == 1 and values[0] in {
        "resolved_positive",
        "resolved_negative",
    }:
        return values[0].removeprefix("resolved_")
    return None


def cross_network_analysis(
    specification: dict,
    seeds: dict[str, dict],
    internal: dict[str, dict],
) -> tuple[dict, dict]:
    """Compute registered cross-network correlations and the outcome tree."""

    seed_keys = tuple(
        str(seed) for seed in specification["artifact_contract"]["mandatory_seeds"]
    )
    if set(seeds) != set(seed_keys) or not all(
        seeds[key]["integrity"]["passed"] for key in seed_keys
    ):
        return {
            "outcome": "noninterpretable_integrity_failure",
            "axes": "not_evaluated",
            "network_population_inference": "not_performed",
        }, {"passed": False, "reason": "seed_integrity"}
    first, second = (internal[key] for key in seed_keys)
    cross_integrity = {
        "uncertainty_cross_network_max_abs_error": float(
            np.max(np.abs(first["uncertainty"] - second["uncertainty"]))
        ),
        "coverage_cross_network_max_abs_error": float(
            np.max(np.abs(first["coverage"] - second["coverage"]))
        ),
    }
    cross_integrity["passed"] = bool(
        cross_integrity["uncertainty_cross_network_max_abs_error"] <= TOLERANCE
        and cross_integrity["coverage_cross_network_max_abs_error"] <= TOLERANCE
    )
    if not cross_integrity["passed"]:
        return {
            "outcome": "noninterpretable_integrity_failure",
            "axes": "not_evaluated",
            "network_population_inference": "not_performed",
        }, cross_integrity
    try:
        correlations = {
            "pair_r_delta": correlation_summary(
                first["pair_point"]["r_delta"],
                second["pair_point"]["r_delta"],
                first["pair_bootstrap"]["r_delta"],
                second["pair_bootstrap"]["r_delta"],
            ),
            "pair_r_q": correlation_summary(
                first["pair_point"]["r_q"],
                second["pair_point"]["r_q"],
                first["pair_bootstrap"]["r_q"],
                second["pair_bootstrap"]["r_q"],
            ),
            "distance_q": correlation_summary(
                first["q_distance_point"],
                second["q_distance_point"],
                first["q_distance_bootstrap"],
                second["q_distance_bootstrap"],
            ),
            "diagnostic_raw_mu_delta": correlation_summary(
                first["pair_point"]["mu_delta"],
                second["pair_point"]["mu_delta"],
                first["pair_bootstrap"]["mu_delta"],
                second["pair_bootstrap"]["mu_delta"],
            ),
            "diagnostic_raw_mu_delta_p": correlation_summary(
                first["pair_point"]["mu_delta_p"],
                second["pair_point"]["mu_delta_p"],
                first["pair_bootstrap"]["mu_delta_p"],
                second["pair_bootstrap"]["mu_delta_p"],
            ),
            "diagnostic_raw_mu_q": correlation_summary(
                first["pair_point"]["mu_q"],
                second["pair_point"]["mu_q"],
                first["pair_bootstrap"]["mu_q"],
                second["pair_bootstrap"]["mu_q"],
            ),
        }
    except NonInterpretableEstimate as error:
        cross_integrity.update(
            {
                "passed": False,
                "reason": "nonfinite_correlation_input",
                "error": str(error),
            }
        )
        return {
            "outcome": "noninterpretable_integrity_failure",
            "axes": "not_evaluated",
            "network_population_inference": "not_performed",
        }, cross_integrity
    status_by_seed = {
        metric: {key: seeds[key]["statistics"]["statuses"][metric] for key in seed_keys}
        for metric in (
            "beta_delta_distance",
            "beta_delta_U",
            "beta_delta_C",
            "beta_q_U",
            "beta_q_C",
        )
    }
    distance_field_direction = _stable_direction(status_by_seed["beta_delta_distance"])
    uncertainty_field_direction = _stable_direction(status_by_seed["beta_delta_U"])
    uncertainty_bridge_direction = _stable_direction(status_by_seed["beta_q_U"])
    coverage_field_direction = _stable_direction(status_by_seed["beta_delta_C"])
    coverage_bridge_direction = _stable_direction(status_by_seed["beta_q_C"])
    axes = {
        "pair_identity": {
            "field_stable": correlations["pair_r_delta"]["status"]
            == "resolved_positive",
            "bridge_stable": correlations["pair_r_q"]["status"] == "resolved_positive",
        },
        "symbolic_distance": {
            "field_stable": distance_field_direction is not None,
            "field_direction": distance_field_direction,
            "bridge_stable": correlations["distance_q"]["status"]
            == "resolved_positive",
        },
        "posterior_uncertainty": {
            "field_stable": uncertainty_field_direction is not None,
            "field_direction": uncertainty_field_direction,
            "bridge_stable": uncertainty_bridge_direction is not None,
            "bridge_direction": uncertainty_bridge_direction,
        },
        "effective_evidence_coverage": {
            "field_stable": coverage_field_direction is not None,
            "field_direction": coverage_field_direction,
            "bridge_stable": coverage_bridge_direction is not None,
            "bridge_direction": coverage_bridge_direction,
        },
    }
    axes["pair_identity"]["policy_effective"] = bool(
        axes["pair_identity"]["field_stable"] and axes["pair_identity"]["bridge_stable"]
    )
    axes["symbolic_distance"]["policy_effective"] = bool(
        axes["symbolic_distance"]["field_stable"]
        and axes["symbolic_distance"]["bridge_stable"]
    )
    for name in ("posterior_uncertainty", "effective_evidence_coverage"):
        axes[name]["policy_effective"] = bool(
            axes[name]["field_stable"]
            and axes[name]["bridge_stable"]
            and axes[name]["field_direction"] == axes[name]["bridge_direction"]
        )
    policy_axes = [name for name, row in axes.items() if row["policy_effective"]]
    any_stable = any(
        row["field_stable"] or row["bridge_stable"] for row in axes.values()
    )
    state_axes = {
        "posterior_uncertainty",
        "effective_evidence_coverage",
    }.intersection(policy_axes)
    structural_axes = {"pair_identity", "symbolic_distance"}.intersection(policy_axes)
    if policy_axes:
        outcome = "policy_effective_allocation_localized"
        if state_axes and structural_axes:
            scope = "both"
        elif state_axes:
            scope = "state_dependent"
        else:
            scope = "structural_only"
    elif any_stable:
        outcome = "field_structure_without_policy_bridge"
        scope = "none"
    else:
        outcome = "no_stable_allocation_localization"
        scope = "none"
    cross_integrity["correlation_degenerate_draws"] = {
        name: int(row["bootstrap"].get("degenerate_draws", 0))
        for name, row in correlations.items()
    }
    decision = {
        "outcome": outcome,
        "localization_scope": scope,
        "policy_effective_axes": policy_axes,
        "axes": axes,
        "correlations": correlations,
        "conditional_next_step": (
            "prospective_state_dependent_P_T_generation_question"
            if scope in {"state_dependent", "both"}
            else "prospective_comparator_adequacy"
        ),
        "network_population_inference": "not_performed",
    }
    return decision, cross_integrity


def evaluate_audit(
    specification: dict,
    runtime: dict,
    source_validation: dict,
    artifact_validation: dict,
    prerequisite: dict,
    prior_q_shape: dict[str, np.ndarray],
) -> dict:
    seeds = {}
    internal = {}
    for seed in specification["artifact_contract"]["mandatory_seeds"]:
        try:
            public, private = analyze_seed(
                specification,
                int(seed),
                artifact_validation,
                prior_q_shape[str(seed)],
            )
            seeds[str(seed)] = public
            internal[str(seed)] = private
        except NonInterpretableEstimate as error:
            artifact = artifact_validation["lock"]["artifacts"][str(seed)]["checkpoint"]
            seeds[str(seed)] = {
                "seed": int(seed),
                "checkpoint": artifact,
                "condition": specification["evaluation"]["condition"],
                "statistics": {},
                "integrity": {
                    "passed": False,
                    "failure_type": "registered_noninterpretable_estimate",
                    "error": str(error),
                },
            }
    decision, cross_integrity = cross_network_analysis(specification, seeds, internal)
    return {
        "schema_version": 1,
        "audit_id": specification["audit_id"],
        "registration_status": specification["registration_status"],
        "claim_boundary": specification["claim_boundary"],
        "claim_boundaries": {
            "multiplicity": specification["statistical_contract"][
                "multiplicity_boundary"
            ],
            "outcome_contingent_interpretation": specification[
                "outcome_contingent_interpretation"
            ],
        },
        "runtime": runtime,
        "source_validation": source_validation,
        "artifact_validation": artifact_validation,
        "fingerprint_prerequisite": prerequisite,
        "edge_contract": {
            "nonlearned_pair_labels": specification["field_contract"][
                "nonlearned_pair_labels"
            ],
            "distance_levels": specification["field_contract"]["distance_levels"],
            "distance_level_pair_counts": specification["field_contract"][
                "distance_level_pair_counts"
            ],
            "distance_mean": 2.8,
            "distance_denominator": 57.2,
        },
        "seeds": seeds,
        "cross_network_integrity": cross_integrity,
        "decision": decision,
    }


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Run the frozen global-policy allocation audit."
    )
    parser.add_argument("stage", choices=("evaluate",))
    parser.add_argument(
        "--specification", type=Path, default=DEFAULT_SPECIFICATION_PATH
    )
    parser.add_argument(
        "--implementation-lock", type=Path, default=DEFAULT_IMPLEMENTATION_LOCK_PATH
    )
    parser.add_argument("--output-root", type=Path, default=UPSTREAM_OUTPUT_ROOT)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT_PATH)
    return parser.parse_args(args)


def main(args=None) -> int:
    parsed = parse_args(args)
    canonical_paths(parsed)
    runtime = require_formal_runtime()
    specification = load_json(parsed.specification)
    if tuple(specification["artifact_contract"]["mandatory_seeds"]) != (2106, 2107):
        raise RuntimeError("allocation audit requires exactly seeds 2106 and 2107")
    source_validation = validate_sources(
        parsed.specification, parsed.implementation_lock
    )
    git_freeze = require_pushed_freeze(
        required_freeze_paths(parsed.specification, parsed.implementation_lock)
    )
    artifact_validation, prerequisite, prior_q_shape = validate_upstream(specification)
    result = evaluate_audit(
        specification,
        runtime,
        source_validation,
        artifact_validation,
        prerequisite,
        prior_q_shape,
    )
    result["git_freeze_validation"] = git_freeze
    write_json_exclusive(parsed.result, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
