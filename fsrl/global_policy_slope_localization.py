"""Read-only localization of nonlearned slope in the frozen global policy."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .assembly_trajectory import (
    bootstrap_counts,
    build_complete_graph_geometry,
    exact_prefix_trajectory,
    hodge_potentials,
    normalize_potentials,
    readout_margin_fields,
    summarize_difference,
    summarize_subjects,
)
from .curvature_gate_pilot import load_json, write_json
from .dual_evidence_access_confirmation import validate_artifacts
from .formal_runtime import require_formal_runtime
from .liu_eval import (
    FastWeightIntervention,
    FrozenFastWeightEvaluator,
    load_retro_checkpoint,
)
from .local_behavior_attribution import exact_probability
from .ranking_protocol import load_ranking_protocol
from .study_registry import registered_file_sha256, resolve_record

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPECIFICATION_PATH = (
    resolve_record("benchmarks/global_policy_slope_localization_v1.json")
)
DEFAULT_IMPLEMENTATION_LOCK_PATH = (
    resolve_record("benchmarks/global_policy_slope_localization_v1.lock.json")
)
DEFAULT_RESULT_PATH = resolve_record("results/global_policy_slope_localization_v1.json")
CONFIRMATION_OUTPUT_ROOT = ROOT / "output" / "dual-evidence-access-confirmation-v2-4"


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else resolve_record(candidate)


def subject_slopes(
    values: np.ndarray, distances: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """Return one fixed-design OLS slope per subject."""

    rows = np.asarray(values, dtype=np.float64)
    x = np.asarray(distances, dtype=np.float64)[np.asarray(mask, dtype=bool)]
    if rows.ndim != 2 or rows.shape[1] != len(distances):
        raise ValueError("values must have shape (subjects, complete-graph edges)")
    centered = x - np.mean(x)
    denominator = float(centered @ centered)
    if denominator <= 0.0:
        raise ValueError("symbolic distance must vary over the selected pairs")
    return rows[:, mask] @ centered / denominator


def choice_link_components(
    correct_margins: np.ndarray,
    probabilities: np.ndarray,
    distances: np.ndarray,
    mask: np.ndarray,
) -> dict[str, np.ndarray]:
    """Project choice probability onto margin and retain the exact slope remainder."""

    q = np.asarray(correct_margins, dtype=np.float64)[:, mask]
    p = np.asarray(probabilities, dtype=np.float64)[:, mask]
    q_centered = q - np.mean(q, axis=1, keepdims=True)
    p_centered = p - np.mean(p, axis=1, keepdims=True)
    denominator = np.sum(q_centered * q_centered, axis=1)
    kappa = np.divide(
        np.sum(q_centered * p_centered, axis=1),
        denominator,
        out=np.full(len(q), np.nan, dtype=np.float64),
        where=denominator > 0.0,
    )
    fitted = np.mean(p, axis=1, keepdims=True) + kappa[:, None] * q_centered
    residual = p - fitted
    selected_distances = np.asarray(distances, dtype=np.float64)[mask]
    selected_mask = np.ones(len(selected_distances), dtype=bool)
    beta_margin = subject_slopes(q, selected_distances, selected_mask)
    beta_probability = subject_slopes(p, selected_distances, selected_mask)
    beta_residual = subject_slopes(residual, selected_distances, selected_mask)
    return {
        "kappa": kappa,
        "beta_margin": beta_margin,
        "beta_probability": beta_probability,
        "beta_linearized": kappa * beta_margin,
        "beta_remainder": beta_residual,
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
        path = _resolve(registration["path"])
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
        raise RuntimeError(f"global-policy slope source lock failed: {checks}")
    return {"passed": True, "checks": checks}


def _artifact_validation(specification: dict) -> dict:
    sources = specification["registered_sources"]
    confirmation_specification_path = _resolve(
        sources["v2_4_confirmation_specification"]["path"]
    )
    confirmation_specification = load_json(confirmation_specification_path)
    return validate_artifacts(
        confirmation_specification,
        confirmation_specification_path,
        _resolve(sources["v2_4_confirmation_implementation_lock"]["path"]),
        _resolve(sources["v2_4_confirmation_artifact_lock"]["path"]),
        CONFIRMATION_OUTPUT_ROOT,
    )


def _distances(protocol, pairs: tuple[tuple[int, int], ...]) -> np.ndarray:
    positions = np.empty(protocol.n_items, dtype=np.int64)
    for position, item in enumerate(protocol.true_order_high_to_low):
        positions[item] = position
    return np.asarray(
        [
            abs(int(positions[first]) - int(positions[second]))
            for first, second in pairs
        ],
        dtype=np.float64,
    )


def _summaries(
    raw: dict[str, np.ndarray], counts: np.ndarray, interval: float
) -> dict[str, dict]:
    return {
        name: summarize_subjects(values, counts, interval=interval)
        for name, values in raw.items()
    }


def seed_decision(metrics: dict[str, dict]) -> dict:
    def lower(name: str) -> float:
        return float(metrics[name]["bootstrap"]["lower"])

    def upper(name: str) -> float:
        return float(metrics[name]["bootstrap"]["upper"])

    flags = {
        "global_potential_dominance": lower("beta_m") > 0.0
        and lower("beta_g_minus_0_9_beta_m") > 0.0,
        "normalized_geometry_excess": lower("beta_hat_minus_posterior") > 0.0,
        "positive_residual_contribution": lower("beta_c") > 0.0,
        "neural_probability_slope_excess": lower("beta_p_minus_posterior") > 0.0,
        "choice_link_amplification": lower("beta_e") > 0.0,
        "choice_link_compression": upper("beta_e") < 0.0,
    }
    return {"flags": flags}


def cross_seed_decision(seeds: dict[str, dict]) -> dict:
    names = tuple(next(iter(seeds.values()))["decision"]["flags"])
    links = {}
    for name in names:
        by_seed = {
            seed: bool(row["decision"]["flags"][name]) for seed, row in seeds.items()
        }
        if all(by_seed.values()):
            status = "replicated"
        elif any(by_seed.values()):
            status = "heterogeneous"
        else:
            status = "not_confirmed"
        links[name] = {"status": status, "by_seed": by_seed}
    return {"links": links, "network_population_inference": "not_performed"}


def analyze_seed(
    specification: dict,
    seed: int,
    artifact_validation: dict,
) -> dict:
    evaluation = specification["evaluation"]
    artifact = artifact_validation["lock"]["artifacts"][str(seed)]["checkpoint"]
    checkpoint_path = _resolve(artifact["path"])
    backbone, model_config, checkpoint = load_retro_checkpoint(
        checkpoint_path, int(evaluation["subjects"])
    )
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    protocol = load_ranking_protocol(
        _resolve(specification["registered_sources"]["liu_protocol"]["path"])
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
    distances = _distances(protocol, geometry.pairs)
    learned = np.asarray(
        [pair in protocol.learned_pairs for pair in geometry.pairs], dtype=bool
    )
    nonlearned = ~learned
    fast_weights = evaluator.learn_fast_weights(FastWeightIntervention.INTACT)
    margins = readout_margin_fields(evaluator, fast_weights, geometry)
    potentials = hodge_potentials(margins, geometry)
    gradient = potentials @ geometry.incidence.T
    residual = margins - gradient
    amplitudes = np.linalg.norm(
        potentials - np.mean(potentials, axis=1, keepdims=True), axis=1
    )
    normalized = normalize_potentials(potentials)
    normalized_gradient = normalized @ geometry.incidence.T

    posterior = exact_prefix_trajectory(
        evaluator,
        protocol,
        geometry,
        temperature=float(evaluation["posterior_temperature"]),
    )
    posterior_field = posterior.fields[-1]
    posterior_normalized = posterior.distributional_potentials[-1]
    posterior_normalized_gradient = posterior_normalized @ geometry.incidence.T
    correct_sign = geometry.true_sign[None, :]
    correct_margin = correct_sign * margins
    correct_gradient = correct_sign * gradient
    correct_residual = correct_sign * residual
    correct_normalized = correct_sign * normalized_gradient
    correct_posterior_normalized = correct_sign * posterior_normalized_gradient
    neural_probability = exact_probability(
        correct_margin, float(evaluation["choice_temperature"])
    )
    posterior_probability = 0.5 * (1.0 + correct_sign * posterior_field)

    beta_m = subject_slopes(correct_margin, distances, nonlearned)
    beta_g = subject_slopes(correct_gradient, distances, nonlearned)
    beta_c = subject_slopes(correct_residual, distances, nonlearned)
    beta_hat = subject_slopes(correct_normalized, distances, nonlearned)
    beta_hat_posterior = subject_slopes(
        correct_posterior_normalized, distances, nonlearned
    )
    beta_p_posterior = subject_slopes(posterior_probability, distances, nonlearned)
    choice = choice_link_components(
        correct_margin, neural_probability, distances, nonlearned
    )
    sensitivity = np.mean(
        neural_probability[:, nonlearned]
        * (1.0 - neural_probability[:, nonlearned])
        / float(evaluation["choice_temperature"]),
        axis=1,
    )
    raw = {
        "beta_m": beta_m,
        "beta_g": beta_g,
        "beta_c": beta_c,
        "beta_g_minus_0_9_beta_m": beta_g - 0.9 * beta_m,
        "potential_amplitude": amplitudes,
        "beta_hat": beta_hat,
        "beta_hat_posterior": beta_hat_posterior,
        "beta_hat_minus_posterior": beta_hat - beta_hat_posterior,
        "potential_cosine_to_posterior": np.sum(
            normalized * posterior_normalized, axis=1
        ),
        "beta_p": choice["beta_probability"],
        "beta_p_posterior": beta_p_posterior,
        "beta_p_minus_posterior": choice["beta_probability"] - beta_p_posterior,
        "choice_kappa": choice["kappa"],
        "choice_mean_local_sensitivity": sensitivity,
        "beta_choice_linearized": choice["beta_linearized"],
        "beta_e": choice["beta_remainder"],
    }
    counts = bootstrap_counts(
        np.random.default_rng(int(evaluation["bootstrap_seeds"][str(seed)])),
        int(evaluation["bootstrap_samples"]),
        model_config.bs,
    )
    interval = float(evaluation["bootstrap_interval"])
    metrics = _summaries(raw, counts, interval)
    tolerance = float(
        specification["exact_decompositions"]["required_numeric_tolerance"]
    )
    integrity = {
        "subjects": model_config.bs,
        "nonlearned_pairs": int(np.sum(nonlearned)),
        "margin_field_max_abs_error": float(
            np.max(np.abs(margins - gradient - residual))
        ),
        "margin_slope_max_abs_error": float(np.max(np.abs(beta_m - beta_g - beta_c))),
        "potential_scale_slope_max_abs_error": float(
            np.max(np.abs(beta_g - amplitudes * beta_hat))
        ),
        "choice_link_slope_max_abs_error": float(
            np.max(
                np.abs(
                    choice["beta_probability"]
                    - choice["beta_linearized"]
                    - choice["beta_remainder"]
                )
            )
        ),
        "expected_rank_equivalence_max_abs_error": float(
            np.max(posterior.expected_rank_equivalence_error[-1])
        ),
    }
    integrity["passed"] = all(
        value <= tolerance
        for name, value in integrity.items()
        if name.endswith("max_abs_error")
    )
    if not integrity["passed"]:
        raise RuntimeError(f"seed {seed} exact decomposition failed: {integrity}")
    return {
        "seed": seed,
        "checkpoint": {
            "path": artifact["path"],
            "sha256": checkpoint.sha256,
        },
        "condition": "pure_L_off_intact_P_T",
        "metrics": metrics,
        "registered_contrasts": {
            "normalized_geometry": summarize_difference(
                beta_hat, beta_hat_posterior, counts, interval=interval
            ),
            "exact_probability_slope": summarize_difference(
                choice["beta_probability"],
                beta_p_posterior,
                counts,
                interval=interval,
            ),
        },
        "integrity": integrity,
        "decision": seed_decision(metrics),
        "raw_subject_level": {name: values.tolist() for name, values in raw.items()},
    }


def _secondary_preservation(specification: dict) -> dict:
    result = load_json(
        _resolve(
            specification["registered_sources"]["v2_4_confirmation_result"]["path"]
        )
    )
    rows = {}
    for seed in specification["network_contract"]["mandatory_frozen_seeds"]:
        condition = result["seeds"][str(seed)]["exact_probability_slope_decomposition"][
            "conditions"
        ]["dual_access_matched"]
        contributions = {
            name: float(summary["mean"])
            for name, summary in condition["group_contributions"].items()
        }
        rows[str(seed)] = {
            "dual_intact_exact_probability_slope": float(condition["total"]["mean"]),
            "group_contributions": contributions,
            "nonlearned_fraction_of_total": contributions["nonlearned"]
            / float(condition["total"]["mean"]),
            "nonlearned_is_largest_positive_contribution": contributions["nonlearned"]
            == max(contributions.values()),
        }
    return {"decision_role": "reported_only_not_a_primary_gate", "seeds": rows}


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
        "cross_seed_decision": cross_seed_decision(seeds),
        "secondary_v2_4_preservation": _secondary_preservation(specification),
    }


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Run the frozen read-only global-policy slope localization."
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
    write_json(parsed.result, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
