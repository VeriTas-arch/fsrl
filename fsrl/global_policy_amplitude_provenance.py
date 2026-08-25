"""Read-only amplitude provenance for the frozen global policy."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .assembly_trajectory import (
    bootstrap_counts,
    build_complete_graph_geometry,
    hodge_potentials,
    ordered_query_schedule,
)
from .constructive import ExactRankingPosterior, RelationEvidence
from .curvature_gate_pilot import _tensor_hashes, load_json, write_json
from .dual_evidence_access_confirmation import validate_artifacts
from .formal_runtime import require_formal_runtime
from .global_policy_slope_localization import subject_slopes
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
    resolve_record("benchmarks/global_policy_amplitude_provenance_v1.json")
)
DEFAULT_IMPLEMENTATION_LOCK_PATH = (
    resolve_record("benchmarks/global_policy_amplitude_provenance_v1.lock.json")
)
DEFAULT_RESULT_PATH = resolve_record("results/global_policy_amplitude_provenance_v1.json")
CONFIRMATION_OUTPUT_ROOT = ROOT / "output" / "dual-evidence-access-confirmation-v2-4"


class NonInterpretableEstimate(RuntimeError):
    """Raised when a frozen estimand is undefined without filtering rows or draws."""


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else resolve_record(candidate)


def _max_abs_or_none(values: np.ndarray) -> float | None:
    rows = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(rows)):
        return None
    return float(np.max(np.abs(rows)))


def vector_item_potentials(fields: np.ndarray, geometry) -> np.ndarray:
    values = np.asarray(fields, dtype=np.float64)
    operator = np.asarray(geometry.score_operator, dtype=np.float64)
    if values.ndim != 3 or values.shape[1] != operator.shape[1]:
        raise ValueError("vector fields must have shape (subjects, edges, features)")
    return np.einsum("ie,seh->sih", operator, values)


def ols_slope(
    predictor: np.ndarray,
    response: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    response = np.asarray(response, dtype=np.float64)
    predictor = np.asarray(predictor, dtype=np.float64)
    if response.shape != predictor.shape or response.ndim != 1:
        raise ValueError("OLS inputs must be matching one-dimensional arrays")
    if weights is None:
        current_weights = np.ones_like(predictor)
    else:
        current_weights = np.asarray(weights, dtype=np.float64)
        if current_weights.shape != predictor.shape or np.any(current_weights < 0.0):
            raise ValueError("OLS weights must be matching and nonnegative")
    total = float(np.sum(current_weights))
    if total <= 0.0:
        raise ValueError("OLS weights must have positive total")
    mean_x = float(current_weights @ predictor / total)
    mean_y = float(current_weights @ response / total)
    centered_x = predictor - mean_x
    denominator = float(current_weights @ (centered_x * centered_x))
    if denominator <= 0.0:
        raise NonInterpretableEstimate("OLS predictor has zero variance")
    return float(current_weights @ (centered_x * (response - mean_y)) / denominator)


def _weighted_moments(
    first: np.ndarray, second: np.ndarray, counts: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(first, dtype=np.float64)
    y = np.asarray(second, dtype=np.float64)
    weights = np.asarray(counts, dtype=np.float64)
    totals = np.sum(weights, axis=1)
    if np.any(totals <= 0.0):
        raise ValueError("every bootstrap draw must retain positive total weight")
    mean_x = weights @ x / totals
    mean_y = weights @ y / totals
    centered_x = x[None, :] - mean_x[:, None]
    centered_y = y[None, :] - mean_y[:, None]
    covariance = np.sum(weights * centered_x * centered_y, axis=1) / totals
    variance_x = np.sum(weights * centered_x * centered_x, axis=1) / totals
    variance_y = np.sum(weights * centered_y * centered_y, axis=1) / totals
    return covariance, variance_x, variance_y


def bootstrap_ols(y: np.ndarray, x: np.ndarray, counts: np.ndarray) -> np.ndarray:
    covariance, variance_x, _variance_y = _weighted_moments(x, y, counts)
    return np.divide(
        covariance,
        variance_x,
        out=np.full_like(covariance, np.nan),
        where=variance_x > 0.0,
    )


def bootstrap_ols_slopes(
    predictor: np.ndarray,
    response: np.ndarray,
    bootstrap_counts: np.ndarray,
) -> np.ndarray:
    return bootstrap_ols(response, predictor, bootstrap_counts)


def amplitude_ledger(
    *,
    a_p: np.ndarray,
    a_h: np.ndarray,
    a_delta: np.ndarray,
    a_n: np.ndarray,
    w_norm: float,
) -> dict[str, np.ndarray]:
    values = tuple(
        np.asarray(row, dtype=np.float64) for row in (a_p, a_h, a_delta, a_n)
    )
    if len({row.shape for row in values}) != 1 or any(
        np.any(~np.isfinite(row)) or np.any(row < 0.0) for row in values
    ):
        raise ValueError(
            "amplitude ledger requires matching finite nonnegative amplitudes"
        )
    if w_norm <= 0.0:
        raise ValueError("readout norm must be positive")
    current_p, current_h, current_delta, current_n = values
    g_rec = np.divide(
        current_h,
        current_p,
        out=np.full_like(current_h, np.nan),
        where=current_p > 0.0,
    )
    g_out = np.divide(
        current_delta,
        current_h,
        out=np.full_like(current_delta, np.nan),
        where=current_h > 0.0,
    )
    g_mix = np.divide(
        current_n,
        current_delta,
        out=np.full_like(current_n, np.nan),
        where=current_delta > 0.0,
    )
    return {
        "g_rec": g_rec,
        "g_out": g_out,
        "g_mix": g_mix,
        "rho_w": g_out / float(w_norm),
    }


def elasticity_ledger(
    *,
    log_a_post: np.ndarray,
    log_a_p: np.ndarray,
    log_a_h: np.ndarray,
    log_a_delta: np.ndarray,
    log_a_n: np.ndarray,
) -> dict[str, float]:
    beta_p = ols_slope(log_a_post, log_a_p)
    beta_h = ols_slope(log_a_post, log_a_h)
    beta_delta = ols_slope(log_a_post, log_a_delta)
    beta_n = ols_slope(log_a_post, log_a_n)
    return {
        "beta_p": beta_p,
        "beta_h": beta_h,
        "beta_delta": beta_delta,
        "beta_n": beta_n,
        "delta_rec": beta_h - beta_p,
        "delta_out": beta_delta - beta_h,
        "delta_mix": beta_n - beta_delta,
        "beta_mismatch": ols_slope(log_a_post, log_a_n - log_a_post),
    }


def cross_network_outcome(by_network: dict[str, str]) -> dict:
    outcomes = set(by_network.values())
    outcome = (
        next(iter(outcomes)) if len(outcomes) == 1 else "heterogeneous_or_unresolved"
    )
    return {
        "outcome": outcome,
        "by_network": dict(by_network),
        "network_population_inference": "not_performed",
    }


def bootstrap_correlation(
    first: np.ndarray, second: np.ndarray, counts: np.ndarray
) -> np.ndarray:
    covariance, variance_first, variance_second = _weighted_moments(
        first, second, counts
    )
    denominator = np.sqrt(variance_first * variance_second)
    return np.divide(
        covariance,
        denominator,
        out=np.full_like(covariance, np.nan),
        where=(variance_first > 0.0) & (variance_second > 0.0),
    )


def _interval_summary(point: float, samples: np.ndarray) -> dict:
    finite = np.asarray(samples, dtype=np.float64)
    if finite.ndim != 1 or len(finite) == 0:
        raise NonInterpretableEstimate("bootstrap estimates must be a nonempty vector")
    if not np.isfinite(point) or not np.all(np.isfinite(finite)):
        raise NonInterpretableEstimate(
            "all subject rows and bootstrap estimates must remain finite"
        )
    return {
        "point": float(point),
        "bootstrap": {
            "mean": float(np.mean(finite)),
            "lower95": float(np.quantile(finite, 0.025)),
            "upper95": float(np.quantile(finite, 0.975)),
            "lower90": float(np.quantile(finite, 0.05)),
            "upper90": float(np.quantile(finite, 0.95)),
            "finite_samples": len(finite),
        },
    }


def summarize_ols(y: np.ndarray, x: np.ndarray, counts: np.ndarray) -> dict:
    slopes = bootstrap_ols(y, x, counts)
    correlations = bootstrap_correlation(x, y, counts)
    return {
        "slope": _interval_summary(ols_slope(x, y), slopes),
        "correlation": _interval_summary(
            float(np.corrcoef(np.asarray(x), np.asarray(y))[0, 1]), correlations
        ),
    }


def through_origin_shape_fit(
    posterior: np.ndarray, neural: np.ndarray, counts: np.ndarray
) -> dict:
    source = np.asarray(posterior, dtype=np.float64)
    target = np.asarray(neural, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2:
        raise ValueError("shape fit requires matching subject-by-item potentials")
    cross = np.sum(source * target, axis=1)
    source_energy = np.sum(source * source, axis=1)
    target_energy = np.sum(target * target, axis=1)
    total_counts = np.sum(counts, axis=1)
    numerator = counts @ cross
    denominator = counts @ source_energy
    scale_samples = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan),
        where=denominator > 0.0,
    )
    residual_energy = (
        counts @ target_energy
        - 2.0 * scale_samples * numerator
        + scale_samples * scale_samples * denominator
    )
    target_total = counts @ target_energy
    explained_samples = np.divide(
        residual_energy,
        target_total,
        out=np.full_like(residual_energy, np.nan),
        where=target_total > 0.0,
    )
    explained_samples = 1.0 - explained_samples
    scale = float(np.sum(cross) / np.sum(source_energy))
    residual = target - scale * source
    explained = 1.0 - float(np.sum(residual * residual) / np.sum(target * target))
    cosine = np.divide(
        cross,
        np.linalg.norm(source, axis=1) * np.linalg.norm(target, axis=1),
        out=np.full(len(source), np.nan, dtype=np.float64),
        where=(np.linalg.norm(source, axis=1) * np.linalg.norm(target, axis=1)) > 0.0,
    )
    mean_cosine_samples = counts @ cosine / total_counts
    return {
        "scale": _interval_summary(scale, scale_samples),
        "energy_explained": _interval_summary(explained, explained_samples),
        "residual_energy": _interval_summary(
            float(np.sum(residual * residual)), residual_energy
        ),
        "normalized_cosine": _interval_summary(
            float(np.mean(cosine)), mean_cosine_samples
        ),
    }


def _source_validation(
    specification: dict, specification_path: Path, implementation_lock: dict
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
        raise RuntimeError(f"amplitude-provenance source lock failed: {checks}")
    return {"passed": True, "checks": checks}


def _artifact_validation(specification: dict) -> dict:
    sources = specification["registered_sources"]
    confirmation_path = _resolve(sources["v2_4_confirmation_specification"]["path"])
    confirmation = load_json(confirmation_path)
    return validate_artifacts(
        confirmation,
        confirmation_path,
        _resolve(sources["v2_4_confirmation_implementation_lock"]["path"]),
        _resolve(sources["v2_4_confirmation_artifact_lock"]["path"]),
        CONFIRMATION_OUTPUT_ROOT,
    )


def _pack_trajectories(rows, pairs, step: int) -> np.ndarray:
    subjects = len(rows)
    first = next(iter(rows[0].values()))
    trailing = first[step].shape if np.ndim(first[step]) else ()
    packed = np.empty((subjects, len(pairs), 2, *trailing), dtype=np.float64)
    for subject, subject_rows in enumerate(rows):
        for edge, pair in enumerate(pairs):
            packed[subject, edge, 0] = subject_rows[pair][step]
            packed[subject, edge, 1] = subject_rows[(pair[1], pair[0])][step]
    return packed


def _neural_layers(evaluator, fast_weights, geometry) -> tuple[dict, dict]:
    schedules = ordered_query_schedule(geometry, evaluator.config.bs)
    intact_hidden_rows, intact_logit_rows = (
        evaluator.readout_hidden_and_logit_trajectories(fast_weights, schedules)
    )
    zero_weights = torch.zeros_like(fast_weights)
    zero_hidden_rows, zero_logit_rows = evaluator.readout_hidden_and_logit_trajectories(
        zero_weights, schedules
    )
    intact_h0 = _pack_trajectories(intact_hidden_rows, geometry.pairs, 0)
    zero_h0 = _pack_trajectories(zero_hidden_rows, geometry.pairs, 0)
    intact_h1 = _pack_trajectories(intact_hidden_rows, geometry.pairs, 1)
    zero_h1 = _pack_trajectories(zero_hidden_rows, geometry.pairs, 1)
    intact_logits = _pack_trajectories(intact_logit_rows, geometry.pairs, 1)
    zero_logits = _pack_trajectories(zero_logit_rows, geometry.pairs, 1)
    subjects, _edges, _orientations, hidden_size = intact_h0.shape
    effective = evaluator.net.alpha.detach()[None] * fast_weights
    h0_tensor = torch.from_numpy(intact_h0.astype(np.float32)).to(fast_weights.device)
    drive = torch.einsum("sij,seoj->seoi", effective, h0_tensor)

    manual_h0 = torch.empty_like(h0_tensor)
    baseline = torch.empty_like(h0_tensor)
    with torch.no_grad():
        for edge, pair in enumerate(geometry.pairs):
            for orientation, oriented in enumerate((pair, (pair[1], pair[0]))):
                left = np.full(subjects, oriented[0], dtype=np.int64)
                right = np.full(subjects, oriented[1], dtype=np.int64)
                signed = np.zeros(subjects, dtype=np.float32)
                x0 = evaluator._step_inputs(
                    left,
                    right,
                    signed,
                    numstep=0,
                    time_value=evaluator.test_time_value,
                    support_trial=False,
                )
                x1 = evaluator._step_inputs(
                    left,
                    right,
                    signed,
                    numstep=1,
                    time_value=evaluator.test_time_value,
                    support_trial=False,
                )
                current_h0 = evaluator.net.activ(evaluator.net.i2h(x0))
                manual_h0[:, edge, orientation] = current_h0
                baseline[:, edge, orientation] = evaluator.net.i2h(x1) + torch.matmul(
                    evaluator.net.w,
                    current_h0.view(subjects, hidden_size, 1),
                ).view(subjects, hidden_size)
    manual_intact = torch.tanh(baseline + drive)
    manual_zero = torch.tanh(baseline)
    drive_np = drive.detach().cpu().numpy().astype(np.float64)
    manual_delta = (
        (manual_intact - manual_zero).detach().cpu().numpy().astype(np.float64)
    )

    antisym_drive = 0.5 * (drive_np[:, :, 0] - drive_np[:, :, 1])
    hidden_intact = 0.5 * (intact_h1[:, :, 0] - intact_h1[:, :, 1])
    hidden_zero = 0.5 * (zero_h1[:, :, 0] - zero_h1[:, :, 1])
    hidden_delta = hidden_intact - hidden_zero
    margin_intact = 0.5 * (intact_logits[:, :, 0] - intact_logits[:, :, 1])
    margin_zero = 0.5 * (zero_logits[:, :, 0] - zero_logits[:, :, 1])
    U = vector_item_potentials(antisym_drive, geometry)
    V_intact = vector_item_potentials(hidden_intact, geometry)
    V_zero = vector_item_potentials(hidden_zero, geometry)
    V_delta = V_intact - V_zero
    output = (
        (evaluator.net.h2o.weight[1] - evaluator.net.h2o.weight[0])
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    s_intact = np.einsum("sih,h->si", V_intact, output)
    s_zero = np.einsum("sih,h->si", V_zero, output)
    s_delta = np.einsum("sih,h->si", V_delta, output)
    a_P = np.linalg.norm(U, axis=(1, 2))
    a_H = np.linalg.norm(V_delta, axis=(1, 2))
    a_delta = np.linalg.norm(s_delta, axis=1)
    a_N = np.linalg.norm(s_intact, axis=1)
    a_0 = np.linalg.norm(s_zero, axis=1)
    w_norm = float(np.linalg.norm(output))
    ledger = amplitude_ledger(a_p=a_P, a_h=a_H, a_delta=a_delta, a_n=a_N, w_norm=w_norm)
    g_rec = ledger["g_rec"]
    g_out = ledger["g_out"]
    g_mix = ledger["g_mix"]
    rho = ledger["rho_w"]
    denominator = np.sum(s_intact * s_intact, axis=1)
    phi_P = np.divide(
        np.sum(s_intact * s_delta, axis=1),
        denominator,
        out=np.full_like(denominator, np.nan),
        where=denominator > 0.0,
    )
    phi_0 = np.divide(
        np.sum(s_intact * s_zero, axis=1),
        denominator,
        out=np.full_like(denominator, np.nan),
        where=denominator > 0.0,
    )
    raw_oriented_drive = np.sqrt(
        np.mean(np.sum(drive_np * drive_np, axis=3), axis=(1, 2))
    )

    projected_drive = np.einsum("ei,sih->seh", geometry.incidence, U)
    projected_intact = np.einsum("ei,sih->seh", geometry.incidence, V_intact)
    projected_zero = np.einsum("ei,sih->seh", geometry.incidence, V_zero)
    margin_from_hidden = np.einsum("seh,h->se", hidden_intact, output)
    s_from_margin = hodge_potentials(margin_intact, geometry)
    a_N_from_margin = np.linalg.norm(s_from_margin, axis=1)
    cross_term = 2.0 * np.sum(s_zero * s_delta, axis=1)
    integrity = {
        "step0_intact_Poff_max_abs_error": float(np.max(np.abs(intact_h0 - zero_h0))),
        "manual_h0_max_abs_error": float(
            np.max(np.abs(intact_h0 - manual_h0.detach().cpu().numpy()))
        ),
        "manual_intact_response_max_abs_error": float(
            np.max(np.abs(intact_h1 - manual_intact.detach().cpu().numpy()))
        ),
        "manual_Poff_response_max_abs_error": float(
            np.max(np.abs(zero_h1 - manual_zero.detach().cpu().numpy()))
        ),
        "manual_delta_response_max_abs_error": float(
            np.max(np.abs((intact_h1 - zero_h1) - manual_delta))
        ),
        "hidden_to_logit_max_abs_error": float(
            np.max(np.abs(margin_intact - margin_from_hidden))
        ),
        "Poff_hidden_to_logit_max_abs_error": float(
            np.max(np.abs(margin_zero - np.einsum("seh,h->se", hidden_zero, output)))
        ),
        "vector_Hodge_gauge_max_abs_error": float(
            max(
                np.max(np.abs(np.sum(U, axis=1))),
                np.max(np.abs(np.sum(V_intact, axis=1))),
                np.max(np.abs(np.sum(V_zero, axis=1))),
            )
        ),
        "vector_Hodge_orthogonality_max_abs_error": float(
            max(
                np.max(
                    np.abs(
                        np.sum(
                            projected_drive * (antisym_drive - projected_drive), axis=1
                        )
                    )
                ),
                np.max(
                    np.abs(
                        np.sum(
                            projected_intact * (hidden_intact - projected_intact),
                            axis=1,
                        )
                    )
                ),
                np.max(
                    np.abs(
                        np.sum(projected_zero * (hidden_zero - projected_zero), axis=1)
                    )
                ),
            )
        ),
        "Hodge_readout_commutation_max_abs_error": float(
            np.max(np.abs(s_from_margin - s_intact))
        ),
        "hidden_margin_amplitude_bridge_max_abs_error": float(
            np.max(np.abs(a_N_from_margin - a_N))
        ),
        "P_subtraction_hidden_potential_max_abs_error": float(
            np.max(np.abs(V_delta - vector_item_potentials(hidden_delta, geometry)))
        ),
        "P_subtraction_policy_potential_max_abs_error": float(
            np.max(np.abs(s_intact - s_zero - s_delta))
        ),
        "squared_norm_cross_term_max_abs_error": float(
            np.max(np.abs(a_N * a_N - a_0 * a_0 - a_delta * a_delta - cross_term))
        ),
        "amplitude_ledger_max_abs_error": _max_abs_or_none(
            a_N - a_P * g_rec * g_out * g_mix
        ),
        "output_alignment_max_abs_error": _max_abs_or_none(g_out - w_norm * rho),
        "phi_sum_max_abs_error": _max_abs_or_none(phi_P + phi_0 - 1.0),
        "rho_min": float(np.min(rho)) if np.all(np.isfinite(rho)) else None,
        "rho_max": float(np.max(rho)) if np.all(np.isfinite(rho)) else None,
        "minimum_required_amplitude": float(
            np.min(np.stack((a_P, a_H, a_delta, a_N), axis=1))
        ),
    }
    arrays = {
        "a_P": a_P,
        "raw_oriented_drive": raw_oriented_drive,
        "a_H": a_H,
        "a_delta": a_delta,
        "a_N": a_N,
        "a_N_from_margin": a_N_from_margin,
        "a_0": a_0,
        "g_rec": g_rec,
        "g_out": g_out,
        "g_mix": g_mix,
        "rho_W": rho,
        "phi_P": phi_P,
        "phi_0": phi_0,
        "cross_term": cross_term,
    }
    fields = {
        "margin_intact": margin_intact,
        "margin_Poff": margin_zero,
        "s_intact": s_intact,
        "s_intact_from_margin": s_from_margin,
        "s_Poff": s_zero,
        "s_P": s_delta,
    }
    return {"arrays": arrays, "fields": fields, "w_norm": w_norm}, integrity


def _logsumexp(values: np.ndarray) -> float:
    maximum = float(np.max(values))
    return maximum + float(np.log(np.sum(np.exp(values - maximum))))


def _stable_sigmoid(value: float) -> float:
    if value >= 0.0:
        inverse = np.exp(-value)
        return float(1.0 / (1.0 + inverse))
    exponential = np.exp(value)
    return float(exponential / (1.0 + exponential))


def _posterior_descriptors(
    evaluator, geometry, specification: dict
) -> tuple[dict, dict]:
    contract = specification["posterior_comparator"]
    posterior_temperature = float(contract["posterior_temperature"])
    choice_temperature = float(contract["choice_temperature"])
    exact = ExactRankingPosterior(
        evaluator.protocol.n_items, temperature=posterior_temperature
    )
    pair_masks = tuple(
        exact.positions[:, first] < exact.positions[:, second]
        for first, second in geometry.pairs
    )
    fields = []
    margins = []
    expected_rank = []
    entropy = []
    coverage = []
    inverse_link_error = 0.0
    reversal_error = 0.0
    coverage_binary_error = 0.0
    coverage_reuse_error = 0.0
    coverage_fraction_error = 0.0
    unique_relation_counts = []
    for rows in evaluator.realized_support_evidence():
        evidence = tuple(
            RelationEvidence(
                higher_item=int(row["higher_item"]),
                lower_item=int(row["lower_item"]),
                magnitude=float(row["magnitude"]),
                reliability=float(row["reliability"]),
            )
            for row in rows
        )
        state = exact.fit(evidence)
        log_weights = -state.energy / posterior_temperature
        current_field = []
        current_margin = []
        for mask in pair_masks:
            probability = float(np.sum(state.probabilities[mask]))
            margin = choice_temperature * (
                _logsumexp(log_weights[mask]) - _logsumexp(log_weights[~mask])
            )
            inverse = _stable_sigmoid(margin / choice_temperature)
            inverse_link_error = max(inverse_link_error, abs(inverse - probability))
            reverse_margin = choice_temperature * (
                _logsumexp(log_weights[~mask]) - _logsumexp(log_weights[mask])
            )
            reversal_error = max(reversal_error, abs(margin + reverse_margin))
            current_field.append(2.0 * probability - 1.0)
            current_margin.append(margin)
        fields.append(current_field)
        margins.append(current_margin)
        expected_positions = state.probabilities @ exact.positions.astype(np.float64)
        expected_rank.append(
            (evaluator.protocol.n_items - 1) / 2.0 - expected_positions
        )
        entropy.append(exact.posterior_entropy(state))
        relation_values = {}
        for row in rows:
            relation = (row["higher_item"], row["lower_item"])
            relation_values.setdefault(relation, []).append(float(row["reliability"]))
        unique_relation_counts.append(len(relation_values))
        relation_reliability = {
            relation: values[0] for relation, values in relation_values.items()
        }
        coverage_binary_error = max(
            coverage_binary_error,
            max(
                min(abs(value), abs(value - 1.0))
                for value in relation_reliability.values()
            ),
        )
        coverage_reuse_error = max(
            coverage_reuse_error,
            max(
                max(abs(value - values[0]) for value in values)
                for values in relation_values.values()
            ),
        )
        unique_values = np.asarray(tuple(relation_reliability.values()))
        current_coverage = float(np.sum(unique_values) / evaluator.protocol.n_items)
        coverage_fraction_error = max(
            coverage_fraction_error,
            abs(current_coverage - float(np.mean(unique_values))),
        )
        coverage.append(current_coverage)
    field = np.asarray(fields, dtype=np.float64)
    margin = np.asarray(margins, dtype=np.float64)
    rank = np.asarray(expected_rank, dtype=np.float64)
    normalized_entropy = np.asarray(entropy, dtype=np.float64) / np.log(
        exact.n_hypotheses
    )
    s_post = hodge_potentials(margin, geometry)
    s_probability = hodge_potentials(field, geometry)
    commitment_denominator = np.sqrt(
        evaluator.protocol.n_items
        * (evaluator.protocol.n_items * evaluator.protocol.n_items - 1)
        / 12.0
    )
    arrays = {
        "a_post": np.linalg.norm(s_post, axis=1),
        "posterior_commitment": np.linalg.norm(rank, axis=1) / commitment_denominator,
        "posterior_entropy": normalized_entropy,
        "posterior_certainty": 1.0 - normalized_entropy,
        "coverage": np.asarray(coverage, dtype=np.float64),
    }
    integrity = {
        "posterior_inverse_link_max_abs_error": float(inverse_link_error),
        "posterior_orientation_reversal_max_abs_error": float(reversal_error),
        "posterior_expected_rank_Hodge_max_abs_error": float(
            np.max(np.abs(s_probability - (2.0 / evaluator.protocol.n_items) * rank))
        ),
        "posterior_entropy_min": float(np.min(normalized_entropy)),
        "posterior_entropy_max": float(np.max(normalized_entropy)),
        "coverage_min": float(np.min(arrays["coverage"])),
        "coverage_max": float(np.max(arrays["coverage"])),
        "coverage_binary_max_abs_error": float(coverage_binary_error),
        "coverage_relation_reuse_max_abs_error": float(coverage_reuse_error),
        "coverage_unique_fraction_max_abs_error": float(coverage_fraction_error),
        "coverage_unique_relations_min": int(min(unique_relation_counts)),
        "coverage_unique_relations_max": int(max(unique_relation_counts)),
    }
    fields = {
        "pair_probability_field": field,
        "same_unit_margin": margin,
        "s_post": s_post,
        "s_probability": s_probability,
    }
    return {"arrays": arrays, "fields": fields}, integrity


def _fit_status(summary: dict, *, target: float) -> str:
    bootstrap = summary["bootstrap"]
    lower95 = float(bootstrap["lower95"]) - target
    upper95 = float(bootstrap["upper95"]) - target
    lower90 = float(bootstrap["lower90"]) - target
    upper90 = float(bootstrap["upper90"]) - target
    if lower95 > 0.10:
        return "material_positive"
    if upper95 < -0.10:
        return "material_negative"
    if lower90 >= -0.10 and upper90 <= 0.10:
        return "equivalent"
    return "unresolved"


def _external_status(summary: dict) -> str:
    bound = float(np.log(1.10))
    bootstrap = summary["bootstrap"]
    if float(bootstrap["lower90"]) >= -bound and float(bootstrap["upper90"]) <= bound:
        return "equivalent"
    if float(bootstrap["lower95"]) > 0.0:
        return "positive"
    if float(bootstrap["upper95"]) < 0.0:
        return "negative"
    return "unresolved"


def _mean_summary(values: np.ndarray, counts: np.ndarray) -> dict:
    rows = np.asarray(values, dtype=np.float64)
    if rows.ndim != 1 or rows.shape[0] != counts.shape[1]:
        raise NonInterpretableEstimate("one scalar is required for every participant")
    if not np.all(np.isfinite(rows)):
        raise NonInterpretableEstimate("participant metrics must all be finite")
    samples = counts @ rows / np.sum(counts, axis=1)
    interval = _interval_summary(float(np.mean(rows)), samples)
    return {
        "subjects": len(rows),
        "mean": float(np.mean(rows)),
        "median": float(np.median(rows)),
        "lower_quartile": float(np.quantile(rows, 0.25)),
        "upper_quartile": float(np.quantile(rows, 0.75)),
        "bootstrap": {
            "mean": interval["bootstrap"]["mean"],
            "lower": interval["bootstrap"]["lower95"],
            "upper": interval["bootstrap"]["upper95"],
            "finite_samples": interval["bootstrap"]["finite_samples"],
        },
    }


def _seed_statistics(
    specification: dict,
    seed: int,
    neural: dict,
    posterior: dict,
    track_b: dict,
    geometry,
) -> dict:
    bootstrap = specification["statistical_estimands"]["bootstrap"]
    counts = bootstrap_counts(
        np.random.default_rng(int(bootstrap["seeds"][str(seed)])),
        int(bootstrap["samples"]),
        len(neural["arrays"]["a_N"]),
    )
    arrays = {**neural["arrays"], **posterior["arrays"]}
    for name in ("a_P", "a_H", "a_delta", "a_N", "a_post"):
        if np.any(arrays[name] <= 1e-12):
            raise NonInterpretableEstimate(
                f"seed {seed} has a registered zero denominator in {name}"
            )
        if not np.all(np.isfinite(arrays[name])):
            raise NonInterpretableEstimate(f"seed {seed} has nonfinite {name}")
    logs = {
        "log_a_P": np.log(arrays["a_P"]),
        "log_a_H": np.log(arrays["a_H"]),
        "log_a_delta": np.log(arrays["a_delta"]),
        "log_a_N": np.log(arrays["a_N"]),
        "log_a_post": np.log(arrays["a_post"]),
    }
    Y = logs["log_a_N"] - logs["log_a_post"]
    arrays["Y"] = Y
    temperature = float(specification["posterior_comparator"]["choice_temperature"])
    neural_probability_field = (
        2.0 * exact_probability(neural["fields"]["margin_intact"], temperature) - 1.0
    )
    d_prob = np.linalg.norm(
        hodge_potentials(neural_probability_field, geometry),
        axis=1,
    ) - np.linalg.norm(posterior["fields"]["s_probability"], axis=1)
    arrays["d_prob"] = d_prob
    slopes = {}
    slope_samples = {}
    Z = logs["log_a_post"]
    for name in ("P", "H", "delta", "N"):
        values = logs[f"log_a_{name}"]
        samples = bootstrap_ols(values, Z, counts)
        slope_samples[f"beta_{name}"] = samples
        slopes[f"beta_{name}"] = _interval_summary(ols_slope(Z, values), samples)
    increment_definitions = {
        "delta_rec": ("beta_H", "beta_P"),
        "delta_out": ("beta_delta", "beta_H"),
        "delta_mix": ("beta_N", "beta_delta"),
    }
    for name, (first, second) in increment_definitions.items():
        samples = slope_samples[first] - slope_samples[second]
        point = slopes[first]["point"] - slopes[second]["point"]
        slopes[name] = _interval_summary(point, samples)
        slope_samples[name] = samples
    direct_mismatch_samples = bootstrap_ols(Y, Z, counts)
    slopes["beta_Y_direct"] = _interval_summary(
        ols_slope(Z, Y), direct_mismatch_samples
    )
    slopes["beta_N_minus_1"] = _interval_summary(
        slopes["beta_N"]["point"] - 1.0,
        slope_samples["beta_N"] - 1.0,
    )
    external = {}
    for predictor_name in ("coverage", "posterior_certainty"):
        predictor = arrays[predictor_name]
        external[predictor_name] = {
            "Y": summarize_ols(Y, predictor, counts),
            **{
                name: summarize_ols(values, predictor, counts)
                for name, values in logs.items()
                if name != "log_a_post"
            },
        }
    external["coverage_certainty_association"] = summarize_ols(
        arrays["posterior_certainty"], arrays["coverage"], counts
    )
    raw_track_mismatch = np.asarray(
        track_b["raw_subject_level"]["beta_p_minus_posterior"], dtype=np.float64
    )
    if not np.all(np.isfinite(raw_track_mismatch)):
        raise NonInterpretableEstimate("Track-B participant rows must all be finite")
    shape = through_origin_shape_fit(
        posterior["fields"]["s_post"], neural["fields"]["s_intact"], counts
    )
    metric_names = (
        "a_P",
        "raw_oriented_drive",
        "a_H",
        "a_delta",
        "a_N",
        "a_0",
        "g_rec",
        "g_out",
        "g_mix",
        "rho_W",
        "phi_P",
        "phi_0",
        "cross_term",
        "a_post",
        "posterior_commitment",
        "posterior_entropy",
        "posterior_certainty",
        "coverage",
        "Y",
        "d_prob",
    )
    metrics = {name: _mean_summary(arrays[name], counts) for name in metric_names}
    metrics["Track_B_probability_slope_mismatch"] = _mean_summary(
        raw_track_mismatch, counts
    )
    direct_identity = float(
        abs(slopes["beta_Y_direct"]["point"] - slopes["beta_N_minus_1"]["point"])
    )
    increment_identity = float(
        abs(
            slopes["beta_N"]["point"]
            - slopes["beta_P"]["point"]
            - slopes["delta_rec"]["point"]
            - slopes["delta_out"]["point"]
            - slopes["delta_mix"]["point"]
        )
    )
    statuses = {
        name: _fit_status(summary, target=1.0 if name.startswith("beta_") else 0.0)
        for name, summary in slopes.items()
        if name in {"beta_P", "beta_H", "beta_delta", "beta_N"}
    }
    statuses.update(
        {
            name: _fit_status(summary, target=0.0)
            for name, summary in slopes.items()
            if name.startswith("delta_")
        }
    )
    statuses["Y_on_coverage"] = _external_status(external["coverage"]["Y"]["slope"])
    statuses["Y_on_certainty"] = _external_status(
        external["posterior_certainty"]["Y"]["slope"]
    )
    return {
        "metrics": metrics,
        "elasticities": slopes,
        "external_tracking": external,
        "shape_model": shape,
        "statuses": statuses,
        "integrity": {
            "elasticity_increment_identity_abs_error": increment_identity,
            "direct_mismatch_slope_identity_abs_error": direct_identity,
            "bootstrap_samples": int(counts.shape[0]),
            "bootstrap_subjects": int(counts.shape[1]),
        },
        "raw_subject_level": {
            **{name: values.tolist() for name, values in arrays.items()},
            **{name: values.tolist() for name, values in logs.items()},
            "Track_B_probability_slope_mismatch": raw_track_mismatch.tolist(),
        },
    }


def _material_direction(status: str) -> str | None:
    if status == "material_positive":
        return "positive"
    if status == "material_negative":
        return "negative"
    return None


def layer_elasticity_decision(statuses_by_network: dict[str, dict[str, str]]) -> dict:
    atomic_terms = {
        "beta_P": "functional_drive_fingerprint",
        "delta_rec": "recurrent_expression_fingerprint",
        "delta_out": "projection_alignment_fingerprint",
        "delta_mix": "P_independent_mixing_fingerprint",
    }
    replicated_material = []
    for term in atomic_terms:
        directions = [
            _material_direction(statuses[term])
            for statuses in statuses_by_network.values()
        ]
        if directions and directions[0] is not None and len(set(directions)) == 1:
            replicated_material.append({"term": term, "direction": directions[0]})

    selected = []
    for term, outcome in atomic_terms.items():
        material = next(
            (row for row in replicated_material if row["term"] == term), None
        )
        if material is not None and all(
            statuses[other] == "equivalent"
            for statuses in statuses_by_network.values()
            for other in atomic_terms
            if other != term
        ):
            selected.append({"outcome": outcome, "direction": material["direction"]})
    if len(selected) == 1:
        return selected[0]
    if len(replicated_material) >= 2:
        return {
            "outcome": "coadapted_scale_fingerprint",
            "material_atomic_terms": replicated_material,
        }
    return {"outcome": "heterogeneous_or_unresolved"}


def _gated_decision(outcome: str) -> dict:
    return {
        "outcome": outcome,
        "final_comparator_fingerprint": "not_evaluated",
        "layer_elasticity_fingerprint": "not_evaluated",
        "network_population_inference": "not_performed",
    }


def cross_seed_decision(seeds: dict[str, dict]) -> dict:
    rows = tuple(seeds.values())
    if not all(row["integrity"]["passed"] for row in rows):
        return _gated_decision("noninterpretable_integrity_failure")
    if not all(
        row["statistics"]["metrics"]["Track_B_probability_slope_mismatch"]["bootstrap"][
            "lower"
        ]
        > 0.0
        for row in rows
    ):
        return _gated_decision("premise_not_reproduced")
    if not all(
        row["statistics"]["metrics"][name]["bootstrap"]["lower"] > 0.0
        for row in rows
        for name in ("Y", "d_prob")
    ):
        return _gated_decision("comparator_sensitive_unresolved")
    shape_good = all(
        row["statistics"]["shape_model"]["energy_explained"]["bootstrap"]["lower95"]
        >= 0.90
        for row in rows
    )
    if not shape_good:
        return {
            "outcome": "heterogeneous_or_unresolved",
            "final_comparator_fingerprint": "scalar_shape_inadequate",
            "layer_elasticity_fingerprint": "not_interpreted_shape_gate",
            "network_population_inference": "not_performed",
        }
    constant = all(
        row["statistics"]["shape_model"]["scale"]["bootstrap"]["lower95"] > 1.0
        and row["statistics"]["statuses"]["beta_N"] == "equivalent"
        and row["statistics"]["statuses"]["Y_on_coverage"] == "equivalent"
        and row["statistics"]["statuses"]["Y_on_certainty"] == "equivalent"
        for row in rows
    )
    final_comparator = (
        "constant_calibration_fingerprint"
        if constant
        else "posterior_dependent_or_nonconstant_mismatch"
    )
    layer = layer_elasticity_decision(
        {seed: row["statistics"]["statuses"] for seed, row in seeds.items()}
    )
    layer_outcome = layer["outcome"]
    overall = "constant_calibration_fingerprint" if constant else layer_outcome
    return {
        "outcome": overall,
        "final_comparator_fingerprint": final_comparator,
        "layer_elasticity_fingerprint": layer_outcome,
        "layer_details": layer,
        "network_population_inference": "not_performed",
    }


def _noninterpretable_statistics(
    neural: dict,
    posterior: dict,
    *,
    reason: str,
    zero_denominators: dict[str, list[int]],
) -> dict:
    retained = {
        name: neural["arrays"][name].tolist()
        for name in ("a_P", "a_H", "a_delta", "a_N", "a_0")
    }
    retained["a_post"] = posterior["arrays"]["a_post"].tolist()
    return {
        "status": "not_computed_noninterpretable",
        "reason": reason,
        "zero_denominators": zero_denominators,
        "raw_subject_amplitudes_retained_without_filtering": retained,
    }


def analyze_seed(specification: dict, seed: int, artifact_validation: dict) -> dict:
    evaluation = load_json(
        _resolve(
            specification["registered_sources"]["slope_localization_specification"][
                "path"
            ]
        )
    )["evaluation"]
    artifact = artifact_validation["lock"]["artifacts"][str(seed)]["checkpoint"]
    checkpoint_path = _resolve(artifact["path"])
    backbone, model_config, checkpoint = load_retro_checkpoint(
        checkpoint_path, int(evaluation["subjects"])
    )
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    before = _tensor_hashes(backbone)
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
    fast_weights = evaluator.learn_fast_weights(FastWeightIntervention.INTACT)
    neural, neural_integrity = _neural_layers(evaluator, fast_weights, geometry)
    posterior, posterior_integrity = _posterior_descriptors(
        evaluator, geometry, specification
    )
    track_b = load_json(
        _resolve(
            specification["registered_sources"]["slope_localization_result"]["path"]
        )
    )["seeds"][str(seed)]
    distances = np.asarray(
        [
            abs(evaluator.item_rank[first] - evaluator.item_rank[second])
            for first, second in geometry.pairs
        ],
        dtype=np.float64,
    )
    nonlearned = np.asarray(
        [pair not in protocol.learned_pairs for pair in geometry.pairs], dtype=bool
    )
    correct = geometry.true_sign[None]
    temperature = float(specification["posterior_comparator"]["choice_temperature"])
    neural_probability = exact_probability(
        correct * neural["fields"]["margin_intact"], temperature
    )
    posterior_probability = 0.5 * (
        1.0 + correct * posterior["fields"]["pair_probability_field"]
    )
    mismatch = subject_slopes(
        neural_probability, distances, nonlearned
    ) - subject_slopes(posterior_probability, distances, nonlearned)
    track_mismatch = np.asarray(
        track_b["raw_subject_level"]["beta_p_minus_posterior"], dtype=np.float64
    )
    track_amplitude = np.asarray(
        track_b["raw_subject_level"]["potential_amplitude"], dtype=np.float64
    )
    required_amplitudes = {
        "a_P": neural["arrays"]["a_P"],
        "a_H": neural["arrays"]["a_H"],
        "a_delta": neural["arrays"]["a_delta"],
        "a_N": neural["arrays"]["a_N"],
        "a_post": posterior["arrays"]["a_post"],
    }
    zero_denominators = {
        name: np.flatnonzero(values <= 1e-12).astype(int).tolist()
        for name, values in required_amplitudes.items()
        if np.any(values <= 1e-12)
    }
    statistics_defined = False
    statistics_error = None
    if zero_denominators:
        statistics_error = "registered amplitude denominator at or below 1e-12"
        statistics = _noninterpretable_statistics(
            neural,
            posterior,
            reason=statistics_error,
            zero_denominators=zero_denominators,
        )
    else:
        try:
            statistics = _seed_statistics(
                specification, seed, neural, posterior, track_b, geometry
            )
            statistics_defined = True
        except NonInterpretableEstimate as error:
            statistics_error = str(error)
            statistics = _noninterpretable_statistics(
                neural,
                posterior,
                reason=statistics_error,
                zero_denominators={},
            )
    after = _tensor_hashes(backbone)
    float_tolerance = float(
        specification["integrity_gates"]["float64_Hodge_and_algebra_tolerance"]
    )
    gpu_tolerance = float(
        specification["integrity_gates"][
            "GPU_hidden_logit_and_transition_reconstruction_tolerance"
        ]
    )
    track_b_tolerance = float(
        specification["integrity_gates"]["Track_B_raw_subject_reproduction_tolerance"]
    )
    integrity = {
        **neural_integrity,
        **posterior_integrity,
        **(statistics["integrity"] if statistics_defined else {}),
        "Track_B_mismatch_reproduction_max_abs_error": float(
            np.max(np.abs(mismatch - track_mismatch))
        ),
        "Track_B_amplitude_reproduction_max_abs_error": float(
            np.max(np.abs(neural["arrays"]["a_N_from_margin"] - track_amplitude))
        ),
        "statistical_estimands_defined_without_filtering": statistics_defined,
        "statistical_estimand_error": statistics_error,
        "zero_denominators": zero_denominators,
        "subjects": model_config.bs,
        "edges": len(geometry.pairs),
        "orientations": 2 * len(geometry.pairs),
        "backbone_tensor_hashes_unchanged": before == after,
    }
    gpu_names = (
        "step0_intact_Poff_max_abs_error",
        "manual_h0_max_abs_error",
        "manual_intact_response_max_abs_error",
        "manual_Poff_response_max_abs_error",
        "manual_delta_response_max_abs_error",
        "hidden_to_logit_max_abs_error",
        "Poff_hidden_to_logit_max_abs_error",
        "Hodge_readout_commutation_max_abs_error",
        "hidden_margin_amplitude_bridge_max_abs_error",
    )
    track_b_names = (
        "Track_B_mismatch_reproduction_max_abs_error",
        "Track_B_amplitude_reproduction_max_abs_error",
    )
    algebra_names = tuple(
        name
        for name in integrity
        if name.endswith("max_abs_error")
        and name not in gpu_names
        and name not in track_b_names
    )
    integrity["passed"] = bool(
        statistics_defined
        and all(
            integrity[name] is not None and integrity[name] <= gpu_tolerance
            for name in gpu_names
        )
        and all(
            integrity[name] is not None and integrity[name] <= float_tolerance
            for name in algebra_names
        )
        and all(integrity[name] <= track_b_tolerance for name in track_b_names)
        and integrity["minimum_required_amplitude"] > 1e-12
        and integrity["rho_min"] is not None
        and integrity["rho_min"] >= -float_tolerance
        and integrity["rho_max"] is not None
        and integrity["rho_max"] <= 1.0 + float_tolerance
        and 0.0 <= integrity["posterior_entropy_min"]
        and integrity["posterior_entropy_max"] <= 1.0
        and 0.0 <= integrity["coverage_min"]
        and integrity["coverage_max"] <= 1.0
        and integrity["coverage_binary_max_abs_error"] <= float_tolerance
        and integrity["coverage_relation_reuse_max_abs_error"] <= float_tolerance
        and integrity["coverage_unique_fraction_max_abs_error"] <= float_tolerance
        and integrity["coverage_unique_relations_min"] == 8
        and integrity["coverage_unique_relations_max"] == 8
        and integrity.get("bootstrap_samples")
        == int(specification["statistical_estimands"]["bootstrap"]["samples"])
        and integrity.get("bootstrap_subjects") == 77
        and integrity["subjects"] == 77
        and integrity["edges"] == 28
        and integrity["orientations"] == 56
        and integrity["backbone_tensor_hashes_unchanged"]
    )
    return {
        "seed": seed,
        "checkpoint": {"path": artifact["path"], "sha256": checkpoint.sha256},
        "condition": "pure_L_off_intact_P_T_with_query_time_Poff_control",
        "w_norm": neural["w_norm"],
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
        "seeds": seeds,
        "decision": cross_seed_decision(seeds),
    }


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Run the frozen global-policy amplitude-provenance audit."
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
