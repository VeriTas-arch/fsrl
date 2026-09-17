"""Execute the frozen synthetic qualification and prediction study."""

from __future__ import annotations

import platform
from datetime import UTC, datetime

import numpy as np

from fsrl.experiments.pl_crosstalk_decomposition.storage import write_npz_exclusive
from fsrl.experiments.training_strategy.locks import reference
from fsrl.infra.provenance import load_json, write_json_exclusive

from .locks import validate_source_input_lock
from .model import posterior_pair_distribution
from .predictions import (
    PAIR_NAMES,
    analytic_readout,
    condition_posterior,
    morphology_summary,
    placement_estimands,
    sampled_readouts,
    task_conditions,
)
from .protocol import (
    PREDICTIONS,
    PROTOCOL_SHA256,
    RESULT,
    SOURCE_INPUT_LOCK,
    specification,
)
from .synthetic import (
    moment_jacobian,
    parameter_recovery,
    simulation_based_calibration,
)

READOUTS = (
    "posterior_mean",
    "trialwise_marginal",
    "persistent_continuous",
    "persistent_equal_spacing",
)
CONDITIONS = ("A", "B")


def _json_ready(value):
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _parameter_cells(spec: dict) -> list[tuple[float, float]]:
    return [
        (sigma, theta)
        for sigma in spec["design"]["parameter_cells"]["sigma_eff_over_tau0"]
        for theta in spec["design"]["parameter_cells"]["theta_dec_over_tau0"]
    ]


def _direction(values: list[float], tolerance: float = 1e-12) -> str:
    if all(value > tolerance for value in values):
        return "positive_in_all_cells"
    if all(value < -tolerance for value in values):
        return "negative_in_all_cells"
    if all(abs(value) <= tolerance for value in values):
        return "zero_in_all_cells"
    return "parameter_dependent_sign"


def run() -> dict:
    if RESULT.exists() or PREDICTIONS.exists():
        if RESULT.exists() and PREDICTIONS.exists():
            return load_json(RESULT)
        raise RuntimeError("partial JMIC-A registered output exists")
    source_lock = validate_source_input_lock()
    spec = specification()
    stage = spec["stage_1"]
    tau2 = spec["design"]["prior_variance_tau0_squared"]
    tau = np.sqrt(tau2)
    nodes = stage["gauss_hermite_nodes"]
    master_seed = stage["master_seed"]
    parameter_cells = _parameter_cells(spec)
    protocol, task = task_conditions()
    learned_pairs = protocol["inherited_frozen_contract"]["pair_partition"]["learned"]

    shape = (len(parameter_cells), len(CONDITIONS))
    arrays = {
        "posterior_mean_v": np.empty(shape + (8,), dtype=np.float64),
        "posterior_covariance_v": np.empty(shape + (8, 8), dtype=np.float64),
        "pair_mean": np.empty(shape + (28,), dtype=np.float64),
        "pair_variance": np.empty(shape + (28,), dtype=np.float64),
        "probability_mean": np.empty(shape + (len(READOUTS), 28), dtype=np.float64),
        "probability_second_moment": np.empty(
            shape + (len(READOUTS), 28), dtype=np.float64
        ),
        "probability_covariance": np.empty(
            shape + (len(READOUTS), 28, 28), dtype=np.float64
        ),
        "latent_reversal_probability": np.empty(shape + (28,), dtype=np.float64),
        "strong_first_loss_probability": np.empty(shape + (28,), dtype=np.float64),
        "persistent_all_first_loss_probability": np.empty(
            shape + (28,), dtype=np.float64
        ),
        "trialwise_all_first_loss_probability": np.empty(
            shape + (28,), dtype=np.float64
        ),
        "recovery_small": np.empty(
            (len(parameter_cells), stage["cohort_recovery"]["small"]["cohorts"], 2),
            dtype=np.float64,
        ),
        "recovery_large": np.empty(
            (len(parameter_cells), stage["cohort_recovery"]["large"]["cohorts"], 2),
            dtype=np.float64,
        ),
    }

    cell_results = []
    morphology = []
    contrast_table = {readout: [] for readout in READOUTS}
    all_sbc_passed = True
    all_rank_two = True
    no_weak_cells = True
    all_large_recovery_passed = True

    for cell_index, (sigma_ratio, theta_ratio) in enumerate(parameter_cells):
        sigma = sigma_ratio * tau
        theta = theta_ratio * tau
        seed_base = master_seed + 100_000 * cell_index
        sbc = simulation_based_calibration(
            sigma=sigma,
            tau2=tau2,
            episodes=stage["model_matched_episodes_per_parameter_cell"],
            edge_counts=tuple(stage["graph_edge_counts"]),
            seed=seed_base + 1,
        )
        jacobian = moment_jacobian(
            sigma_ratio,
            theta_ratio,
            tau2=tau2,
            nodes=nodes,
            step=stage["jacobian_log_step"],
        )
        recovery = {}
        for size_index, size_name in enumerate(("small", "large")):
            settings = stage["cohort_recovery"][size_name]
            summary, estimates = parameter_recovery(
                sigma_ratio=sigma_ratio,
                theta_ratio=theta_ratio,
                tau2=tau2,
                nodes=nodes,
                cohorts=settings["cohorts"],
                subjects=settings["subjects"],
                repetitions=stage["cohort_recovery"]["query_repetitions"],
                bounds=stage["cohort_recovery"]["bounds"],
                seed=seed_base + 10 + size_index,
            )
            arrays[f"recovery_{size_name}"][cell_index] = estimates
            recovery[size_name] = summary

        large_threshold = stage["large_cohort_recovery_thresholds"]
        large_passed = (
            recovery["large"]["optimizer_successes"] == recovery["large"]["cohorts"]
            and max(recovery["large"]["median_absolute_log_ratio"])
            <= large_threshold["median_absolute_log_ratio_max"]
            and max(recovery["large"]["p90_absolute_log_ratio"])
            <= large_threshold["p90_absolute_log_ratio_max"]
        )
        weak = jacobian["condition_number"] > stage["weak_condition_number"]
        all_sbc_passed &= sbc["passed"]
        all_rank_two &= jacobian["numerical_rank"] == 2
        no_weak_cells &= not weak
        all_large_recovery_passed &= large_passed

        condition_fields = {}
        for condition_index, condition in enumerate(CONDITIONS):
            posterior = condition_posterior(condition, sigma, tau2)
            analytic = analytic_readout(posterior, theta, nodes)
            sampled = sampled_readouts(
                posterior,
                theta,
                subjects=stage["fixed_prediction_subjects"],
                equal_spacing_draws=stage["equal_spacing_draws"],
                seed=seed_base + 100 + condition_index,
            )
            _, pair_mean, pair_covariance = posterior_pair_distribution(posterior)
            arrays["posterior_mean_v"][cell_index, condition_index] = posterior.mean_v
            arrays["posterior_covariance_v"][cell_index, condition_index] = (
                posterior.covariance_v
            )
            arrays["pair_mean"][cell_index, condition_index] = pair_mean
            arrays["pair_variance"][cell_index, condition_index] = np.diag(
                pair_covariance
            )

            probability_mean = {
                "posterior_mean": analytic["posterior_mean_probability"],
                "trialwise_marginal": analytic["marginal_probability"],
                "persistent_continuous": analytic["marginal_probability"],
                "persistent_equal_spacing": sampled["equal_spacing_mean"],
            }
            probability_second = {
                "posterior_mean": probability_mean["posterior_mean"] ** 2,
                "trialwise_marginal": probability_mean["trialwise_marginal"] ** 2,
                "persistent_continuous": analytic["persistent_second_moment"],
                "persistent_equal_spacing": sampled["equal_spacing_second"],
            }
            probability_covariance = {
                "posterior_mean": np.zeros((28, 28), dtype=np.float64),
                "trialwise_marginal": np.zeros((28, 28), dtype=np.float64),
                "persistent_continuous": analytic["persistent_probability_covariance"],
                "persistent_equal_spacing": sampled["equal_spacing_covariance"],
            }
            for readout_index, readout in enumerate(READOUTS):
                arrays["probability_mean"][
                    cell_index, condition_index, readout_index
                ] = probability_mean[readout]
                arrays["probability_second_moment"][
                    cell_index, condition_index, readout_index
                ] = probability_second[readout]
                arrays["probability_covariance"][
                    cell_index, condition_index, readout_index
                ] = probability_covariance[readout]
            arrays["latent_reversal_probability"][cell_index, condition_index] = (
                analytic["latent_reversal_probability"]
            )
            arrays["strong_first_loss_probability"][cell_index, condition_index] = (
                analytic["first_item_strong_loss_probability"]
            )
            arrays["persistent_all_first_loss_probability"][
                cell_index, condition_index
            ] = analytic["persistent_all_first_loss_probability"]
            arrays["trialwise_all_first_loss_probability"][
                cell_index, condition_index
            ] = analytic["trialwise_all_first_loss_probability"]
            condition_fields[condition] = probability_mean

            if condition == "A":
                subject_probabilities = {
                    "posterior_mean": np.broadcast_to(
                        probability_mean["posterior_mean"],
                        (stage["fixed_prediction_subjects"], 28),
                    ),
                    "trialwise_marginal": np.broadcast_to(
                        probability_mean["trialwise_marginal"],
                        (stage["fixed_prediction_subjects"], 28),
                    ),
                    "persistent_continuous": sampled["persistent_continuous"],
                    "persistent_equal_spacing": sampled["persistent_equal_spacing"],
                }
                for readout_index, readout in enumerate(READOUTS):
                    morphology.append(
                        {
                            "cell": cell_index,
                            "sigma_eff_over_tau0": sigma_ratio,
                            "theta_dec_over_tau0": theta_ratio,
                            "readout": readout,
                            **morphology_summary(
                                subject_probabilities[readout],
                                task["A"]["levels"],
                                learned_pairs,
                                seed=seed_base + 1000 + readout_index,
                                repetitions=stage["cohort_recovery"][
                                    "query_repetitions"
                                ],
                            ),
                        }
                    )

        for readout in READOUTS:
            contrasts = placement_estimands(
                condition_fields["A"][readout],
                condition_fields["B"][readout],
                protocol,
            )
            contrast_table[readout].append(
                {
                    "cell": cell_index,
                    "sigma_eff_over_tau0": sigma_ratio,
                    "theta_dec_over_tau0": theta_ratio,
                    **contrasts,
                }
            )

        cell_results.append(
            {
                "cell": cell_index,
                "sigma_eff_over_tau0": sigma_ratio,
                "theta_dec_over_tau0": theta_ratio,
                "sbc": sbc,
                "jacobian": jacobian,
                "weakly_conditioned": weak,
                "recovery": recovery,
                "large_recovery_passed": large_passed,
            }
        )

    implementation_status = "qualified" if all_sbc_passed else "noninterpretable"
    identification_status = (
        "qualified_for_prediction_registration"
        if all_rank_two and no_weak_cells and all_large_recovery_passed
        else "qualified_but_weakly_identified"
    )
    directions = {
        readout: {
            estimand: {
                "direction": _direction([row[estimand] for row in rows]),
                "minimum": float(min(row[estimand] for row in rows)),
                "maximum": float(max(row[estimand] for row in rows)),
            }
            for estimand in ("delta_flip", "beta_conf", "beta_learned")
        }
        for readout, rows in contrast_table.items()
    }
    marginal_control_error = float(
        np.max(
            np.abs(
                arrays["probability_mean"][:, :, READOUTS.index("trialwise_marginal")]
                - arrays["probability_mean"][
                    :, :, READOUTS.index("persistent_continuous")
                ]
            )
        )
    )
    payload = {
        "schema_version": 1,
        "study_id": "jmic_a_v1",
        "protocol_sha256": PROTOCOL_SHA256,
        "source_input_lock": reference(SOURCE_INPUT_LOCK),
        "executed_at_utc": datetime.now(UTC).isoformat(),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "device": "CPU float64",
        },
        "axes": {
            "parameter_cells": [
                {
                    "cell": index,
                    "sigma_eff_over_tau0": sigma,
                    "theta_dec_over_tau0": theta,
                }
                for index, (sigma, theta) in enumerate(parameter_cells)
            ],
            "conditions": list(CONDITIONS),
            "readouts": list(READOUTS),
            "pairs": list(PAIR_NAMES),
        },
        "rng": {
            "master_seed": master_seed,
            "cell_stride": 100000,
            "domains": {
                "sbc": 1,
                "small_recovery": 10,
                "large_recovery": 11,
                "A_prediction": 100,
                "B_prediction": 101,
                "morphology": [1000, 1001, 1002, 1003],
            },
        },
        "implementation_status": implementation_status,
        "identification_status": identification_status,
        "prediction_status": {
            "exact_identities": {
                "trialwise_persistent_pair_marginal_max_abs_error": marginal_control_error,
                "A_B_covariance_identity_qualified": True,
                "cycle_variance_classes": 4,
                "magnitude_changes_covariance": False,
                "repetition_law_defined": False,
            },
            "A_B_estimand_directions": directions,
            "interpretation": (
                "Predictions are computational and use no participant responses; "
                "they do not constitute human rescue or model promotion."
            ),
        },
        "cells": cell_results,
        "A_B_contrasts": contrast_table,
        "morphology_compatibility": morphology,
        "completeness": {
            "parameter_cells": len(cell_results),
            "expected_parameter_cells": 9,
            "morphology_rows": len(morphology),
            "expected_morphology_rows": 36,
            "all_sbc_passed": all_sbc_passed,
            "all_jacobians_rank_two": all_rank_two,
            "no_weak_cells": no_weak_cells,
            "all_large_recovery_passed": all_large_recovery_passed,
            "participant_response_inputs": len(
                source_lock["participant_response_inputs"]
            ),
            "neural_checkpoint_inputs": len(source_lock["neural_checkpoint_inputs"]),
        },
        "claim_boundary": spec["claim_boundary"],
    }
    write_npz_exclusive(PREDICTIONS, arrays)
    payload["prediction_arrays"] = reference(PREDICTIONS)
    write_json_exclusive(RESULT, _json_ready(payload))
    return payload


__all__ = ["CONDITIONS", "READOUTS", "run"]
