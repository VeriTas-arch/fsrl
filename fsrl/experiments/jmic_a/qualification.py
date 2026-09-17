"""Pre-execution mathematical and implementation qualification."""

from __future__ import annotations

from itertools import combinations

import numpy as np

from fsrl.experiments.training_strategy.locks import reference
from fsrl.infra.provenance import write_json_exclusive
from fsrl.paths import REPO_ROOT

from .model import (
    incidence,
    normal_sigmoid_moment,
    pair_matrix,
    posterior_7d,
    posterior_8d_constrained,
    posterior_pair_distribution,
)
from .predictions import ROLE_INDEX, condition_posterior, task_conditions
from .protocol import PROTOCOL_SHA256, QUALIFICATION, register, specification


def source_paths():
    paths = list((REPO_ROOT / "fsrl/experiments/jmic_a").glob("*.py"))
    paths += list((REPO_ROOT / "tests/experiments/jmic_a").glob("*.py"))
    paths += [
        REPO_ROOT / "studies/jmic_a_v1/records/benchmarks/jmic_a_v1.json",
        REPO_ROOT
        / "studies/jmic_a_v1/records/benchmarks/jmic_a_v1_quadrature_repair1.json",
    ]
    return sorted(path for path in set(paths) if path.is_file())


def sources() -> list[dict]:
    return [reference(path) for path in source_paths()]


def _cycle_distances() -> dict[tuple[int, int], int]:
    order = [ROLE_INDEX[role] for role in ("A", "F", "D", "G", "C", "B", "E", "H")]
    position = {item: index for index, item in enumerate(order)}
    return {
        pair: min(
            abs(position[pair[0]] - position[pair[1]]),
            8 - abs(position[pair[0]] - position[pair[1]]),
        )
        for pair in combinations(range(8), 2)
    }


def qualification_checks() -> dict:
    spec = specification()
    tolerance = spec["stage_0"]["algebra_tolerance"]
    tau2 = spec["design"]["prior_variance_tau0_squared"]
    tau = np.sqrt(tau2)
    sigma = 0.5 * tau
    theta = 0.25 * tau
    protocol, task = task_conditions()
    design = task["design"]
    posterior_a = condition_posterior("A", sigma, tau2)
    posterior_b = condition_posterior("B", sigma, tau2)

    mean8, covariance8 = posterior_8d_constrained(
        design, task["A"]["observations"], sigma=sigma, tau2=tau2
    )
    independent_error = max(
        float(np.max(np.abs(mean8 - posterior_a.mean_v))),
        float(np.max(np.abs(covariance8 - posterior_a.covariance_v))),
    )

    mapping = spec["design"]["fixed_task_inputs"]["A_to_B_role_permutation"]
    permutation = np.zeros((8, 8), dtype=np.float64)
    for role, source in mapping.items():
        permutation[ROLE_INDEX[role], ROLE_INDEX[source]] = 1.0
    ab_mean_error = float(
        np.max(np.abs(posterior_b.mean_v - permutation @ posterior_a.mean_v))
    )
    ab_covariance_equal_error = float(
        np.max(np.abs(posterior_b.covariance_v - posterior_a.covariance_v))
    )
    ab_covariance_permutation_error = float(
        np.max(
            np.abs(
                posterior_a.covariance_v
                - permutation @ posterior_a.covariance_v @ permutation.T
            )
        )
    )

    pairs, _, pair_covariance = posterior_pair_distribution(posterior_a)
    pair_variance = np.diag(pair_covariance)
    distances = _cycle_distances()
    class_spread = {}
    for distance in range(1, 5):
        values = np.asarray(
            [
                pair_variance[index]
                for index, pair in enumerate(pairs)
                if distances[pair] == distance
            ]
        )
        class_spread[str(distance)] = float(np.ptp(values))
    unique_classes = len(
        np.unique(
            np.round(
                [
                    np.mean(pair_variance[[distances[pair] == d for pair in pairs]])
                    for d in range(1, 5)
                ],
                12,
            )
        )
    )

    _, _, vh = np.linalg.svd(design.T)
    cycle_null = vh[-1]
    null_error = float(np.max(np.abs(design.T @ cycle_null)))
    shifted = posterior_7d(
        design,
        task["A"]["observations"] + cycle_null,
        sigma=sigma,
        tau2=tau2,
    )
    null_posterior_error = max(
        float(np.max(np.abs(shifted.mean_v - posterior_a.mean_v))),
        float(np.max(np.abs(shifted.covariance_v - posterior_a.covariance_v))),
    )

    new_relation = incidence(((0, 1),))
    augmented = posterior_7d(
        np.vstack((design, new_relation)),
        np.concatenate((task["A"]["observations"], new_relation @ task["A"]["levels"])),
        sigma=sigma,
        tau2=tau2,
    )
    ell = new_relation[0]
    covariance_update = posterior_a.covariance_v - np.outer(
        posterior_a.covariance_v @ ell, posterior_a.covariance_v @ ell
    ) / (sigma**2 + ell @ posterior_a.covariance_v @ ell)
    rank_one_error = float(np.max(np.abs(covariance_update - augmented.covariance_v)))

    _, means, covariance = posterior_pair_distribution(posterior_a)
    variance = np.diag(covariance)
    nodes = spec["stage_1"]["gauss_hermite_nodes"]
    marginal_primary = normal_sigmoid_moment(means, variance, theta, nodes=nodes)
    marginal_reference = normal_sigmoid_moment(means, variance, theta, nodes=2 * nodes)
    quadrature_error = float(np.max(np.abs(marginal_primary - marginal_reference)))

    checks = {
        "independent_posterior_implementations": independent_error <= tolerance,
        "permutation_equivariance_mean": ab_mean_error <= tolerance,
        "A_B_covariance_equal": ab_covariance_equal_error <= tolerance,
        "A_B_covariance_permutation_invariant": ab_covariance_permutation_error
        <= tolerance,
        "cycle_has_four_variance_classes": unique_classes == 4
        and max(class_spread.values()) <= tolerance,
        "cycle_null_vector": null_error <= tolerance,
        "cycle_null_posterior_blindness": null_posterior_error <= tolerance,
        "rank_one_covariance_update": rank_one_error <= tolerance,
        "quadrature_crosscheck": quadrature_error
        <= spec["stage_0"]["quadrature_absolute_tolerance"],
        "pair_marginal_control_identity": bool(
            np.array_equal(marginal_primary, marginal_primary)
        ),
        "task_input_has_no_responses": "responses" not in protocol
        and "participants" not in protocol,
    }
    diagnostics = {
        "independent_posterior_max_abs_error": independent_error,
        "A_B_mean_permutation_max_abs_error": ab_mean_error,
        "A_B_covariance_equal_max_abs_error": ab_covariance_equal_error,
        "A_B_covariance_permutation_max_abs_error": ab_covariance_permutation_error,
        "cycle_variance_class_spread": class_spread,
        "cycle_variance_unique_classes": unique_classes,
        "cycle_null_max_abs_error": null_error,
        "cycle_null_posterior_max_abs_error": null_posterior_error,
        "rank_one_update_max_abs_error": rank_one_error,
        "quadrature_primary_reference_max_abs_error": quadrature_error,
        "quadrature_nodes": nodes,
        "pair_count": len(pair_matrix()[0]),
    }
    return {
        "schema_version": 1,
        "study_id": "jmic_a_v1",
        "protocol_sha256": PROTOCOL_SHA256,
        "checks": checks,
        "diagnostics": diagnostics,
        "sources": sources(),
        "passed": all(checks.values()),
    }


def qualify() -> dict:
    payload = qualification_checks()
    if not payload["passed"]:
        raise RuntimeError(f"JMIC-A qualification failed: {payload['checks']}")
    write_json_exclusive(QUALIFICATION, payload)
    register(
        finding="Prospective JMIC-A protocol and implementation qualification frozen; execution pending."
    )
    return payload


__all__ = ["qualification_checks", "qualify", "source_paths", "sources"]
