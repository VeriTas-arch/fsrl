"""Synthetic calibration and two-parameter identification for JMIC-A."""

from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.optimize import least_squares
from scipy.special import ndtri

from .model import (
    incidence,
    normal_sigmoid_moment,
    pair_matrix,
    posterior_7d,
    posterior_pair_distribution,
    zero_sum_basis,
)
from .predictions import condition_posterior


def random_connected_edges(
    rng: np.random.Generator, n_items: int, edge_count: int
) -> tuple[tuple[int, int], ...]:
    if not n_items - 1 <= edge_count <= n_items * (n_items - 1) // 2:
        raise ValueError("invalid connected graph edge count")
    permutation = rng.permutation(n_items)
    edges = {
        tuple(sorted((int(permutation[index]), int(permutation[index + 1]))))
        for index in range(n_items - 1)
    }
    available = [pair for pair in combinations(range(n_items), 2) if pair not in edges]
    rng.shuffle(available)
    edges.update(available[: edge_count - len(edges)])
    return tuple((int(pair[0]), int(pair[1])) for pair in sorted(edges))


def simulation_based_calibration(
    *,
    sigma: float,
    tau2: float,
    episodes: int,
    edge_counts: tuple[int, ...],
    seed: int,
) -> dict:
    if episodes % len(edge_counts):
        raise ValueError("episodes must divide equally across edge counts")
    rng = np.random.default_rng(seed)
    basis = zero_sum_basis()
    pairs, pair_design = pair_matrix()
    coordinate_z = []
    pair_z = []
    graph_counts = {str(count): 0 for count in edge_counts}
    for edge_count in np.repeat(edge_counts, episodes // len(edge_counts)):
        edges = random_connected_edges(rng, 8, int(edge_count))
        design = incidence(edges)
        true_z = rng.normal(0.0, np.sqrt(tau2), size=7)
        true_v = basis @ true_z
        observation = design @ true_v + rng.normal(0.0, sigma, size=edge_count)
        posterior = posterior_7d(design, observation, sigma=sigma, tau2=tau2)
        root = np.linalg.cholesky(posterior.covariance_z)
        coordinate_z.extend(np.linalg.solve(root, true_z - posterior.mean_z))
        pair_mean = pair_design @ posterior.mean_v
        pair_variance = np.einsum(
            "pi,ij,pj->p", pair_design, posterior.covariance_v, pair_design
        )
        pair_true = pair_design @ true_v
        pair_z.extend((pair_true - pair_mean) / np.sqrt(pair_variance))
        graph_counts[str(int(edge_count))] += 1
    coordinate_z = np.asarray(coordinate_z)
    pair_z = np.asarray(pair_z)
    levels = (0.5, 0.8, 0.95)

    def summary(values: np.ndarray) -> dict:
        coverage = {}
        for level in levels:
            bound = float(ndtri((1.0 + level) / 2.0))
            observed = float(np.mean(np.abs(values) <= bound))
            standard_error = float(np.sqrt(level * (1.0 - level) / len(values)))
            coverage[str(level)] = {
                "observed": observed,
                "absolute_error": abs(observed - level),
                "four_se_plus_0_01": 4.0 * standard_error + 0.01,
                "passed": abs(observed - level) <= 4.0 * standard_error + 0.01,
            }
        return {
            "samples": len(values),
            "mean": float(np.mean(values)),
            "variance": float(np.var(values)),
            "coverage": coverage,
            "passed": abs(float(np.mean(values))) <= 0.08
            and abs(float(np.var(values)) - 1.0) <= 0.12
            and all(row["passed"] for row in coverage.values()),
        }

    return {
        "episodes": episodes,
        "graph_counts": graph_counts,
        "coordinate": summary(coordinate_z),
        "pair_difference": summary(pair_z),
        "passed": summary(coordinate_z)["passed"] and summary(pair_z)["passed"],
        "pair_count": len(pairs),
    }


def fixed_moments(
    sigma_ratio: float,
    theta_ratio: float,
    *,
    tau2: float,
    nodes: int,
) -> tuple[np.ndarray, np.ndarray]:
    tau = np.sqrt(tau2)
    posterior = condition_posterior("A", sigma_ratio * tau, tau2)
    _, means, covariance = posterior_pair_distribution(posterior)
    variance = np.diag(covariance)
    theta = theta_ratio * tau
    return (
        normal_sigmoid_moment(means, variance, theta, nodes=nodes),
        normal_sigmoid_moment(means, variance, theta, power=2, nodes=nodes),
    )


def moment_jacobian(
    sigma_ratio: float,
    theta_ratio: float,
    *,
    tau2: float,
    nodes: int,
    step: float,
) -> dict:
    center = np.log([sigma_ratio, theta_ratio])
    columns = []
    for index in range(2):
        delta = np.zeros(2)
        delta[index] = step
        plus = np.exp(center + delta)
        minus = np.exp(center - delta)
        plus_moments = np.concatenate(fixed_moments(*plus, tau2=tau2, nodes=nodes))
        minus_moments = np.concatenate(fixed_moments(*minus, tau2=tau2, nodes=nodes))
        columns.append((plus_moments - minus_moments) / (2.0 * step))
    jacobian = np.column_stack(columns)
    singular = np.linalg.svd(jacobian, compute_uv=False)
    condition = float(singular[0] / singular[-1])
    return {
        "singular_values": [float(value) for value in singular],
        "condition_number": condition,
        "numerical_rank": int(np.linalg.matrix_rank(jacobian)),
    }


def _estimate_parameters(
    observed_a: np.ndarray,
    observed_b: np.ndarray,
    *,
    tau2: float,
    nodes: int,
    bounds: dict,
) -> tuple[float, float, bool]:
    lower = np.log([bounds["sigma_eff_over_tau0"][0], bounds["theta_dec_over_tau0"][0]])
    upper = np.log([bounds["sigma_eff_over_tau0"][1], bounds["theta_dec_over_tau0"][1]])

    def residual(log_parameters: np.ndarray) -> np.ndarray:
        predicted = fixed_moments(*np.exp(log_parameters), tau2=tau2, nodes=nodes)
        return np.concatenate((predicted[0] - observed_a, predicted[1] - observed_b))

    fitted = least_squares(
        residual,
        np.log([0.5, 0.25]),
        bounds=(lower, upper),
        xtol=1e-9,
        ftol=1e-9,
        gtol=1e-9,
        max_nfev=100,
    )
    values = np.exp(fitted.x)
    return float(values[0]), float(values[1]), bool(fitted.success)


def parameter_recovery(
    *,
    sigma_ratio: float,
    theta_ratio: float,
    tau2: float,
    nodes: int,
    cohorts: int,
    subjects: int,
    repetitions: int,
    bounds: dict,
    seed: int,
) -> tuple[dict, np.ndarray]:
    rng = np.random.default_rng(seed)
    tau = np.sqrt(tau2)
    sigma = sigma_ratio * tau
    theta = theta_ratio * tau
    posterior = condition_posterior("A", sigma, tau2)
    estimates = np.empty((cohorts, 2), dtype=np.float64)
    successes = np.zeros(cohorts, dtype=bool)
    cross_pair_errors = []
    true_a, _ = fixed_moments(sigma_ratio, theta_ratio, tau2=tau2, nodes=nodes)
    for cohort in range(cohorts):
        z = rng.multivariate_normal(
            posterior.mean_z, posterior.covariance_z, size=subjects
        )
        probability = 1.0 / (
            1.0 + np.exp(-(z @ posterior.basis.T @ pair_matrix()[1].T) / theta)
        )
        counts = rng.binomial(repetitions, probability)
        observed_a = np.mean(counts / repetitions, axis=0)
        observed_b = np.mean(
            counts * (counts - 1) / (repetitions * (repetitions - 1)), axis=0
        )
        sigma_hat, theta_hat, success = _estimate_parameters(
            observed_a,
            observed_b,
            tau2=tau2,
            nodes=nodes,
            bounds=bounds,
        )
        estimates[cohort] = (sigma_hat, theta_hat)
        successes[cohort] = success
        observed_cross = np.cov(counts / repetitions, rowvar=False, ddof=0)
        cross_pair_errors.append(float(np.mean(np.abs(observed_cross))))
    truth = np.asarray([sigma_ratio, theta_ratio])
    absolute_log_ratio = np.abs(np.log(estimates / truth))
    return (
        {
            "cohorts": cohorts,
            "subjects_per_cohort": subjects,
            "optimizer_successes": int(np.sum(successes)),
            "median_absolute_log_ratio": [
                float(value) for value in np.median(absolute_log_ratio, axis=0)
            ],
            "p90_absolute_log_ratio": [
                float(value) for value in np.quantile(absolute_log_ratio, 0.9, axis=0)
            ],
            "estimate_median": [float(value) for value in np.median(estimates, axis=0)],
            "mean_observed_probability_reference": float(np.mean(true_a)),
            "mean_absolute_empirical_choice_covariance": float(
                np.mean(cross_pair_errors)
            ),
        },
        estimates,
    )


__all__ = [
    "fixed_moments",
    "moment_jacobian",
    "parameter_recovery",
    "random_connected_edges",
    "simulation_based_calibration",
]
