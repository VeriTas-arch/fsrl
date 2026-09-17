"""Focused tests for the prospective JMIC-A equations."""

from __future__ import annotations

import numpy as np

from fsrl.experiments.jmic_a.model import (
    amplitude_matched_equal_spacing,
    incidence,
    posterior_7d,
    posterior_8d_constrained,
    zero_sum_basis,
)
from fsrl.experiments.jmic_a.qualification import qualification_checks


def test_zero_sum_basis_is_orthonormal() -> None:
    basis = zero_sum_basis()
    assert np.allclose(basis.T @ basis, np.eye(7), atol=1e-14)
    assert np.allclose(np.sum(basis, axis=0), 0.0, atol=1e-14)


def test_constrained_and_reduced_posteriors_agree() -> None:
    design = incidence(((0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7), (6, 0)))
    observations = np.linspace(-0.4, 0.5, len(design))
    reduced = posterior_7d(design, observations, sigma=0.2, tau2=6 / 49)
    mean, covariance = posterior_8d_constrained(
        design, observations, sigma=0.2, tau2=6 / 49
    )
    assert np.allclose(reduced.mean_v, mean, atol=1e-12)
    assert np.allclose(reduced.covariance_v, covariance, atol=1e-12)


def test_equal_spacing_preserves_order_and_norm() -> None:
    values = np.asarray([[0.2, -1.0, 0.7, 0.1, 2.0, -0.4, 0.5, -0.1]])
    projected = amplitude_matched_equal_spacing(values)
    assert np.array_equal(np.argsort(values), np.argsort(projected))
    assert np.allclose(
        np.linalg.norm(values, axis=1), np.linalg.norm(projected, axis=1)
    )


def test_registered_qualification_checks_pass() -> None:
    result = qualification_checks()
    assert result["passed"], result
