import unittest

import numpy as np

from fsrl.analysis.hodge import build_complete_graph_geometry
from fsrl.experiments.global_policy.slope_localization import (
    choice_link_components,
    cross_seed_decision,
    seed_decision,
    subject_slopes,
)
from fsrl.infra.provenance import load_json
from fsrl.infra.study_registry import resolve_record
from fsrl.tasks.registered_protocol import load_ranking_protocol


class GlobalPolicySlopeLocalizationTests(unittest.TestCase):
    def setUp(self):
        self.protocol = load_ranking_protocol(resolve_record("benchmarks/liu_v2.json"))
        self.geometry = build_complete_graph_geometry(self.protocol)
        positions = np.empty(self.protocol.n_items, dtype=np.int64)
        for position, item in enumerate(self.protocol.true_order_high_to_low):
            positions[item] = position
        self.distances = np.asarray(
            [abs(positions[i] - positions[j]) for i, j in self.geometry.pairs],
            dtype=np.float64,
        )
        self.mask = np.asarray(
            [pair not in self.protocol.learned_pairs for pair in self.geometry.pairs]
        )

    def test_subject_slopes_preserve_exact_hodge_additivity(self):
        rng = np.random.default_rng(7)
        potentials = rng.normal(size=(5, self.protocol.n_items))
        gradient = potentials @ self.geometry.incidence.T
        residual = rng.normal(scale=0.1, size=gradient.shape)
        total = gradient + residual
        beta_total = subject_slopes(total, self.distances, self.mask)
        beta_gradient = subject_slopes(gradient, self.distances, self.mask)
        beta_residual = subject_slopes(residual, self.distances, self.mask)
        np.testing.assert_allclose(
            beta_total, beta_gradient + beta_residual, atol=1e-14
        )

    def test_choice_projection_has_exact_slope_remainder(self):
        rng = np.random.default_rng(8)
        margins = rng.normal(size=(4, len(self.geometry.pairs)))
        probabilities = 1.0 / (1.0 + np.exp(-margins / 0.25))
        result = choice_link_components(
            margins, probabilities, self.distances, self.mask
        )
        np.testing.assert_allclose(
            result["beta_probability"],
            result["beta_linearized"] + result["beta_remainder"],
            atol=1e-14,
        )

    def test_seed_and_cross_seed_decisions_do_not_pool(self):
        def summary(lower, upper):
            return {"bootstrap": {"lower": lower, "upper": upper}}

        metrics = {
            "beta_m": summary(0.1, 0.2),
            "beta_g_minus_0_9_beta_m": summary(0.01, 0.02),
            "beta_hat_minus_posterior": summary(0.01, 0.02),
            "beta_c": summary(-0.01, 0.01),
            "beta_p_minus_posterior": summary(0.01, 0.02),
            "beta_e": summary(-0.02, -0.01),
        }
        first = seed_decision(metrics)
        second = {"flags": {**first["flags"], "normalized_geometry_excess": False}}
        result = cross_seed_decision(
            {"2104": {"decision": first}, "2105": {"decision": second}}
        )
        self.assertEqual(
            result["links"]["normalized_geometry_excess"]["status"],
            "heterogeneous",
        )
        self.assertEqual(result["network_population_inference"], "not_performed")

    def test_frozen_specification_keeps_local_trace_out_of_primary(self):
        specification = load_json(
            resolve_record("benchmarks/global_policy_slope_localization_v1.json")
        )
        self.assertEqual(
            specification["network_contract"]["primary_condition"],
            "Pure L-off v1/global branch with intact P_T and frozen W_out; no local trace is constructed or read.",
        )
        self.assertEqual(specification["evaluation"]["choice_temperature"], 0.25)
        self.assertEqual(
            specification["network_contract"]["mandatory_frozen_seeds"],
            [2104, 2105],
        )
