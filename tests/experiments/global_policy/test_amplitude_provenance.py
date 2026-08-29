import unittest
from importlib import import_module

import numpy as np

from fsrl.analysis.hodge import build_complete_graph_geometry, hodge_potentials
from fsrl.tasks.protocol import RankingProtocol

_PROVENANCE = import_module("fsrl.experiments.global_policy.amplitude_provenance")
amplitude_ledger = _PROVENANCE.amplitude_ledger
bootstrap_ols_slopes = _PROVENANCE.bootstrap_ols_slopes
bootstrap_correlation = _PROVENANCE.bootstrap_correlation
cross_network_outcome = _PROVENANCE.cross_network_outcome
cross_seed_decision = _PROVENANCE.cross_seed_decision
elasticity_ledger = _PROVENANCE.elasticity_ledger
layer_elasticity_decision = _PROVENANCE.layer_elasticity_decision
NonInterpretableEstimate = _PROVENANCE.NonInterpretableEstimate
ols_slope = _PROVENANCE.ols_slope
vector_item_potentials = _PROVENANCE.vector_item_potentials
interval_summary = _PROVENANCE.interval_summary


class GlobalPolicyAmplitudeProvenanceTests(unittest.TestCase):
    def setUp(self):
        protocol = RankingProtocol(
            protocol_id="amplitude-provenance-test",
            item_labels=("a", "b", "c", "d"),
            true_order_high_to_low=(0, 1, 2, 3),
            support_pairs_higher_lower=((0, 1),),
            support_blocks=1,
            query_blocks=1,
            human_targets={},
        )
        self.geometry = build_complete_graph_geometry(protocol)

    def test_vector_hodge_and_readout_commute_in_zero_sum_gauge(self):
        rng = np.random.default_rng(11)
        latent = rng.normal(size=(3, 4, 5))
        latent -= np.mean(latent, axis=1, keepdims=True)
        pure_gradient = np.einsum("ei,bih->beh", self.geometry.incidence, latent)
        np.testing.assert_allclose(
            vector_item_potentials(pure_gradient, self.geometry),
            latent,
            atol=1e-12,
        )

        fields = rng.normal(size=(3, len(self.geometry.pairs), 5))
        readout = rng.normal(size=5)

        item_vectors = vector_item_potentials(fields, self.geometry)

        self.assertEqual(item_vectors.shape, (3, 4, 5))
        np.testing.assert_allclose(
            np.sum(item_vectors, axis=1),
            0.0,
            atol=1e-12,
        )
        scalar_after_hodge = np.einsum("bih,h->bi", item_vectors, readout)
        scalar_before_hodge = hodge_potentials(
            np.einsum("beh,h->be", fields, readout), self.geometry
        )
        np.testing.assert_allclose(
            scalar_after_hodge,
            scalar_before_hodge,
            atol=1e-12,
        )

    def test_amplitude_ledger_preserves_both_multiplicative_identities(self):
        a_p = np.asarray([2.0, 3.0, 5.0])
        a_h = np.asarray([3.0, 6.0, 10.0])
        a_delta = np.asarray([1.5, 3.0, 8.0])
        a_n = np.asarray([1.8, 2.4, 10.0])
        w_norm = 4.0

        ledger = amplitude_ledger(
            a_p=a_p,
            a_h=a_h,
            a_delta=a_delta,
            a_n=a_n,
            w_norm=w_norm,
        )

        np.testing.assert_allclose(ledger["g_rec"], a_h / a_p)
        np.testing.assert_allclose(ledger["g_out"], a_delta / a_h)
        np.testing.assert_allclose(ledger["g_mix"], a_n / a_delta)
        np.testing.assert_allclose(
            a_p * ledger["g_rec"] * ledger["g_out"] * ledger["g_mix"],
            a_n,
            atol=1e-14,
        )
        np.testing.assert_allclose(
            ledger["g_out"], w_norm * ledger["rho_w"], atol=1e-14
        )
        self.assertTrue(np.all((ledger["rho_w"] >= 0.0) & (ledger["rho_w"] <= 1.0)))

    def test_zero_denominator_is_retained_without_epsilon(self):
        ledger = amplitude_ledger(
            a_p=np.asarray([0.0, 2.0]),
            a_h=np.asarray([1.0, 3.0]),
            a_delta=np.asarray([0.5, 1.5]),
            a_n=np.asarray([0.4, 1.2]),
            w_norm=2.0,
        )
        self.assertTrue(np.isnan(ledger["g_rec"][0]))
        self.assertAlmostEqual(ledger["g_rec"][1], 1.5)

    def test_elasticity_ledger_preserves_increment_and_mismatch_identities(self):
        z = np.linspace(-1.0, 1.0, 9)
        log_a_p = 0.8 * z + 0.2
        log_a_h = 1.1 * z - 0.4
        log_a_delta = 0.9 * z + 0.7
        log_a_n = 1.2 * z - 0.1

        ledger = elasticity_ledger(
            log_a_post=z,
            log_a_p=log_a_p,
            log_a_h=log_a_h,
            log_a_delta=log_a_delta,
            log_a_n=log_a_n,
        )

        self.assertAlmostEqual(ledger["beta_p"], 0.8)
        self.assertAlmostEqual(ledger["beta_h"], 1.1)
        self.assertAlmostEqual(ledger["beta_delta"], 0.9)
        self.assertAlmostEqual(ledger["beta_n"], 1.2)
        self.assertAlmostEqual(
            ledger["beta_n"],
            ledger["beta_p"]
            + ledger["delta_rec"]
            + ledger["delta_out"]
            + ledger["delta_mix"],
        )
        self.assertAlmostEqual(ledger["beta_mismatch"], ledger["beta_n"] - 1.0)
        self.assertAlmostEqual(
            ledger["beta_mismatch"],
            ols_slope(predictor=z, response=log_a_n - z),
        )

    def test_bootstrap_ols_recomputes_each_weighted_participant_fit(self):
        predictor = np.asarray([0.0, 0.5, 1.0, 1.5, 2.0])
        response = np.asarray([0.2, 0.4, 1.2, 1.1, 2.5])
        counts = np.asarray(
            [
                [1, 1, 1, 1, 1],
                [2, 1, 0, 1, 1],
                [0, 2, 1, 2, 0],
            ],
            dtype=np.float64,
        )

        observed = bootstrap_ols_slopes(
            predictor=predictor,
            response=response,
            bootstrap_counts=counts,
        )
        expected = np.asarray(
            [
                ols_slope(predictor=predictor, response=response, weights=row)
                for row in counts
            ]
        )
        np.testing.assert_allclose(observed, expected, atol=1e-14)

    def test_centered_bootstrap_moments_are_stable_under_large_offsets(self):
        values = 1e8 + np.linspace(-1.0, 1.0, 77)
        counts = np.ones((1, 77), dtype=np.float64)
        observed = bootstrap_correlation(values, values, counts)
        np.testing.assert_allclose(observed, np.asarray([1.0]), atol=1e-14)

    def test_nonfinite_bootstrap_draw_is_not_filtered(self):
        with self.assertRaises(NonInterpretableEstimate):
            interval_summary(1.0, np.asarray([0.9, np.nan, 1.1]))

    def test_atomic_layer_gate_detects_upstream_cancellation(self):
        equivalent = "equivalent"
        decisions = layer_elasticity_decision(
            {
                seed: {
                    "beta_P": "material_positive",
                    "delta_rec": "material_negative",
                    "delta_out": "material_positive",
                    "delta_mix": equivalent,
                }
                for seed in ("2104", "2105")
            }
        )
        self.assertEqual(decisions["outcome"], "coadapted_scale_fingerprint")
        self.assertEqual(
            {row["term"] for row in decisions["material_atomic_terms"]},
            {"beta_P", "delta_rec", "delta_out"},
        )

        projection = layer_elasticity_decision(
            {
                seed: {
                    "beta_P": equivalent,
                    "delta_rec": equivalent,
                    "delta_out": "material_positive",
                    "delta_mix": equivalent,
                }
                for seed in ("2104", "2105")
            }
        )
        self.assertEqual(projection["outcome"], "projection_alignment_fingerprint")

    @staticmethod
    def _decision_seed(statuses, *, integrity=True):
        metric = {"bootstrap": {"lower": 0.1}}
        return {
            "integrity": {"passed": integrity},
            "statistics": {
                "metrics": {
                    "Track_B_probability_slope_mismatch": metric,
                    "Y": metric,
                    "d_prob": metric,
                },
                "shape_model": {
                    "energy_explained": {"bootstrap": {"lower95": 0.95}},
                    "scale": {"bootstrap": {"lower95": 1.1}},
                },
                "statuses": statuses,
            },
        }

    def test_final_constant_and_internal_coadaptation_are_separate_axes(self):
        statuses = {
            "beta_P": "material_positive",
            "beta_N": "equivalent",
            "delta_rec": "material_negative",
            "delta_out": "equivalent",
            "delta_mix": "equivalent",
            "Y_on_coverage": "equivalent",
            "Y_on_certainty": "equivalent",
        }
        result = cross_seed_decision(
            {seed: self._decision_seed(dict(statuses)) for seed in ("2104", "2105")}
        )
        self.assertEqual(result["outcome"], "constant_calibration_fingerprint")
        self.assertEqual(
            result["final_comparator_fingerprint"],
            "constant_calibration_fingerprint",
        )
        self.assertEqual(
            result["layer_elasticity_fingerprint"],
            "coadapted_scale_fingerprint",
        )

    def test_integrity_gate_precedes_all_outcomes(self):
        statuses = {
            "beta_P": "equivalent",
            "beta_N": "equivalent",
            "delta_rec": "equivalent",
            "delta_out": "equivalent",
            "delta_mix": "equivalent",
            "Y_on_coverage": "equivalent",
            "Y_on_certainty": "equivalent",
        }
        result = cross_seed_decision(
            {
                "2104": self._decision_seed(statuses, integrity=False),
                "2105": self._decision_seed(statuses),
            }
        )
        self.assertEqual(result["outcome"], "noninterpretable_integrity_failure")

    def test_cross_network_outcome_requires_replication_without_pooling(self):
        heterogeneous = cross_network_outcome(
            {
                "2104": "functional_drive_fingerprint",
                "2105": "recurrent_expression_fingerprint",
            }
        )
        self.assertEqual(heterogeneous["outcome"], "heterogeneous_or_unresolved")
        self.assertEqual(
            heterogeneous["by_network"],
            {
                "2104": "functional_drive_fingerprint",
                "2105": "recurrent_expression_fingerprint",
            },
        )
        self.assertEqual(heterogeneous["network_population_inference"], "not_performed")

        replicated = cross_network_outcome(
            {
                "2104": "projection_alignment_fingerprint",
                "2105": "projection_alignment_fingerprint",
            }
        )
        self.assertEqual(replicated["outcome"], "projection_alignment_fingerprint")
