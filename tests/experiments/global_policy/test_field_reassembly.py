import unittest

import numpy as np

from fsrl.experiments.assembly.trajectory import build_complete_graph_geometry
from fsrl.experiments.global_policy.field_reassembly import (
    NonInterpretableEstimate,
    classify_status,
    cross_seed_decision,
    decompose_field,
    field_reassembly_estimands,
    summarize_estimand,
)
from fsrl.experiments.global_policy.slope_localization import subject_slopes
from fsrl.experiments.local_fidelity.behavior_attribution import exact_probability
from fsrl.infra.study_registry import resolve_record
from fsrl.tasks.registered_protocol import load_ranking_protocol


class GlobalPolicyFieldReassemblyTests(unittest.TestCase):
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
        self.nonlearned = np.asarray(
            [pair not in self.protocol.learned_pairs for pair in self.geometry.pairs],
            dtype=bool,
        )

        rng = np.random.default_rng(41)
        subjects = 7
        potential_n = rng.normal(scale=0.7, size=(subjects, self.protocol.n_items))
        potential_p = rng.normal(scale=1.1, size=(subjects, self.protocol.n_items))
        potential_n -= np.mean(potential_n, axis=1, keepdims=True)
        potential_p -= np.mean(potential_p, axis=1, keepdims=True)
        self.potential_n = potential_n
        self.potential_p = potential_p
        self.g_n = potential_n @ self.geometry.incidence.T
        self.g_p = potential_p @ self.geometry.incidence.T

        raw_n = rng.normal(scale=0.35, size=self.g_n.shape)
        raw_p = rng.normal(scale=0.25, size=self.g_p.shape)
        self.c_n = raw_n - raw_n @ self.geometry.projection.T
        self.c_p = raw_p - raw_p @ self.geometry.projection.T
        self.neural_margin = self.g_n + self.c_n
        self.posterior_margin = self.g_p + self.c_p
        self.estimands = field_reassembly_estimands(
            self.neural_margin,
            self.posterior_margin,
            self.geometry,
            self.distances,
            self.nonlearned,
            0.25,
        )

    def test_decompose_field_reconstructs_zero_sum_hodge_components(self):
        observed = decompose_field(self.neural_margin, self.geometry)

        np.testing.assert_allclose(observed["potential"], self.potential_n, atol=1e-12)
        np.testing.assert_allclose(observed["gradient"], self.g_n, atol=1e-12)
        np.testing.assert_allclose(observed["residual"], self.c_n, atol=1e-12)
        np.testing.assert_allclose(
            observed["gradient"] + observed["residual"],
            self.neural_margin,
            atol=1e-14,
        )
        np.testing.assert_allclose(
            np.sum(observed["potential"], axis=1), 0.0, atol=1e-12
        )
        np.testing.assert_allclose(
            observed["residual"] @ self.geometry.incidence, 0.0, atol=1e-12
        )
        np.testing.assert_allclose(observed["reconstruction_error"], 0.0)
        self.assertLess(float(np.max(observed["zero_sum_gauge_error"])), 1e-12)
        self.assertLess(float(np.max(observed["residual_orthogonality_error"])), 1e-12)

    def test_four_cells_and_all_registered_identities_are_exact(self):
        result = self.estimands
        expected_names = {
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
        }
        for name in expected_names:
            self.assertEqual(np.asarray(result[name]).shape, (7,), name)

        np.testing.assert_allclose(result["D"], result["A"] + result["R"])
        np.testing.assert_allclose(result["D"], result["Delta_A"] + result["C_A"])
        np.testing.assert_allclose(result["D"], result["Delta_R"] + result["C_R"])
        np.testing.assert_allclose(result["D"], result["Q_shape"] + result["C_shape"])
        np.testing.assert_allclose(result["I"], result["Delta_A"] - result["C_R"])
        np.testing.assert_allclose(result["I"], result["Delta_R"] - result["C_A"])
        np.testing.assert_allclose(result["Delta_A"], result["A"] + 0.5 * result["I"])
        np.testing.assert_allclose(result["C_A"], result["R"] - 0.5 * result["I"])
        np.testing.assert_allclose(result["Delta_R"], result["R"] + 0.5 * result["I"])
        np.testing.assert_allclose(result["C_R"], result["A"] - 0.5 * result["I"])
        np.testing.assert_allclose(
            result["Delta_A"], result["Q_shape"] + result["Q_amp"]
        )
        np.testing.assert_allclose(result["C_shape"], result["C_A"] + result["Q_amp"])
        for name, errors in result["factorial_identity_errors"].items():
            self.assertLess(float(np.max(errors)), 1e-14, name)

    def test_natural_cells_reproduce_fixed_probability_slopes(self):
        correct_sign = self.geometry.true_sign[None, :]
        expected_nn = subject_slopes(
            exact_probability(correct_sign * self.neural_margin, 0.25),
            self.distances,
            self.nonlearned,
        )
        expected_pp = subject_slopes(
            exact_probability(correct_sign * self.posterior_margin, 0.25),
            self.distances,
            self.nonlearned,
        )

        np.testing.assert_allclose(self.estimands["S_NN"], expected_nn, atol=1e-14)
        np.testing.assert_allclose(self.estimands["S_PP"], expected_pp, atol=1e-14)
        np.testing.assert_allclose(
            self.estimands["D"], expected_nn - expected_pp, atol=1e-14
        )
        np.testing.assert_allclose(
            self.estimands["fields"]["NN"], self.neural_margin, atol=1e-14
        )
        np.testing.assert_allclose(
            self.estimands["fields"]["PP"], self.posterior_margin, atol=1e-14
        )

    def test_pre_sigmoid_factorial_interaction_is_exactly_zero(self):
        result = self.estimands
        fields = result["fields"]
        margin_field_i = fields["NN"] - fields["PN"] - fields["NP"] + fields["PP"]

        np.testing.assert_allclose(margin_field_i, 0.0, atol=1e-14)
        np.testing.assert_allclose(result["margin_I"], 0.0, atol=1e-14)
        np.testing.assert_allclose(
            result["margin_field_interaction_error"], 0.0, atol=1e-14
        )

    def test_norm_match_holds_additive_norm_and_full_margin_energy(self):
        result = self.estimands
        decompositions = result["decompositions"]
        scale = result["posterior_to_neural_scale_k"]
        expected_tilde = scale[:, None] * decompositions["P"]["gradient"]

        np.testing.assert_allclose(
            result["fields"]["tildePN"],
            expected_tilde + decompositions["N"]["residual"],
            atol=1e-14,
        )
        np.testing.assert_allclose(
            result["norm_g_P_tilde"], result["norm_g_N"], atol=1e-12
        )
        np.testing.assert_allclose(
            np.sum(result["fields"]["tildePN"] ** 2, axis=1),
            np.sum(result["fields"]["NN"] ** 2, axis=1),
            atol=1e-11,
        )
        self.assertLess(float(np.max(result["norm_match_norm_error"])), 1e-12)
        self.assertLess(float(np.max(result["norm_match_energy_error"])), 1e-11)

    def test_complete_graph_sqrt_eight_amplitude_bridge(self):
        result = self.estimands
        potential_n = result["decompositions"]["N"]["potential"]
        potential_p = result["decompositions"]["P"]["potential"]

        np.testing.assert_allclose(
            result["a_N_bridge"], np.linalg.norm(potential_n, axis=1), atol=1e-12
        )
        np.testing.assert_allclose(
            result["a_post_bridge"],
            np.linalg.norm(potential_p, axis=1),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            result["norm_g_N"], np.sqrt(8.0) * result["a_N_bridge"], atol=1e-12
        )
        np.testing.assert_allclose(
            result["norm_g_P"],
            np.sqrt(8.0) * result["a_post_bridge"],
            atol=1e-12,
        )
        np.testing.assert_allclose(
            result["Y_bridge"],
            np.log(result["a_N_bridge"] / result["a_post_bridge"]),
            atol=1e-14,
        )

    def test_zero_additive_norm_is_not_repaired_with_epsilon(self):
        pure_residual = self.c_p.copy()
        with self.assertRaises(NonInterpretableEstimate):
            field_reassembly_estimands(
                self.neural_margin,
                pure_residual,
                self.geometry,
                self.distances,
                self.nonlearned,
            )
        with self.assertRaises(NonInterpretableEstimate):
            field_reassembly_estimands(
                self.c_n,
                self.posterior_margin,
                self.geometry,
                self.distances,
                self.nonlearned,
            )

    @staticmethod
    def _summary(lower95, upper95, lower90, upper90):
        return {
            "bootstrap": {
                "lower95": lower95,
                "upper95": upper95,
                "lower90": lower90,
                "upper90": upper90,
            }
        }

    def test_statuses_are_mutually_exclusive_at_point_zero_zero_five(self):
        self.assertEqual(
            classify_status(self._summary(0.006, 0.020, 0.007, 0.018)),
            "material_positive",
        )
        self.assertEqual(
            classify_status(self._summary(-0.020, -0.006, -0.018, -0.007)),
            "material_negative",
        )
        self.assertEqual(
            classify_status(self._summary(0.001, 0.004, 0.0015, 0.0035)),
            "equivalent",
        )
        self.assertEqual(
            classify_status(self._summary(0.001, 0.010, 0.002, 0.008)),
            "unresolved",
        )
        self.assertEqual(
            classify_status(self._summary(0.005, 0.010, 0.006, 0.009)),
            "unresolved",
        )
        self.assertEqual(
            classify_status(self._summary(-0.010, -0.005, -0.009, -0.006)),
            "unresolved",
        )

    def test_nonfinite_participant_or_bootstrap_draw_is_never_discarded(self):
        counts = np.asarray([[1.0, 1.0], [2.0, 0.0]])
        with self.assertRaises(NonInterpretableEstimate):
            summarize_estimand(np.asarray([1.0, np.nan]), counts)

        counts_with_undefined_draw = np.asarray([[1.0, 1.0], [0.0, 0.0]])
        with self.assertRaises(NonInterpretableEstimate):
            summarize_estimand(np.asarray([1.0, 2.0]), counts_with_undefined_draw)

    @staticmethod
    def _decision_seed(statuses, *, integrity=True, d_lower95=0.02):
        defaults = {
            "A": "unresolved",
            "R": "unresolved",
            "I": "unresolved",
            "Delta_A": "unresolved",
            "C_A": "unresolved",
            "Delta_R": "unresolved",
            "C_R": "unresolved",
            "Q_shape": "unresolved",
            "C_shape": "unresolved",
            "Q_amp": "unresolved",
        }
        return {
            "integrity": {"passed": integrity},
            "statistics": {
                "summaries": {
                    "D": {"bootstrap": {"lower95": d_lower95}},
                },
                "statuses": {**defaults, **statuses},
            },
        }

    def test_field_source_fingerprint_priority_is_mutually_exclusive(self):
        additive_only = {
            "Delta_A": "material_positive",
            "C_A": "equivalent",
        }
        result = cross_seed_decision(
            {seed: self._decision_seed(additive_only) for seed in ("2104", "2105")}
        )
        self.assertEqual(result["outcome"], "additive_replacement_only")

        residual_only = {
            "Delta_R": "material_positive",
            "C_R": "equivalent",
        }
        result = cross_seed_decision(
            {seed: self._decision_seed(residual_only) for seed in ("2104", "2105")}
        )
        self.assertEqual(result["outcome"], "residual_replacement_only")

        both_replacements = {
            "Delta_A": "material_positive",
            "C_A": "equivalent",
            "Delta_R": "material_positive",
            "C_R": "equivalent",
        }
        result = cross_seed_decision(
            {seed: self._decision_seed(both_replacements) for seed in ("2104", "2105")}
        )
        self.assertEqual(result["outcome"], "both_replacements_sufficient")

        both_material = {
            "Delta_A": "material_positive",
            "C_A": "material_positive",
            "Delta_R": "material_positive",
            "C_R": "material_positive",
        }
        result = cross_seed_decision(
            {seed: self._decision_seed(both_material) for seed in ("2104", "2105")}
        )
        self.assertEqual(result["outcome"], "both_components_material")

    def test_cross_seed_decision_never_pools_disagreeing_networks(self):
        additive = {
            "A": "material_positive",
            "Delta_A": "material_positive",
            "C_A": "equivalent",
        }
        residual = {
            "A": "equivalent",
            "Delta_R": "material_positive",
            "C_R": "equivalent",
        }
        result = cross_seed_decision(
            {
                "2104": self._decision_seed(additive),
                "2105": self._decision_seed(residual),
            }
        )

        self.assertEqual(result["outcome"], "mixed_or_unresolved")
        self.assertEqual(
            result["main_effect_descriptions"]["A"]["status"],
            "heterogeneous_or_unresolved",
        )
        self.assertEqual(result["network_population_inference"], "not_performed")

    def test_integrity_and_premise_gates_precede_every_fingerprint(self):
        statuses = {
            "Delta_A": "material_positive",
            "C_A": "equivalent",
        }
        integrity = cross_seed_decision(
            {
                "2104": self._decision_seed(statuses, integrity=False),
                "2105": self._decision_seed(statuses),
            }
        )
        self.assertEqual(integrity["outcome"], "noninterpretable_integrity_failure")

        premise = cross_seed_decision(
            {
                "2104": self._decision_seed(statuses, d_lower95=-0.001),
                "2105": self._decision_seed(statuses),
            }
        )
        self.assertEqual(premise["outcome"], "premise_not_confirmed")

    def test_norm_matched_shape_axis_requires_replicated_reduction_and_closure(self):
        sufficient = {
            "Q_shape": "material_positive",
            "C_shape": "equivalent",
            "Q_amp": "material_positive",
        }
        result = cross_seed_decision(
            {seed: self._decision_seed(sufficient) for seed in ("2104", "2105")}
        )
        self.assertEqual(
            result["norm_matched_shape_axis"]["outcome"],
            "norm_matched_shape_replacement_sufficient",
        )

        heterogeneous = cross_seed_decision(
            {
                "2104": self._decision_seed(sufficient),
                "2105": self._decision_seed({**sufficient, "C_shape": "unresolved"}),
            }
        )
        self.assertEqual(
            heterogeneous["norm_matched_shape_axis"]["outcome"],
            "mixed_or_unresolved",
        )


if __name__ == "__main__":
    unittest.main()
