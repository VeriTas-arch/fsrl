import argparse
import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from fsrl.assembly_trajectory import build_complete_graph_geometry
from fsrl.curvature_gate_pilot import load_json
from fsrl.git_provenance import verify_git_registrations
from fsrl.global_policy_allocation_audit import (
    DEFAULT_IMPLEMENTATION_LOCK_PATH,
    DEFAULT_SPECIFICATION_PATH,
    INITIAL_IMPLEMENTATION_LOCK_PATH,
    NONINTERPRETABLE_ATTEMPT_PATH,
    UPSTREAM_OUTPUT_ROOT,
    _canonical_paths,
    _q_shape_rows_complete,
    allocation_tensor_view,
    correlation_summary,
    cross_network_analysis,
    edge_metadata,
    joint_model_statistics,
    pair_fingerprint_vectors,
    required_freeze_paths,
    seed_statistics,
    validate_sources,
    write_json_exclusive,
)
from fsrl.global_policy_amplitude_provenance import NonInterpretableEstimate
from fsrl.global_policy_field_reassembly import field_reassembly_estimands
from fsrl.ranking_protocol import load_ranking_protocol
from fsrl.study_registry import legacy_identifier, resolve_record


class GlobalPolicyAllocationAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.specification = load_json(DEFAULT_SPECIFICATION_PATH)
        cls.protocol = load_ranking_protocol(resolve_record("benchmarks/liu_v2.json"))
        cls.geometry = build_complete_graph_geometry(cls.protocol)
        cls.metadata = edge_metadata(cls.specification, cls.protocol, cls.geometry)

    def test_edge_contract_is_exact(self):
        self.assertEqual(len(self.metadata["pair_labels"]), 28)
        self.assertEqual(len(self.metadata["nonlearned_pair_labels"]), 20)
        self.assertEqual(self.metadata["distance_levels"].tolist(), [1, 2, 3, 4, 5, 6])
        self.assertEqual(
            self.metadata["distance_level_pair_counts"].tolist(), [6, 5, 2, 3, 2, 2]
        )
        self.assertAlmostEqual(self.metadata["distance_mean"], 2.8)
        self.assertAlmostEqual(self.metadata["distance_denominator"], 57.2)

    def test_exact_probability_bridge_sums_to_q_shape(self):
        rng = np.random.default_rng(11)
        neural_margin = rng.normal(size=(5, 28))
        posterior_margin = rng.normal(size=(5, 28))
        estimands = field_reassembly_estimands(
            neural_margin,
            posterior_margin,
            self.geometry,
            self.metadata["distances"],
            self.metadata["nonlearned"],
            0.25,
        )
        posterior = {
            "arrays": {
                "posterior_entropy": np.linspace(0.1, 0.5, 5),
                "coverage": np.linspace(0.5, 1.0, 5),
            }
        }
        arrays, integrity = allocation_tensor_view(
            estimands, posterior, self.metadata, 0.25
        )
        np.testing.assert_allclose(
            np.sum(arrays["q"], axis=1), arrays["q_shape"], atol=1e-12
        )
        np.testing.assert_allclose(
            np.sum(arrays["q_by_distance"], axis=1),
            arrays["q_shape"],
            atol=1e-12,
        )
        self.assertLess(integrity["q_sum_equals_q_shape_max_abs_error"], 1e-12)
        self.assertTrue(integrity["all_allocation_arrays_finite"])

    def test_pair_bridge_does_not_turn_linear_distance_into_pair_identity(self):
        distances = self.metadata["selected_distances"]
        weights = self.metadata["distance_weights"]
        subjects = 4
        delta_p = np.stack(
            [0.1 * row + (0.2 + 0.03 * row) * distances for row in range(subjects)]
        )
        delta = 2.0 * delta_p
        q = delta_p * weights[None, :]
        counts = np.tile(np.ones(subjects), (7, 1))
        vectors = pair_fingerprint_vectors(
            delta, delta_p, q, distances, weights, counts
        )
        np.testing.assert_allclose(vectors["point"]["r_delta_p"], 0.0, atol=1e-12)
        np.testing.assert_allclose(vectors["point"]["r_q"], 0.0, atol=1e-12)
        self.assertLess(
            vectors["integrity"]["q_ledger_reconstruction_max_abs_error"], 1e-12
        )

    def test_joint_pair_fe_model_recovers_conditional_coefficients_and_q_identity(self):
        subjects = 77
        pairs = 20
        uncertainty = np.linspace(0.05, 0.95, subjects)
        coverage = 0.6 + 0.2 * np.sin(np.linspace(0.0, 5.0, subjects))
        uncertainty_z = (uncertainty - np.mean(uncertainty)) / np.std(uncertainty)
        coverage_z = (coverage - np.mean(coverage)) / np.std(coverage)
        pair_effect = np.linspace(-0.2, 0.2, pairs)[None, :]
        delta = pair_effect + 0.4 * uncertainty_z[:, None] - 0.2 * coverage_z[:, None]
        q = (
            0.1 * pair_effect
            + 0.03 * uncertainty_z[:, None]
            + 0.01 * coverage_z[:, None]
        )
        q_shape = np.sum(q, axis=1)
        counts = (
            np.random.default_rng(9)
            .multinomial(
                subjects,
                np.full(subjects, 1.0 / subjects),
                size=200,
            )
            .astype(np.float64)
        )
        public, _ = joint_model_statistics(
            delta, q, q_shape, uncertainty, coverage, counts
        )
        self.assertAlmostEqual(public["summaries"]["beta_delta_U"]["point"], 0.4)
        self.assertAlmostEqual(public["summaries"]["beta_delta_C"]["point"], -0.2)
        self.assertAlmostEqual(public["summaries"]["beta_q_U"]["point"], 0.03)
        self.assertAlmostEqual(public["summaries"]["beta_q_C"]["point"], 0.01)
        integrity = public["integrity"]
        self.assertEqual(integrity["delta_point_full_rank"], 22)
        self.assertEqual(integrity["q_minimum_bootstrap_reduced_rank"], 3)
        self.assertLess(
            integrity["q_coefficient_identity_bootstrap_max_abs_error"], 1e-10
        )

        weights = self.metadata["distance_weights"]
        q_by_distance = np.stack(
            [
                np.sum(q[:, self.metadata["selected_distances"] == level], axis=1)
                for level in self.metadata["distance_levels"]
            ],
            axis=1,
        )
        statistics, _ = seed_statistics(
            self.specification,
            2106,
            {
                "delta_g_canonical": delta,
                "delta": delta,
                "delta_p": q / weights[None, :],
                "q": q,
                "q_by_distance": q_by_distance,
                "delta_distance_slope": delta @ weights,
                "q_shape": q_shape,
                "uncertainty": uncertainty,
                "coverage": coverage,
            },
            self.metadata,
        )
        self.assertEqual(
            set(statistics["q_by_distance_summaries"]),
            {"1", "2", "3", "4", "5", "6"},
        )
        self.assertLess(
            statistics["integrity"]["q_bootstrap_sum_equals_q_shape_max_abs_error"],
            1e-10,
        )
        self.assertLess(
            statistics["integrity"][
                "q_by_distance_bootstrap_sum_equals_q_shape_max_abs_error"
            ],
            1e-10,
        )

    def test_joint_model_nonfinite_input_fails_before_linear_algebra(self):
        subjects = 77
        delta = np.zeros((subjects, 20))
        delta[0, 0] = np.nan
        with self.assertRaisesRegex(NonInterpretableEstimate, "nonfinite"):
            joint_model_statistics(
                delta,
                np.zeros((subjects, 20)),
                np.zeros(subjects),
                np.linspace(0.0, 1.0, subjects),
                np.sin(np.linspace(0.0, 3.0, subjects)),
                np.ones((2, subjects)),
            )

    def test_correlation_degeneracy_is_retained(self):
        result = correlation_summary(
            np.ones(4),
            np.arange(4),
            np.ones((5, 4)),
            np.tile(np.arange(4), (5, 1)),
        )
        self.assertEqual(result["status"], "unresolved_degenerate")
        self.assertEqual(result["bootstrap"]["degenerate_draws"], 5)

    def test_correlation_nonfinite_input_is_retained_as_unresolved(self):
        first = np.tile(np.arange(4, dtype=np.float64), (5, 1))
        second = first.copy()
        second[2, 1] = np.nan
        result = correlation_summary(first[0], second[0], first, second)
        self.assertEqual(result["status"], "unresolved_degenerate")
        self.assertEqual(result["bootstrap"]["degenerate_draws"], 1)
        self.assertEqual(result["bootstrap"]["nonfinite_draws"], 1)
        self.assertEqual(result["bootstrap"]["zero_norm_draws"], 0)

    @staticmethod
    def _cross_seed_fixture(state_status: str):
        seeds = {}
        internal = {}
        pair = np.linspace(-1.0, 1.0, 20)
        distance = np.asarray([-0.2, -0.1, 0.0, 0.1, 0.2, 0.3])
        for seed in ("2106", "2107"):
            statuses = {
                "beta_delta_distance": "resolved_positive",
                "beta_delta_U": state_status,
                "beta_q_U": state_status,
                "beta_delta_C": "unresolved",
                "beta_q_C": "unresolved",
            }
            seeds[seed] = {
                "integrity": {"passed": True},
                "statistics": {"statuses": statuses},
            }
            internal[seed] = {
                "pair_point": {
                    "r_delta": pair,
                    "r_q": pair,
                    "mu_delta": pair,
                    "mu_delta_p": pair,
                    "mu_q": pair,
                },
                "pair_bootstrap": {
                    "r_delta": np.tile(pair, (10, 1)),
                    "r_q": np.tile(pair, (10, 1)),
                    "mu_delta": np.tile(pair, (10, 1)),
                    "mu_delta_p": np.tile(pair, (10, 1)),
                    "mu_q": np.tile(pair, (10, 1)),
                },
                "q_distance_point": distance,
                "q_distance_bootstrap": np.tile(distance, (10, 1)),
                "uncertainty": np.linspace(0.0, 1.0, 77),
                "coverage": np.linspace(0.2, 0.9, 77),
            }
        return seeds, internal

    def test_outcome_scope_distinguishes_state_from_structural_localization(self):
        seeds, internal = self._cross_seed_fixture("resolved_positive")
        decision, integrity = cross_network_analysis(
            self.specification, seeds, internal
        )
        self.assertTrue(integrity["passed"])
        self.assertEqual(decision["outcome"], "policy_effective_allocation_localized")
        self.assertEqual(decision["localization_scope"], "both")
        self.assertEqual(
            decision["conditional_next_step"],
            "prospective_state_dependent_P_T_generation_question",
        )

        seeds, internal = self._cross_seed_fixture("unresolved")
        decision, _ = cross_network_analysis(self.specification, seeds, internal)
        self.assertEqual(decision["localization_scope"], "structural_only")
        self.assertEqual(
            decision["conditional_next_step"], "prospective_comparator_adequacy"
        )

    @staticmethod
    def _disable_structural_axes(internal):
        for name in (
            "r_delta",
            "r_q",
            "mu_delta",
            "mu_delta_p",
            "mu_q",
        ):
            internal["2107"]["pair_point"][name] *= -1.0
            internal["2107"]["pair_bootstrap"][name] *= -1.0
        internal["2107"]["q_distance_point"] *= -1.0
        internal["2107"]["q_distance_bootstrap"] *= -1.0

    def test_outcome_tree_preserves_opposite_bridge_and_unresolved_cases(self):
        seeds, internal = self._cross_seed_fixture("resolved_positive")
        self._disable_structural_axes(internal)
        for seed in seeds.values():
            seed["statistics"]["statuses"]["beta_q_U"] = "resolved_negative"
        decision, _ = cross_network_analysis(self.specification, seeds, internal)
        self.assertEqual(decision["outcome"], "field_structure_without_policy_bridge")
        self.assertFalse(decision["axes"]["posterior_uncertainty"]["policy_effective"])

        seeds, internal = self._cross_seed_fixture("unresolved")
        self._disable_structural_axes(internal)
        for seed in seeds.values():
            seed["statistics"]["statuses"]["beta_q_U"] = "resolved_positive"
        decision, _ = cross_network_analysis(self.specification, seeds, internal)
        self.assertEqual(decision["outcome"], "field_structure_without_policy_bridge")

        seeds, internal = self._cross_seed_fixture("unresolved")
        for seed in seeds.values():
            seed["statistics"]["statuses"]["beta_delta_distance"] = "unresolved"
        for network in internal.values():
            for group in ("pair_point", "pair_bootstrap"):
                for name in network[group]:
                    network[group][name] = np.zeros_like(network[group][name])
            network["q_distance_point"] = np.zeros(6)
            network["q_distance_bootstrap"] = np.zeros((10, 6))
        decision, _ = cross_network_analysis(self.specification, seeds, internal)
        self.assertEqual(decision["outcome"], "no_stable_allocation_localization")

    def test_cross_network_predictor_identity_is_integrity_first(self):
        seeds, internal = self._cross_seed_fixture("resolved_positive")
        internal["2107"]["uncertainty"][0] += 0.01
        decision, integrity = cross_network_analysis(
            self.specification, seeds, internal
        )
        self.assertFalse(integrity["passed"])
        self.assertEqual(decision["outcome"], "noninterpretable_integrity_failure")
        self.assertEqual(decision["axes"], "not_evaluated")

    def test_formal_paths_reject_noncanonical_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            parsed = argparse.Namespace(
                stage="evaluate",
                specification=DEFAULT_SPECIFICATION_PATH,
                implementation_lock=DEFAULT_IMPLEMENTATION_LOCK_PATH,
                output_root=UPSTREAM_OUTPUT_ROOT,
                result=Path(directory) / "result.json",
            )
            _canonical_paths(parsed)
            parsed.output_root = Path("/tmp/not-the-frozen-output")
            with self.assertRaisesRegex(RuntimeError, "canonical output_root"):
                _canonical_paths(parsed)

    def test_repair_git_gate_requires_the_full_provenance_chain(self):
        self.assertEqual(
            required_freeze_paths(
                DEFAULT_SPECIFICATION_PATH, DEFAULT_IMPLEMENTATION_LOCK_PATH
            ),
            (
                DEFAULT_SPECIFICATION_PATH,
                DEFAULT_IMPLEMENTATION_LOCK_PATH,
                INITIAL_IMPLEMENTATION_LOCK_PATH,
                NONINTERPRETABLE_ATTEMPT_PATH,
            ),
        )

    def test_result_writer_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            write_json_exclusive(path, {"value": 1})
            with self.assertRaises(FileExistsError):
                write_json_exclusive(path, {"value": 2})

    def test_result_writer_serializes_before_exclusive_create(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            with self.assertRaises(TypeError):
                write_json_exclusive(path, {"value": np.bool_(True)})
            self.assertFalse(path.exists())

    def test_prerequisite_row_check_returns_builtin_bool(self):
        complete = _q_shape_rows_complete(np.ones(77), 77)
        self.assertIs(type(complete), bool)
        self.assertTrue(complete)

    def test_source_lock_is_complete_and_fails_closed(self):
        lock = load_json(DEFAULT_IMPLEMENTATION_LOCK_PATH)
        historical = verify_git_registrations(
            Path(__file__).resolve().parents[1],
            "d216bdb8f06f81dd3d2aef74a01d48c7f2bd279a",
            {
                **load_json(DEFAULT_SPECIFICATION_PATH)["registered_sources"],
                "audit_specification": {
                    "path": legacy_identifier(DEFAULT_SPECIFICATION_PATH),
                    "sha256": lock["audit_specification_sha256"],
                },
                "superseded_implementation_lock": lock["supersedes"],
                "noninterpretable_attempt": lock["noninterpretable_attempt"],
                **lock["implementation_sources"],
                **lock["reused_frozen_sources"],
            },
        )
        self.assertTrue(historical["passed"])
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            changed_source = copy.deepcopy(lock)
            changed_source["reused_frozen_sources"]["hash_validation"]["sha256"] = (
                "0" * 64
            )
            changed_source_path = directory / "changed-source.json"
            write_json_exclusive(changed_source_path, changed_source)
            with self.assertRaisesRegex(RuntimeError, "registered source"):
                validate_sources(DEFAULT_SPECIFICATION_PATH, changed_source_path)

            missing_upstream = copy.deepcopy(lock)
            missing_upstream["upstream_fingerprint"].pop("report")
            missing_upstream_path = directory / "missing-upstream.json"
            write_json_exclusive(missing_upstream_path, missing_upstream)
            with self.assertRaisesRegex(RuntimeError, "upstream fingerprint"):
                validate_sources(DEFAULT_SPECIFICATION_PATH, missing_upstream_path)


if __name__ == "__main__":
    unittest.main()
