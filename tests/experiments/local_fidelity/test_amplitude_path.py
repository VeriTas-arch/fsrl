import unittest

import numpy as np
import torch

from fsrl.experiments.assembly.diagnostics import load_json
from fsrl.experiments.assembly.trajectory import bootstrap_counts
from fsrl.experiments.local_fidelity.amplitude_path import (
    crossing_regime,
    mean_sign_change_bracket,
    quadratic_coefficient,
    robust_transition_bracket,
    select_v2_outcome,
    subject_crossing_summary,
)
from fsrl.experiments.local_fidelity.hidden_residual import validate_registered_sources
from fsrl.infra.study_registry import registered_file_sha256, resolve_record


class OperatorAmplitudePathTests(unittest.TestCase):
    def test_quadratic_coefficient_matches_small_amplitude_expansion(self):
        baseline = torch.tensor([-1.2, -0.1, 0.4, 1.1], dtype=torch.float64)
        action = torch.tensor([0.7, -0.3, 0.5, -0.8], dtype=torch.float64)
        sensitivity = 1.0 - torch.tanh(baseline).square()
        jacobian = sensitivity * action
        curvature = quadratic_coefficient(baseline, action)
        epsilon = 1e-4
        exact = torch.tanh(baseline + epsilon * action) - torch.tanh(baseline)
        observed = (exact - epsilon * jacobian) / epsilon**2
        torch.testing.assert_close(observed, curvature, atol=5e-5, rtol=5e-5)

    def test_fixed_grid_crossing_brackets_and_regimes(self):
        amplitudes = np.asarray([0.0, 0.05, 0.10, 0.15, 0.20])
        means = np.asarray([0.0, 0.2, 0.1, -0.1, -0.2])
        bracket = mean_sign_change_bracket(amplitudes, means)
        self.assertEqual(
            (bracket["lambda_minus"], bracket["lambda_plus"]), (0.10, 0.15)
        )
        self.assertEqual(crossing_regime(bracket), "early")
        statuses = [
            "structural_zero",
            "robust_positive",
            "robust_positive",
            "unresolved",
            "robust_negative",
        ]
        self.assertEqual(
            robust_transition_bracket(amplitudes, statuses),
            {
                "last_robust_positive_lambda": 0.10,
                "first_later_robust_negative_lambda": 0.20,
            },
        )

    def test_subject_crossing_uses_all_retained_subjects(self):
        amplitudes = np.asarray([0.0, 0.05, 0.10, 0.15, 0.20])
        values = np.asarray(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.2, 0.1, -0.2, -0.1],
                [0.1, 0.2, -0.1, 0.1],
                [-0.1, 0.3, -0.2, -0.1],
                [-0.2, 0.4, -0.3, -0.2],
            ]
        )
        retained = np.ones(4, dtype=bool)
        counts = bootstrap_counts(np.random.default_rng(4), 100, 4)
        result = subject_crossing_summary(
            values, retained, amplitudes, counts, interval=0.95
        )
        self.assertEqual(result["retained_subjects"], 4)
        self.assertAlmostEqual(result["crossing_proportion"]["mean"], 0.5)
        self.assertEqual(result["crossing_bracket_counts"], {"0.10-0.15": 2})
        self.assertAlmostEqual(result["never_positive_proportion"]["mean"], 0.25)
        self.assertAlmostEqual(
            result["positive_without_crossing_proportion"]["mean"], 0.25
        )

    def test_v2_selection_separates_sparse_and_widespread_crossings(self):
        self.assertEqual(
            select_v2_outcome("late", 1),
            "isolated_or_sparse_late_crossing_relation_conditioned_amplitude_v2",
        )
        self.assertEqual(
            select_v2_outcome("intermediate", 5),
            "widespread_early_or_intermediate_crossing_near_linear_residual_v2",
        )
        self.assertEqual(
            select_v2_outcome("no_crossing", 0),
            "integrity_contradiction_no_H_greater_A_crossing",
        )

    def test_registered_sources_are_immutable(self):
        specification = load_json(
            resolve_record("benchmarks/operator_amplitude_path_v1.json")
        )
        validation = validate_registered_sources(specification)
        self.assertEqual(len(validation["pilot_artifacts"]), 2)

    def test_registered_result_is_complete_and_source_locked(self):
        result = load_json(resolve_record("results/operator_amplitude_path_v1.json"))
        self.assertFalse(result["formal_seed_access"])
        self.assertEqual(set(result["seed_results"]), {"1901", "1902"})
        self.assertEqual(
            result["implementation"]["sha256"],
            registered_file_sha256(
                result["implementation"]["path"],
                result["implementation"]["sha256"],
            ),
        )
        self.assertEqual(
            result["specification"]["sha256"],
            registered_file_sha256(
                result["specification"]["path"],
                result["specification"]["sha256"],
            ),
        )
        overall = result["overall_diagnosis"]
        self.assertEqual(overall["replicated_mean_crossing_relations"], ["H>A"])
        self.assertEqual(overall["union_mean_crossing_relations"], ["H>A"])
        self.assertEqual(
            overall["v2_selection"],
            "heterogeneous_or_nonreplicated_crossings_register_online_relation_conditioned_v2",
        )
        expected_brackets = {"1901": (0.55, 0.60), "1902": (0.60, 0.65)}
        for seed, row in result["seed_results"].items():
            self.assertEqual(len(row["lambda_grid"]), 21)
            bracket = row["prospective_H_greater_A"]["mean_sign_change_bracket"]
            self.assertEqual(
                (bracket["lambda_minus"], bracket["lambda_plus"]),
                expected_brackets[seed],
            )
            self.assertTrue(row["validation"]["zero_controls_pass"])
            self.assertEqual(
                row["validation"]["prior_J_and_H_summary_consistency_max_abs_error"],
                0.0,
            )
            curvature = next(
                relation
                for relation in row["curvature"]["per_relation"]
                if relation["relation_label"] == "H>A"
            )
            self.assertLess(curvature["direct_correctness"]["bootstrap"]["upper"], 0.0)


if __name__ == "__main__":
    unittest.main()
