"""Synthetic scoring, undefined cohorts and strict admission semantics."""

import copy
import unittest
from unittest.mock import patch

import numpy as np

from fsrl.experiments.score_circuit.analysis import endpoint_vectors, summarize
from fsrl.experiments.score_circuit.decisions import behavior_preservation, decide_fit
from fsrl.experiments.score_circuit.evidence import (
    input_records,
    parameters,
    specification,
)
from fsrl.experiments.score_circuit.verification import check_estimate, manual_endpoints
from fsrl.tasks.protocol import RankingProtocol


class AnalysisTests(unittest.TestCase):
    def test_frozen_input_witnesses_without_evaluation(self):
        self.assertEqual(specification()["seeds"], [2111, 2112, 2113])
        self.assertTrue(input_records())
        self.assertEqual(set(parameters()), {"2111", "2112", "2113"})

    def test_independent_scoring_with_undefined_groups(self):
        protocol = RankingProtocol(
            "synthetic", ("a", "b", "c"), (2, 0, 1), ((2, 0),), 1, 1, {}
        )
        pairs = np.asarray([[0, 1], [1, 0], [0, 2], [2, 0], [1, 2], [2, 1]])
        inputs = {
            "liu__inputs__query_pairs": np.tile(pairs[None], (2, 1, 1)),
            "liu__inputs__support_pairs": np.asarray([[[2, 0]], [[0, 2]]]),
            "liu__inputs__retention": np.asarray([[1, 0]]),
            "liu__retention": np.asarray([[True], [False]]),
            "generic__groups__4__episode_indices": np.asarray([0, 1]),
            "generic__signs": np.asarray([[1, -1, 1], [-1, 1, -1]]),
            "generic__learned": np.asarray(
                [[True, False, False], [False, True, False]]
            ),
        }
        margins = {
            "generic": np.asarray([[0.0, -0.5, 0.7], [0.2, 0.8, -0.9]]),
            "liu": np.asarray(
                [[1, -1, -0.4, 0.4, -0.7, 0.7], [-0.5, 0.5, -0.2, 0.2, -0.3, 0.3]]
            ),
        }
        with (
            patch(
                "fsrl.experiments.score_circuit.analysis.load_registered_protocol",
                return_value=protocol,
            ),
            patch(
                "fsrl.experiments.score_circuit.verification.load_registered_protocol",
                return_value=protocol,
            ),
        ):
            primary = endpoint_vectors(margins, inputs)
            independent = manual_endpoints(
                {
                    "generic_4__margin": margins["generic"],
                    "liu__margin": margins["liu"],
                },
                inputs,
            )
        for name in primary:
            np.testing.assert_allclose(primary[name], independent[name], equal_nan=True)
        summary = summarize(primary, 2111)
        for name, value in independent.items():
            self.assertLess(check_estimate(value, summary[name], 2111), 1e-10)

    def test_equivalence_requires_whole_interval(self):
        case = {
            "paired_differences": {
                "liu/probability/overall": {
                    "mean": 0.0,
                    "bootstrap": {"lower": -0.02, "upper": 0.001},
                }
            },
            "endpoints": {
                f"generic/exact_decision/{group}": {"mean": 0.9}
                for group in ("learned", "nonlearned")
            },
        }
        competence = {"generic_learned": 0.8, "generic_nonlearned": 0.7}
        self.assertFalse(behavior_preservation(case, competence))
        case["paired_differences"]["liu/probability/overall"]["bootstrap"][
            "lower"
        ] = -0.005
        self.assertTrue(behavior_preservation(case, competence))

    def test_all_scales_steps_and_rows_are_required(self):
        physical = {
            "minimum_efficacy": 0.8,
            "maximum_efficacy": 1.2,
            "minimum_activity_rate": 1,
            "minimum_input_rate": 0,
            "bound_engagements": 0,
            "maximum_pair_sum_error": 0,
        }
        case = {
            "physical": {"example": physical},
            "trajectory_errors": {"example": 0.001},
            "margin_errors": {"example": 0.001},
            "paired_differences": {},
            "endpoints": {
                f"generic/exact_decision/{group}": {"mean": 0.9}
                for group in ("learned", "nonlearned")
            },
        }
        flags = {str(i): {"qualitative": True, "calibration": False} for i in range(9)}
        fit = {
            "cases": {
                f"{scale}/{steps}": copy.deepcopy(case)
                for scale in ("fast", "primary", "slow")
                for steps in (4096, 8192)
            },
            "parent_bridge": {"example": 0},
            "refinement": {"primary": {"example": {"margin": 0}}},
            "reference_checks": {
                "affine_max_error": 0,
                "query_no_write": True,
                "query_errors": {"example": 0},
            },
            "behavior": {"flags": copy.deepcopy(flags)},
            "parent_behavior": {"flags": flags},
            "control_no_write": {"teacher_off": True, "mismatch_clamp": True},
        }
        competence = {"generic_learned": 0.8, "generic_nonlearned": 0.7}
        self.assertEqual(
            decide_fit(fit, competence)["outcome"], "conditional_circuit_sufficiency"
        )
        fit["cases"]["slow/8192"]["margin_errors"]["example"] = 0.2
        self.assertFalse(decide_fit(fit, competence)["checks"]["correspondence"])
        fit["behavior"]["flags"] = {}
        self.assertFalse(decide_fit(fit, competence)["checks"]["behavior_preservation"])


if __name__ == "__main__":
    unittest.main()
