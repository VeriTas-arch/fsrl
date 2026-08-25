import unittest
from types import SimpleNamespace

import numpy as np

from fsrl.assembly_diagnostics import load_json
from fsrl.history_state_factorial import (
    first_exposure_target_indices,
    scalar_factorial,
    validate_registered_sources,
    vector_factorial,
)
from fsrl.study_registry import resolve_record


class HistoryStateFactorialTests(unittest.TestCase):
    def test_scalar_matched_history_contrast_is_half_interaction(self):
        cells = {
            "NN": np.asarray([1.0, 2.0]),
            "NH": np.asarray([4.0, 1.0]),
            "HN": np.asarray([3.0, 5.0]),
            "HH": np.asarray([9.0, 8.0]),
        }
        effects = scalar_factorial(cells)
        np.testing.assert_allclose(
            effects["matched_history_contrast"], 0.5 * effects["interaction"]
        )

    def test_vector_factorial_uses_cell_vectors_before_norm(self):
        cells = {
            "NN": np.asarray([[[1.0, 0.0]]]),
            "NH": np.asarray([[[3.0, 2.0]]]),
            "HN": np.asarray([[[2.0, 4.0]]]),
            "HH": np.asarray([[[7.0, 10.0]]]),
        }
        effects = vector_factorial(cells)
        np.testing.assert_allclose(
            effects["factor_vector_norm"], [[np.linalg.norm([3.5, 4.0])]]
        )
        np.testing.assert_allclose(
            effects["baseline_vector_norm"], [[np.linalg.norm([2.5, 6.0])]]
        )
        np.testing.assert_allclose(
            effects["interaction_vector_norm"], [[np.linalg.norm([3.0, 4.0])]]
        )

    def test_fourth_exposure_targets_same_relation_at_first_exposure(self):
        relations = np.asarray(
            [
                [[0, 1], [2, 3]],
                [[4, 5], [6, 7]],
                [[0, 1], [6, 7]],
                [[4, 5], [2, 3]],
            ]
        )
        factors = SimpleNamespace(
            exposure=np.asarray([[1, 1], [1, 1], [4, 4], [4, 4]]),
            relations=relations,
            retained=np.ones((4, 2), dtype=bool),
        )
        np.testing.assert_array_equal(
            first_exposure_target_indices(factors), np.asarray([[0, 1], [1, 0]])
        )

    def test_registered_sources_are_immutable(self):
        specification = load_json(resolve_record("benchmarks/history_state_factorial_v1.json"))
        validation = validate_registered_sources(specification)
        self.assertEqual(len(validation["pilot_artifacts"]), 2)

    def test_result_closes_pilots_without_formal_seed_access(self):
        result = load_json(resolve_record("results/history_state_factorial_v1.json"))
        self.assertEqual(set(result["pilot_seeds"]), {"1901", "1902"})
        self.assertTrue(result["overall_diagnosis"]["pilot_stop_rule_met"])
        self.assertEqual(
            result["overall_diagnosis"]["formal_confirmation_status"],
            "deferred; formal seeds remain untouched",
        )


if __name__ == "__main__":
    unittest.main()
