import unittest

import numpy as np

from fsrl.experiments.assembly.diagnostics import (
    build_field_design,
    choice_fields_from_pair_accuracy,
    directional_diagnosis,
    load_human_choice_fields,
    metric_arrays,
)
from fsrl.infrastructure.study_registry import resolve_record
from fsrl.tasks.registered_protocol import load_ranking_protocol


class AssemblyDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.protocol = load_ranking_protocol(resolve_record("benchmarks/liu_v2.json"))
        self.design = build_field_design(self.protocol)

    def test_pure_item_potential_has_no_hodge_residual(self):
        potential = np.linspace(-1.0, 1.0, self.protocol.n_items)
        field = np.asarray(
            [
                potential[first] - potential[second]
                for first, second in self.design.pairs
            ]
        )
        metrics = metric_arrays(field, self.design)
        self.assertAlmostEqual(metrics["gradient_energy_fraction"][0], 1.0)
        self.assertAlmostEqual(metrics["mean_absolute_residual"][0], 0.0)
        self.assertAlmostEqual(
            metrics["total_accuracy_slope"][0],
            metrics["gradient_accuracy_slope"][0],
        )
        self.assertAlmostEqual(metrics["residual_accuracy_slope"][0], 0.0)

    def test_cycle_field_has_nonzero_conjunctive_residual(self):
        field = np.zeros(len(self.design.pairs), dtype=np.float64)
        pair_to_index = {pair: index for index, pair in enumerate(self.design.pairs)}
        field[pair_to_index[(0, 1)]] = 1.0
        field[pair_to_index[(1, 2)]] = 1.0
        field[pair_to_index[(0, 2)]] = -1.0
        metrics = metric_arrays(field, self.design)
        self.assertLess(metrics["gradient_energy_fraction"][0], 1.0)
        self.assertGreater(metrics["mean_absolute_residual"][0], 0.0)
        self.assertAlmostEqual(
            metrics["total_accuracy_slope"][0],
            metrics["gradient_accuracy_slope"][0]
            + metrics["residual_accuracy_slope"][0],
        )

    def test_pair_accuracy_conversion_respects_source_correct_order(self):
        accuracy = np.ones((2, len(self.design.pairs)), dtype=np.float64)
        fields = choice_fields_from_pair_accuracy(accuracy, self.design)
        np.testing.assert_allclose(fields * self.design.true_sign, 1.0)
        metrics = metric_arrays(np.mean(fields, axis=0), self.design)
        self.assertAlmostEqual(metrics["total_accuracy_slope"][0], 0.0)

    def test_human_field_reproduces_registered_symbolic_distance_slope(self):
        fields = load_human_choice_fields(self.protocol, self.design)
        metrics = metric_arrays(np.mean(fields, axis=0), self.design)
        self.assertEqual(fields.shape, (77, 28))
        self.assertAlmostEqual(metrics["total_accuracy_slope"][0], 0.03982860676738232)

    def test_directional_diagnosis_uses_registered_interval_directions(self):
        def comparison(values):
            return {
                "bootstrap": {
                    name: {"lower": lower, "upper": upper}
                    for name, (lower, upper) in values.items()
                }
            }

        slope_positive = comparison({"total_accuracy_slope": (0.001, 0.01)})
        neural_minus_human = comparison({"gradient_energy_fraction": (0.01, 0.20)})
        human_minus_neural = comparison(
            {
                "learned_correctness_aligned_residual_accuracy_effect_adjusted_for_symbolic_distance": (
                    0.001,
                    0.02,
                )
            }
        )
        result = directional_diagnosis(
            slope_positive,
            slope_positive,
            neural_minus_human,
            human_minus_neural,
        )
        self.assertTrue(result["evidence_model_contribution"])
        self.assertTrue(result["neural_over_sharpening"])
        self.assertTrue(result["human_mixed_code_signal"])


if __name__ == "__main__":
    unittest.main()
