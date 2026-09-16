import io
import unittest

import numpy as np

from fsrl.experiments.pl_crosstalk_decomposition.estimands import (
    keyed_numeric_max_error,
    packed_keys,
    probability_components,
    relation_source_contributions,
    retained_subject_mean,
    source_concentration,
)
from fsrl.experiments.pl_crosstalk_decomposition.storage import (
    deterministic_npz_bytes,
)


class CrossTalkEstimandTests(unittest.TestCase):
    def test_nested_summary_comparison_is_keyed(self):
        observed = {"bootstrap": {"lower": -0.1}, "subjects": 77}
        expected = {"subjects": 77, "bootstrap": {"lower": -0.1}}
        self.assertEqual(keyed_numeric_max_error(observed, expected), 0.0)
        expected["bootstrap"]["lower"] = -0.2
        self.assertAlmostEqual(keyed_numeric_max_error(observed, expected), 0.1)

    def test_packed_key_is_normalized_and_antisymmetric(self):
        left = np.asarray([[1.0, -1.0, 1.0]], dtype=np.float32)
        right = np.asarray([[-1.0, 1.0, 1.0]], dtype=np.float32)
        forward = packed_keys(left, right)
        reverse = packed_keys(right, left)
        np.testing.assert_allclose(np.linalg.norm(forward, axis=1), 1.0)
        np.testing.assert_array_equal(reverse, -forward)

    def test_source_contributions_sum_to_trial_level_cross_talk(self):
        support = np.asarray([[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]], dtype=np.float32)
        query = np.asarray([[[1.0, 0.0], [0.5, 0.5]]], dtype=np.float32)
        evidence = np.asarray([[0.4, -0.2, 0.1]], dtype=np.float32)
        indices = np.asarray([[0, 1, 0]], dtype=np.int64)
        observed = relation_source_contributions(
            evidence, support, indices, query, relation_count=2
        )
        expected = np.asarray([[[0.5, 0.25], [0.0, -0.1]]])
        np.testing.assert_allclose(observed, expected)
        np.testing.assert_allclose(
            observed.sum(axis=1),
            np.einsum("it,itq->iq", evidence, support @ query[0].T),
        )

    def test_probability_effect_uses_total_shared_operating_point(self):
        baseline = np.asarray([[[0.2, -0.1]]])
        cross_talk = np.asarray([[[0.3, -0.2]]])
        signs = np.asarray([[1.0, -1.0]])
        result = probability_components(
            baseline, cross_talk, signs, gain=0.25, temperature=0.5
        )
        perturbation = 0.25 * cross_talk * signs[None]
        sigmoid = lambda value: 1.0 / (1.0 + np.exp(-value))
        expected = sigmoid((baseline + perturbation) / 0.5) - sigmoid(baseline / 0.5)
        np.testing.assert_allclose(result["exact_effect"], expected)
        np.testing.assert_allclose(result["correct_perturbation"], perturbation)

    def test_retained_weighting_and_concentration(self):
        values = np.asarray(
            [
                [[1.0, 3.0], [100.0, 100.0]],
                [[2.0, 4.0], [6.0, 8.0]],
            ]
        )
        retention = np.asarray([[True, False], [True, True]])
        np.testing.assert_allclose(retained_subject_mean(values, retention), [2.0, 5.0])

        sources = np.asarray([[[[3.0, 1.0], [-2.0, 2.0]]]])
        concentration = source_concentration(sources, np.asarray([[1.0, -1.0]]))
        np.testing.assert_allclose(concentration["top_one_share"], [[[0.75, 0.5]]])
        np.testing.assert_allclose(concentration["source_count_80pct"], [[[2.0, 2.0]]])

    def test_npz_serialization_is_typed_and_deterministic(self):
        arrays = {
            "z": np.asarray([True, False]),
            "a": np.asarray([1.0, 2.0], dtype=np.float64),
        }
        first = deterministic_npz_bytes(arrays)
        second = deterministic_npz_bytes(dict(reversed(list(arrays.items()))))
        self.assertEqual(first, second)
        with np.load(io.BytesIO(first), allow_pickle=False) as restored:
            np.testing.assert_array_equal(restored["z"], arrays["z"])
            np.testing.assert_array_equal(restored["a"], arrays["a"])


if __name__ == "__main__":
    unittest.main()
