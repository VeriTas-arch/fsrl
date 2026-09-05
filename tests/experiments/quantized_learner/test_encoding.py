import unittest

import numpy as np

from fsrl.experiments.minimal_learner.data import ModelBatch
from fsrl.experiments.quantized_learner.encoding import (
    canonical_addresses,
    draw_indices,
    encode_batch,
    rounding_parameters,
)
from fsrl.experiments.quantized_learner.protocol import specification
from fsrl.experiments.quantized_learner.reference import (
    enumerate_persistent,
    joint_choice_log_probability,
    rollout,
)


def fixture():
    codes = np.asarray([[-1, -1, 1], [-1, 1, -1], [1, -1, -1]], dtype=np.float64)
    pairs = np.asarray([[0, 1], [1, 2], [1, 0], [2, 1]])
    cues = np.concatenate((codes[pairs[:, 0]], codes[pairs[:, 1]]), axis=-1)[:, None]
    query_pairs = np.asarray([[0, 1], [0, 2], [1, 2], [1, 0]])
    query = np.concatenate(
        (codes[query_pairs[:, 0]], codes[query_pairs[:, 1]]), axis=-1
    )[None]
    return ModelBatch(
        {
            "support_cues": cues,
            "signed": np.asarray([[0.2], [0.55], [-0.2], [-0.55]]),
            "retention": np.ones((4, 1)),
            "local_evidence": np.zeros((4, 1)),
            "query_cues": query,
        }
    )


class EncodingTests(unittest.TestCase):
    def setUp(self):
        self.codebook = np.asarray(specification()["encoding"]["codebook"])

    def test_exact_branch_mean_variance_and_endpoints(self):
        values = np.unique(np.concatenate((np.linspace(-1, 1, 99), self.codebook)))
        lower, p, variance = rounding_parameters(values, self.codebook)
        low, high = self.codebook[lower], self.codebook[lower + 1]
        mean = (1 - p) * low + p * high
        observed_variance = (1 - p) * (low - mean) ** 2 + p * (high - mean) ** 2
        np.testing.assert_allclose(mean, values, atol=1e-15)
        np.testing.assert_allclose(observed_variance, variance, atol=1e-15)
        for value in self.codebook:
            draws = draw_indices(
                np.full(10, value), np.linspace(0, 0.99, 10), self.codebook
            )
            np.testing.assert_array_equal(self.codebook[draws], value)

    def test_sign_error_probabilities_are_codec_predictions(self):
        lower, p, _ = rounding_parameters(np.asarray([1 / 7, 2 / 7]), self.codebook)
        np.testing.assert_array_equal(lower, [1, 1])
        np.testing.assert_allclose(1 - p, [2 / 7, 1 / 14], atol=1e-15)

    def test_joint_choices_share_code_not_averaged_policy(self):
        margins = np.asarray([[20, 20], [-20, -20]])
        weights = np.asarray([0.5, 0.5])
        same = joint_choice_log_probability(
            margins, weights, np.asarray([1, 1]), np.asarray([0, 0]), temperature=1
        )
        opposed = joint_choice_log_probability(
            margins, weights, np.asarray([1, 0]), np.asarray([0, 1]), temperature=1
        )
        self.assertAlmostEqual(float(np.exp(same)), 0.5, places=7)
        self.assertLess(float(np.exp(opposed)), 1e-8)
        # Averaging both marginal policies first would give 0.25 to either event.
        self.assertNotAlmostEqual(float(np.exp(same)), 0.25)
        extreme = joint_choice_log_probability(
            margins * 1000,
            weights,
            np.asarray([10, 0]),
            np.asarray([0, 10]),
            temperature=0.25,
        )
        self.assertTrue(np.isfinite(extreme))

    def test_out_of_range_or_invalid_uniform_is_not_clipped(self):
        for value, uniform in ((1.001, 0.5), (np.nan, 0.5), (0.2, 1), (0.2, np.nan)):
            with (
                self.subTest(value=value, uniform=uniform),
                self.assertRaises(ValueError),
            ):
                draw_indices(np.asarray(value), np.asarray(uniform), self.codebook)

    def test_canonical_identity_uses_cues_and_reverses(self):
        batch = fixture()
        keys, signs = canonical_addresses(batch.arrays["support_cues"])
        np.testing.assert_array_equal(keys[:2], keys[2:])
        np.testing.assert_array_equal(signs[:2], -signs[2:])
        swapped = batch.arrays["support_cues"][..., [3, 4, 5, 0, 1, 2]]
        other_keys, other_signs = canonical_addresses(swapped)
        np.testing.assert_array_equal(keys, other_keys)
        np.testing.assert_array_equal(signs, -other_signs)

    def test_first_code_reused_and_fresh_code_not_reused(self):
        batch = fixture()
        uniforms = np.asarray([[0.0], [0.0], [0.99], [0.99]])
        retained, witness = encode_batch(batch, "persistent", uniforms, self.codebook)
        fresh, fresh_witness = encode_batch(batch, "resampled", uniforms, self.codebook)
        np.testing.assert_array_equal(
            witness["code_indices"][:2], witness["code_indices"][2:]
        )
        np.testing.assert_array_equal(
            retained.arrays["signed"][:2], -retained.arrays["signed"][2:]
        )
        self.assertTrue(
            np.all(
                fresh_witness["code_indices"][:2] != fresh_witness["code_indices"][2:]
            )
        )
        self.assertFalse(
            np.array_equal(retained.arrays["signed"], fresh.arrays["signed"])
        )
        np.testing.assert_array_equal(witness["cache_content_bits"], [4])
        np.testing.assert_array_equal(fresh_witness["cache_entries"], [0])
        reset, _ = encode_batch(
            batch, "persistent", 1 - uniforms - 0.001, self.codebook
        )
        self.assertFalse(
            np.array_equal(retained.arrays["signed"], reset.arrays["signed"])
        )

    def test_omission_has_no_cache_and_no_state_update(self):
        batch = fixture()
        batch.arrays["retention"][[0, 2]] = 0
        encoded, witness = encode_batch(
            batch, "persistent", np.full((4, 1), 0.5), self.codebook
        )
        np.testing.assert_array_equal(witness["code_indices"][[0, 2]], -1)
        np.testing.assert_array_equal(witness["cache_entries"], [1])
        result = rollout(encoded.arrays, eta=0.6, gain=2, epsilon=1e-8)
        np.testing.assert_array_equal(result["trajectory"][0], result["trajectory"][1])
        np.testing.assert_array_equal(result["trajectory"][2], result["trajectory"][3])

    def test_no_exact_metric_or_local_evidence_is_passed_to_readout(self):
        batch = fixture()
        batch.arrays["local_evidence"][:] = 999
        encoded, _ = encode_batch(
            batch, "persistent", np.full((4, 1), 0.1), self.codebook
        )
        self.assertTrue(np.isin(encoded.arrays["signed"], self.codebook).all())
        np.testing.assert_array_equal(encoded.arrays["local_evidence"], 0)
        self.assertEqual(len(encoded.tensors("cpu")), 5)

    def test_enumerated_mean_and_covariance_and_empty_admission(self):
        for admitted in (True, False):
            batch = fixture()
            if not admitted:
                batch.arrays["retention"][:] = 0
            result = enumerate_persistent(
                batch.arrays, 0, self.codebook, eta=0.6, gain=2, epsilon=1e-8
            )
            self.assertEqual(len(result["weights"]), 4 if admitted else 1)
            self.assertAlmostEqual(float(result["weights"].sum()), 1)
            np.testing.assert_allclose(
                result["mean"], result["exact_margin"], atol=1e-14
            )
            np.testing.assert_allclose(
                result["covariance"], result["analytic_covariance"], atol=1e-14
            )
            for codes, margin in zip(result["codes"], result["margins"], strict=True):
                inputs = {
                    **batch.arrays,
                    "signed": (result["projection"] @ codes)[:, None],
                }
                independent = rollout(inputs, eta=0.6, gain=2, epsilon=1e-8)
                np.testing.assert_allclose(
                    independent["margins"][0], margin, atol=1e-14
                )


if __name__ == "__main__":
    unittest.main()
