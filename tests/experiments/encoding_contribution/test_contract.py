import unittest

import numpy as np

from fsrl.experiments.encoding_contribution.execution import internal_record
from fsrl.experiments.encoding_contribution.moments import (
    conditional_moments,
    enumerate_moments,
)
from fsrl.experiments.encoding_contribution.statistics import (
    paired_change,
    signed_counts,
)
from fsrl.experiments.minimal_learner.data import ModelBatch, generic_batch, pair_cues
from fsrl.experiments.minimal_learner.protocol import task_generator
from fsrl.experiments.structural_identification.model import reference
from fsrl.experiments.training_strategy.batches import sample_episodes
from fsrl.tasks.protocol import ordered_pairs
from fsrl.tasks.protocol_catalog import load_registered_protocol


class EncodingContributionContracts(unittest.TestCase):
    def fixture(self):
        rng = np.random.default_rng(841)
        batch = generic_batch(sample_episodes(task_generator(), rng, 1))
        return ModelBatch(
            {
                key: value if key == "query_cues" else value[:4]
                for key, value in batch.arrays.items()
                if key in ("query_cues", "support_cues", "signed", "retention")
            }
        )

    def test_exact_moments_enumeration_zero_admission_and_reversal(self):
        fixture = self.fixture()
        width = fixture.arrays["support_cues"].shape[-1] // 2
        reverse = ModelBatch(
            {
                **fixture.arrays,
                "support_cues": np.roll(fixture.arrays["support_cues"], width, axis=-1),
                "signed": -fixture.arrays["signed"],
            }
        )
        zero = ModelBatch(
            {**fixture.arrays, "retention": np.zeros_like(fixture.arrays["retention"])}
        )
        moments = []
        for batch in (fixture, reverse, zero):
            mean, covariance, query_variance = conditional_moments(batch, 0.7, 6)
            exact_mean, exact_cov = enumerate_moments(batch, 0.7, 6)
            np.testing.assert_allclose(mean, exact_mean, atol=1e-12, rtol=1e-10)
            np.testing.assert_allclose(covariance, exact_cov, atol=1e-12, rtol=1e-10)
            np.testing.assert_allclose(
                mean, reference(batch, 0.7, 6, "decay")[1], atol=1e-12, rtol=1e-10
            )
            self.assertGreaterEqual(np.linalg.eigvalsh(covariance).min(), -1e-12)
            self.assertTrue(np.all(query_variance >= 0))
            moments.append((mean, covariance))
        for index in (0, 1):
            np.testing.assert_allclose(moments[0][index], moments[1][index], atol=1e-12)
            np.testing.assert_array_equal(moments[2][index], 0)
        self.assertGreater(np.trace(moments[0][1][0]), 0)

    def test_paired_change_cancels_instance_variation(self):
        x = np.arange(256, dtype=float)
        values, estimate = paired_change(x + 2, x, -x + 1, -x, 842)
        np.testing.assert_array_equal(values, 1)
        self.assertEqual(estimate, {"mean": 1, "lower": 1, "upper": 1})

    def test_signed_counts_preserve_undefined_and_closed_boundaries(self):
        self.assertEqual(
            signed_counts([None, np.nan, -1, 0, 1, 2], {"lower": 0, "upper": 1}),
            {"below": 1, "inside": 2, "above": 1, "undefined": 2},
        )

    def test_internal_truth_uses_registered_order_without_query_labels(self):
        protocol = load_registered_protocol("liu_v2")
        scores = 8 - np.argsort(protocol.true_order_high_to_low)
        codes = np.eye(8)[None]
        pairs = np.asarray(ordered_pairs(8))[None]
        batch = ModelBatch(
            {
                "query_pairs": pairs,
                "codes": codes,
                "query_cues": pair_cues(codes, pairs),
            }
        )
        margins = scores[pairs[..., 0]] - scores[pairs[..., 1]]
        result = internal_record(margins, scores[None], batch, protocol)
        self.assertEqual(
            result, {"nonlearned_sign_accuracy": 1, "latent_correct_order": 1}
        )


if __name__ == "__main__":
    unittest.main()
