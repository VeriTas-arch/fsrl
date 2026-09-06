import unittest

import numpy as np
from scipy.special import ndtri

from fsrl.experiments.cohort_diagnostic.statistics import reference_intervals
from fsrl.experiments.encoding_distribution.encoding import gaussian, impulse_moments
from fsrl.experiments.encoding_distribution.execution import moment_check
from fsrl.experiments.encoding_distribution.manipulations import summarize_change
from fsrl.experiments.encoding_distribution.statistics import (
    equivalence,
    failure_contrast,
    improvement,
    outside,
)
from fsrl.experiments.minimal_learner.data import ModelBatch
from fsrl.experiments.quantized_learner.encoding import (
    canonical_addresses,
    rounding_parameters,
)
from fsrl.experiments.structural_identification.model import reference
from fsrl.experiments.structural_identification.protocol import CODEBOOK


def fixture():
    codes = np.array([[1, 1, 1], [-1, 1, -1], [1, -1, -1]], dtype=float)
    support = np.array([[0, 1], [1, 2], [0, 1], [1, 2]])
    queries = np.array([[0, 1], [1, 2], [0, 2], [1, 0]])
    return ModelBatch(
        {
            "support_cues": np.concatenate(
                (codes[support[:, 0]], codes[support[:, 1]]), axis=-1
            )[:, None],
            "query_cues": np.concatenate(
                (codes[queries[:, 0]], codes[queries[:, 1]]), axis=-1
            )[None],
            "signed": np.array([0.2, 0.5, 0.2, 0.5])[:, None],
            "retention": np.ones((4, 1)),
            "local_evidence": np.zeros((4, 1)),
        }
    )


def reference_fixture():
    interval = {"lower": 0.2, "upper": 0.8}
    keys = (
        "learned_accuracy",
        "nonlearned_accuracy",
        "symbolic_distance_slope",
        "stable_error_80_analysis_proportion",
        "self_consistent_incorrect_proportion",
        "self_inconsistent_proportion",
        "correct_ranker_proportion",
    )
    return {
        "intervals": dict.fromkeys(keys, interval),
        "serial": interval,
        "tau": interval,
    }


class DistributionContracts(unittest.TestCase):
    def test_directional_failures_keep_pairing_and_undefined_cohorts(self):
        refs = reference_fixture()
        values = dict.fromkeys(reference_intervals(refs), 0.5)
        baseline = {
            "values": values,
            "flags": {"ranking": {"qualitative": True, "calibration": True}},
        }
        changed = [
            {**baseline, "values": {**values, "correct_ranker": p}} for p in (0.1, 0.9)
        ]
        result = failure_contrast(changed, [baseline, baseline], refs, 1)
        self.assertEqual(
            result["directional_failure"]["correct_ranker"]["below"]["mean"], 0.5
        )
        self.assertEqual(
            result["directional_failure"]["correct_ranker"]["above"]["mean"], 0.5
        )
        changed[0]["values"]["correct_ranker"] = None
        result = failure_contrast(changed, [baseline, baseline], refs, 1)
        self.assertIsNone(
            result["directional_failure"]["correct_ranker"]["below"]["mean"]
        )

    def test_tradeoff_requires_both_category_failures_and_no_new_mismatch(self):
        profile = {"estimates": {"nonlearned_accuracy": {"lower": 0.4, "upper": 0.6}}}
        contrast = {"nonlearned_accuracy": {"lower": -0.001, "upper": 0.001}}
        failures = {
            "directional_failure": {
                "correct_ranker": {"below": {"upper": -0.1}},
                "self_consistent_incorrect": {"above": {"upper": -0.1}},
            }
        }
        self.assertTrue(
            improvement(profile, contrast, failures, [], reference_fixture())
        )
        self.assertFalse(
            improvement(
                profile,
                contrast,
                failures,
                ["serial_position_effect"],
                reference_fixture(),
            )
        )
        failures["directional_failure"]["self_consistent_incorrect"]["above"][
            "upper"
        ] = 0.1
        self.assertFalse(
            improvement(profile, contrast, failures, [], reference_fixture())
        )

    def test_independent_dense_impulses_match_all_conditional_moments(self):
        base = fixture()
        encoded, _ = gaussian(base, np.array([0.1, 0.2, 0.7, 0.8])[:, None])
        for eta in (0.2, 0.98):
            errors = moment_check(base, encoded, eta, 6)
            self.assertLess(max(errors.values()), 1e-12)
        _, _, covariance, gamma = impulse_moments(base, 0.98, 6)
        self.assertLess(np.linalg.matrix_rank(covariance[0]), 3)
        np.testing.assert_allclose(gamma[0, :, 3], -gamma[0, :, 0], atol=1e-12)
        self.assertGreater(gamma[0, 0, 0], 0)

    def test_orientation_reversal_preserves_realized_state(self):
        base = fixture()
        uniforms = np.array([0.1, 0.2, 0.7, 0.8])[:, None]
        original, _ = gaussian(base, uniforms)
        reversed_batch = ModelBatch(
            {
                **base.arrays,
                "support_cues": np.roll(base.arrays["support_cues"], 3, axis=-1),
                "signed": -base.arrays["signed"],
            }
        )
        reversed_encoded, _ = gaussian(reversed_batch, uniforms)
        np.testing.assert_array_equal(
            original.arrays["signed"], -reversed_encoded.arrays["signed"]
        )
        np.testing.assert_allclose(
            reference(original, 0.8, 6, "decay")[1],
            reference(reversed_encoded, 0.8, 6, "decay")[1],
            atol=1e-12,
        )

    def test_zero_admission_and_codepoint_variance_remain_exact(self):
        base = fixture()
        uniforms = np.full((4, 1), 0.2)
        zero = ModelBatch({**base.arrays, "retention": np.zeros((4, 1))})
        encoded, tails = gaussian(zero, uniforms)
        np.testing.assert_array_equal(reference(encoded, 0.8, 6, "decay")[1], 0)
        self.assertEqual(
            tails, {"admitted": 0, "outside_adjacent": 0, "outside_display": 0}
        )
        moment_check(zero, encoded, 0.8, 6)
        exact = ModelBatch(
            {**base.arrays, "signed": np.array([1, 1 / 3, 1, 1 / 3])[:, None]}
        )
        encoded, _ = gaussian(exact, uniforms)
        np.testing.assert_array_equal(encoded.arrays["signed"], exact.arrays["signed"])

    def test_no_clipping_or_empirical_recentering(self):
        base = fixture()
        uniforms = np.full((4, 1), 1e-12)
        encoded, tails = gaussian(base, uniforms)
        _, orientation = canonical_addresses(base.arrays["support_cues"])
        canonical = orientation * base.arrays["signed"]
        _, _, variance = rounding_parameters(canonical, np.asarray(CODEBOOK))
        noise = (orientation * encoded.arrays["signed"] - canonical) / np.sqrt(variance)
        np.testing.assert_allclose(noise, -ndtri(uniforms), atol=1e-12)
        self.assertEqual(tails["outside_display"], 4)
        self.assertGreater(np.abs(encoded.arrays["signed"]).max(), 1)
        with self.assertRaisesRegex(ValueError, "open-interval"):
            gaussian(base, np.zeros((4, 1)))

    def test_equivalence_is_not_nonsignificance_and_boundaries_do_not_pass(self):
        self.assertEqual(
            equivalence({"lower": -0.02, "upper": 0.02}, 0.01), "unresolved"
        )
        self.assertEqual(
            equivalence({"lower": -0.01, "upper": 0.001}, 0.01), "unresolved"
        )
        self.assertEqual(
            equivalence({"lower": 0.002, "upper": 0.004}, 0.005), "equivalent"
        )
        self.assertEqual(
            equivalence({"lower": 0.006, "upper": 0.01}, 0.005),
            "meaningfully_different",
        )
        self.assertEqual(
            equivalence({"lower": None, "upper": None}, 0.005), "unresolved"
        )
        self.assertIsNone(outside(None, {"lower": 0, "upper": 1}, "below"))

    def test_K_transport_preserves_whole_instance_pairing(self):
        arrays = {}
        mask = np.zeros((8, 28), bool)
        mask[:, :16] = True
        bridge = np.zeros_like(mask)
        bridge[:, 0] = True
        for structure in ("M10", "M11", "G"):
            for k in (4, 8):
                prefix = f"{structure}__balanced_K{k}__"
                arrays[prefix + "cross"] = mask
                arrays[prefix + "bridge"] = bridge
                arrays[prefix + "sampled_correct"] = np.broadcast_to(
                    np.arange(8)[:, None] / 16, mask.shape
                ) + (0.25 if k == 8 else 0)
        result = summarize_change(arrays, 852)
        self.assertEqual(
            result["cross"]["G_minus_Q_change"], {"mean": 0, "lower": 0, "upper": 0}
        )
        self.assertEqual(result["cross"]["status"], "equivalent")


if __name__ == "__main__":
    unittest.main()
