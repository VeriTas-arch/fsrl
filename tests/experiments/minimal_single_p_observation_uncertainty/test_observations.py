from __future__ import annotations

import unittest

import numpy as np

from fsrl.experiments.minimal_single_p_observation_uncertainty.observations import (
    encode,
    epsilon_for,
    observation_checks,
)
from fsrl.experiments.training_strategy.batches import EpisodeBatch


class ObservationTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(11)
        support = np.zeros((4, 4, 3, 38), dtype=np.float32)
        signed = (
            np.asarray(
                [[-2, -1, 1], [-1, 1, 2], [1, 2, -2], [2, -2, -1]],
                dtype=np.float64,
            )
            / 7
        )
        retention = np.ones((4, 3), dtype=np.float64)
        support[:, 0, :, 37] = signed
        support[:, 0, :, 34] = signed
        self.cpu = EpisodeBatch(
            {
                "support_inputs": support,
                "local_evidence": signed.astype(np.float32),
                "query_inputs": rng.standard_normal((2, 6, 38)).astype(np.float32),
                "targets": np.zeros(6, dtype=np.int64),
                "support_pairs": np.zeros((4, 3, 2), dtype=np.int64),
                "query_pairs": np.zeros((2, 2), dtype=np.int64),
                "retention": np.ones((3, 8), dtype=bool),
                "signed_magnitudes": signed,
                "trial_retention": retention,
                "probabilities": np.ones((4, 3), dtype=np.float64),
                "item_codes": np.zeros((3, 8, 15), dtype=np.float32),
            }
        )

    def test_clean_and_zero_are_exact(self):
        epsilon = epsilon_for(self.cpu, 12)
        self.assertEqual(
            encode(self.cpu, "clean", 1 / 7, epsilon).fingerprint(),
            self.cpu.fingerprint(),
        )
        self.assertEqual(
            encode(self.cpu, "noisy", 0, epsilon).fingerprint(),
            self.cpu.fingerprint(),
        )

    def test_folded_matches_amplitude_and_restores_sign(self):
        epsilon = epsilon_for(self.cpu, 13)
        clean = encode(self.cpu, "clean", 1 / 7, epsilon)
        folded = encode(self.cpu, "folded", 1 / 7, epsilon)
        noisy = encode(self.cpu, "noisy", 1 / 7, epsilon)
        checks = observation_checks(clean, folded, noisy)
        self.assertTrue(checks["absolute_amplitudes_equal"])
        self.assertTrue(checks["folded_signs_restored"])
        np.testing.assert_array_equal(
            noisy.arrays["query_inputs"], self.cpu.arrays["query_inputs"]
        )

    def test_unknown_condition_fails(self):
        with self.assertRaises(ValueError):
            encode(self.cpu, "other", 1 / 7, epsilon_for(self.cpu, 14))


if __name__ == "__main__":
    unittest.main()
