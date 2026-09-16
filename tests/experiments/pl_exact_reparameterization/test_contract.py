import unittest

import numpy as np
import torch

from fsrl.core.factorized_plastic_rnn import FactorizedPlasticRNN
from fsrl.core.model_config import RetroModelConfig
from fsrl.core.plastic_rnn import RetroModulRNN
from fsrl.experiments.pl_exact_reparameterization.protocol import load_specification
from fsrl.experiments.pl_exact_reparameterization.qualification import (
    _run_panel,
    active_input_sequence,
)
from fsrl.tasks.protocol_catalog import load_registered_protocol


class ExactReparameterizationContractTests(unittest.TestCase):
    def test_random_model_panels_match_at_both_registered_horizons(self):
        torch.manual_seed(1702)
        legacy = RetroModulRNN(
            RetroModelConfig(
                input_size=37,
                hidden_size=7,
                output_size=2,
                batch_size=4,
            ),
            device="cpu",
        )
        factorized = FactorizedPlasticRNN.from_legacy(legacy)
        protocol = load_registered_protocol("liu_v2")
        for panel_kind, trials in (("liu", 32), ("generic", 40)):
            with self.subTest(panel_kind=panel_kind):
                result = _run_panel(
                    legacy,
                    factorized,
                    -1.8,
                    protocol,
                    device=torch.device("cpu"),
                    seed=2104,
                    panel_kind=panel_kind,
                )
                self.assertEqual(result["support_trials"], trials)
                self.assertEqual(result["support_microsteps"], trials * 4)
                self.assertEqual(result["effective_P_writes"], trials * 2)
                self.assertEqual(result["query_unordered_pairs"], 28)
                self.assertEqual(result["query_orientations"], 56)
                self.assertLessEqual(max(result["max_absolute_errors"].values()), 2e-6)
                self.assertEqual(sum(result["categorical_mismatches"].values()), 0)

    def test_timestep_accounting_preserves_parent_support_horizon(self):
        accounting = load_specification()["complete_timestep_accounting"]
        generic = accounting["generic_training_episode"]
        self.assertEqual(generic["support_microsteps_per_trial"], 4)
        self.assertEqual(generic["support_microsteps"], "112--160")
        self.assertEqual(generic["query_microsteps"], 56)
        self.assertEqual(generic["total_active_microsteps"], "168--216")
        self.assertEqual(generic["effective_P_writes"], "56--80")
        self.assertEqual(
            accounting["liu_v2_literal_episode"]["total_active_microsteps"], 688
        )

    def test_active_inputs_preserve_only_the_declared_legacy_signals(self):
        codes = np.ones((2, 8, 15), dtype=np.float32)
        pairs = np.asarray([[0, 1], [2, 3]], dtype=np.int64)
        evidence = np.asarray([0.25, -0.5], dtype=np.float32)
        inputs = active_input_sequence(
            codes, pairs, evidence, steps=4, time_value=0.4
        ).numpy()
        np.testing.assert_array_equal(inputs[:, :, 31], 1.0)
        np.testing.assert_array_equal(inputs[:, :, 32], np.float32(0.4))
        np.testing.assert_array_equal(inputs[:, :, 33], 0.0)
        np.testing.assert_array_equal(inputs[:, :, 35:], 0.0)
        np.testing.assert_array_equal(inputs[0, :, 34], evidence)
        np.testing.assert_array_equal(inputs[1, :, 30], 1.0)
        np.testing.assert_array_equal(inputs[2:, :, 30], 0.0)


if __name__ == "__main__":
    unittest.main()
