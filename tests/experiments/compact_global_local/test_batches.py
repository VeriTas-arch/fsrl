from __future__ import annotations

import unittest

import numpy as np

from fsrl.experiments.compact_global_local.batches import compact_input_arrays
from fsrl.experiments.compact_global_local.model import CompactModelConfig


class CompactInputTests(unittest.TestCase):
    def test_only_cues_evidence_and_response_pulse_exist(self):
        codes = np.asarray([[[1.0, -1.0, 1.0], [-1.0, 1.0, -1.0]]], dtype=np.float32)
        pairs = np.asarray([[[0, 1]]], dtype=np.int64)
        inputs = compact_input_arrays(codes, pairs, np.asarray([[0.25]]), steps=2)
        config = CompactModelConfig(cue_size=3)

        self.assertEqual(inputs.shape, (1, 2, 1, 8))
        np.testing.assert_array_equal(inputs[0, 0, 0, :6], codes[0].reshape(-1))
        self.assertEqual(inputs[0, 0, 0, config.evidence_index], 0.25)
        self.assertEqual(inputs[0, 1, 0, config.response_index], 1.0)
        self.assertEqual(np.count_nonzero(inputs[0, 1]), 1)

    def test_third_support_step_is_externally_blank(self):
        codes = np.asarray([[[1.0, -1.0, 1.0], [-1.0, 1.0, -1.0]]], dtype=np.float32)
        pairs = np.asarray([[[0, 1]]], dtype=np.int64)
        inputs = compact_input_arrays(codes, pairs, np.asarray([[0.25]]), steps=3)
        self.assertEqual(inputs.shape, (1, 3, 1, 8))
        self.assertEqual(np.count_nonzero(inputs[0, 2]), 0)


if __name__ == "__main__":
    unittest.main()
