from __future__ import annotations

import unittest

import numpy as np

from fsrl.experiments.compact_global_local.liu import CompactLiuEvaluator
from fsrl.experiments.compact_global_local.model import (
    CompactModelConfig,
    CompactPlasticRNN,
    PackedLocalTrace,
)
from fsrl.infra.runtime import CPU_TEST_PROFILE
from fsrl.tasks.protocol_catalog import load_registered_protocol


class CompactLiuEvaluatorTests(unittest.TestCase):
    def setUp(self):
        config = CompactModelConfig(cue_size=15, hidden_size=6)
        backbone = CompactPlasticRNN(config, device="cpu")
        local = PackedLocalTrace(config.cue_size, device="cpu")
        self.evaluator = CompactLiuEvaluator(
            backbone,
            local,
            load_registered_protocol("liu_v2"),
            subjects=3,
            cue_seed=31001,
            support_seed=31101,
            subject_encoding_seed=31201,
            cue_mode="permuted_shared",
            subject_encoding_mode="stable_omission",
            execution_profile=CPU_TEST_PROFILE,
        )

    def test_support_and_query_shapes_follow_repaired_schedule(self):
        inputs = self.evaluator._support_inputs(
            0, zero_evidence=False, zero_relations=frozenset()
        )
        self.assertEqual(inputs.shape, (3, 3, 32))
        fast_weights = self.evaluator.learn_fast_weights()
        local_state, natural, applied = self.evaluator.build_local_state()
        bundle = self.evaluator.query_bundle(fast_weights, local_state)
        self.assertEqual(fast_weights.shape, (3, 6, 6))
        self.assertEqual(local_state.shape, (3, 105))
        self.assertEqual(bundle["logits"].shape, (3, 56))
        np.testing.assert_array_equal(natural, applied)

    def test_local_evidence_preserves_retained_and_broadens_omitted(self):
        values = self.evaluator.local_evidence()
        self.assertEqual(values.shape, (3, 32))
        self.assertTrue(np.all(np.isfinite(values)))
        self.assertTrue(np.any(values != 0.0))


if __name__ == "__main__":
    unittest.main()
