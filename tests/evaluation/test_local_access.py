import unittest

import numpy as np
import torch

from fsrl.core.config import TrainConfig
from fsrl.core.local_trace import ConjunctiveLocalTrace
from fsrl.core.plastic_rnn import RetroModulRNN
from fsrl.evaluation.frozen_fast_weight import (
    FastWeightIntervention,
    FrozenFastWeightEvaluator,
)
from fsrl.evaluation.local_access import (
    access_factor,
    readout_dual_access_query_conditions,
)
from fsrl.evaluation.relational_query import readout_relational_query_bundle
from fsrl.tasks.protocol import ordered_pairs
from fsrl.tasks.protocol_catalog import load_registered_protocol


class LocalAccessTests(unittest.TestCase):
    def test_access_factor_preserves_retained_and_weakens_omitted(self):
        admission = np.asarray([1.0, 0.0, 0.0])
        reliability = np.asarray([0.2, 0.3, 0.8])
        observed = access_factor(admission, reliability)
        np.testing.assert_array_equal(observed, np.asarray([1.0, 0.3, 0.8]))

    def test_access_factor_rejects_nonbinary_global_admission(self):
        with self.assertRaisesRegex(ValueError, "binary"):
            access_factor(np.asarray([0.5]), np.asarray([0.5]))

    def test_dual_access_readout_preserves_direct_condition_semantics(self):
        torch.manual_seed(41)
        protocol = load_registered_protocol("liu_v1")
        config = TrainConfig(bs=2, hs=4, cs=8, nbcues_min=8, nbcues_max=8)
        net = RetroModulRNN(config.to_model_dict(), device="cpu")
        evaluator = FrozenFastWeightEvaluator(
            net,
            config,
            protocol,
            cue_seed=5,
            support_seed=7,
            subject_encoding_mode="stable_omission",
            subject_encoding_seed=11,
        )
        local = ConjunctiveLocalTrace(config.cs, device="cpu")
        schedules = tuple(ordered_pairs(protocol.n_items) for _ in range(config.bs))

        readout = readout_dual_access_query_conditions(evaluator, local, schedules)
        intact_fast_weights = evaluator.learn_fast_weights(
            FastWeightIntervention.INTACT
        )
        expected = readout_relational_query_bundle(
            evaluator,
            local,
            intact_fast_weights,
            readout["intact_trace"].state,
            schedules,
            local_off=False,
            global_off=False,
            shuffled_indices=None,
        )
        observed = readout["condition_bundles"]["intact"]
        self.assertEqual(set(observed), set(expected))
        for name in expected:
            np.testing.assert_array_equal(observed[name], expected[name])
        self.assertEqual(
            len(readout["global_loo_bundles"]),
            len(protocol.support_pairs_higher_lower),
        )
        self.assertEqual(
            len(readout["local_loo_bundles"]),
            len(protocol.support_pairs_higher_lower),
        )
