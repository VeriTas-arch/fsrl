import unittest

import numpy as np
import torch

from fsrl.analysis.hodge import build_complete_graph_geometry
from fsrl.core.config import TrainConfig
from fsrl.core.plastic_rnn import RetroModulRNN
from fsrl.evaluation.frozen_fast_weight import (
    FastWeightIntervention,
    FrozenFastWeightEvaluator,
)
from fsrl.experiments.assembly.trajectory import readout_margin_fields
from fsrl.experiments.assembly.write_localization import (
    exact_support_innovations,
    matrix_norm,
    norm_match,
    readout_effective_margin_fields,
    trace_support_trial,
    validate_registered_sources,
)
from fsrl.infra.provenance import load_json
from fsrl.infra.study_registry import resolve_record
from fsrl.tasks.registered_protocol import load_ranking_protocol


class SupportWriteLocalizationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(11)
        self.protocol = load_ranking_protocol(resolve_record("benchmarks/liu_v2.json"))
        self.config = TrainConfig(bs=3, hs=8, cs=8, nbcues_min=8, nbcues_max=8)
        self.net = RetroModulRNN(self.config.to_model_dict())
        self.net.eval()
        self.evaluator = FrozenFastWeightEvaluator(
            self.net,
            self.config,
            self.protocol,
            cue_seed=5,
            support_seed=7,
        )
        self.geometry = build_complete_graph_geometry(self.protocol)

    def test_trace_reproduces_endpoint_and_registered_write_timing(self):
        initial = self.evaluator.initialize_fast_weights()
        trace = trace_support_trial(self.evaluator, initial, 0)
        expected = self.evaluator.advance_support_trial(initial, 0)
        torch.testing.assert_close(trace.final_fast_weights, expected)
        self.assertEqual(trace.da.shape, (self.config.bs, self.config.triallen))
        self.assertEqual(
            trace.intended_increment.shape,
            (self.config.bs, self.config.triallen, self.config.hs, self.config.hs),
        )
        self.assertEqual(trace.forward_max_abs_error, 0.0)
        self.assertEqual(
            float(torch.max(torch.abs(trace.actual_increment[:, :2]))), 0.0
        )
        self.assertGreater(
            float(torch.max(torch.abs(trace.actual_increment[:, 2:]))), 0.0
        )

    def test_da_eligibility_decomposition_is_exact(self):
        initial = self.evaluator.initialize_fast_weights()
        plus = trace_support_trial(self.evaluator, initial, 0)
        zero = trace_support_trial(
            self.evaluator,
            initial,
            0,
            evidence_scales=np.zeros(self.config.bs),
        )
        da_plus = plus.da[:, :, None, None]
        da_zero = zero.da[:, :, None, None]
        da_component = (
            0.5
            * (da_plus - da_zero)
            * (plus.eligibility_before + zero.eligibility_before)
        )
        eligibility_component = (
            0.5
            * (da_plus + da_zero)
            * (plus.eligibility_before - zero.eligibility_before)
        )
        torch.testing.assert_close(
            plus.intended_increment - zero.intended_increment,
            da_component + eligibility_component,
        )

    def test_explicit_effective_modulation_reproduces_standard_readout(self):
        fast_weights = self.evaluator.learn_fast_weights(FastWeightIntervention.INTACT)
        standard = readout_margin_fields(self.evaluator, fast_weights, self.geometry)
        explicit = readout_effective_margin_fields(
            self.evaluator, self.net.alpha * fast_weights, self.geometry
        )
        np.testing.assert_allclose(explicit, standard, atol=1e-6)

    def test_norm_match_preserves_target_matrix_norm(self):
        candidate = torch.randn(3, 4, 4)
        target = torch.randn(3, 4, 4)
        matched = norm_match(candidate, target)
        torch.testing.assert_close(matrix_norm(matched), matrix_norm(target))

    def test_exact_innovations_follow_every_support_slot(self):
        exact = exact_support_innovations(
            self.evaluator, self.protocol, temperature=0.05
        )
        self.assertEqual(
            exact.delta_q.shape,
            (self.protocol.support_trials, self.config.bs, self.protocol.n_items),
        )
        self.assertTrue(np.all(exact.information_gain >= -1e-12))
        self.assertGreater(float(np.max(exact.q_norm)), 0.0)

    def test_registered_sources_are_immutable(self):
        specification = load_json(
            resolve_record("benchmarks/support_write_localization_v1.json")
        )
        validation = validate_registered_sources(specification)
        self.assertEqual(len(validation["pilot_artifacts"]), 2)

    def test_result_reports_all_seeds_and_separates_exploratory_checks(self):
        result = load_json(resolve_record("results/support_write_localization_v1.json"))
        self.assertEqual(set(result["pilot_seeds"]), {"1901", "1902"})
        self.assertNotIn(
            "alpha_functional_amplification_replicated_across_pilot_seeds",
            result["overall_diagnosis"],
        )
        self.assertIn(
            "alpha_functional_amplification_replicated_across_pilot_seeds",
            result["overall_secondary_exploratory_diagnosis"],
        )
