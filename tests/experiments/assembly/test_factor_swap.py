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
from fsrl.experiments.assembly.factor_swap import (
    compose_factors,
    donor_indices,
    norm_match_trailing,
    readout_effective_margin_fields_batched,
    validate_registered_sources,
)
from fsrl.experiments.assembly.trajectory import readout_margin_fields
from fsrl.experiments.assembly.write_localization import trace_support_trial
from fsrl.infra.provenance import load_json
from fsrl.infra.study_registry import resolve_record
from fsrl.tasks.registered_protocol import load_ranking_protocol


class SupportFactorSwapTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)
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

    def test_factor_composition_reproduces_matched_write(self):
        initial = self.evaluator.initialize_fast_weights()
        plus = trace_support_trial(self.evaluator, initial, 0)
        zero = trace_support_trial(
            self.evaluator,
            initial,
            0,
            evidence_scales=np.zeros(self.config.bs, dtype=np.float32),
        )
        steps = torch.as_tensor((2, 3), device=plus.da.device)
        da_plus = torch.index_select(plus.da, 1, steps)
        da_zero = torch.index_select(zero.da, 1, steps)
        e_plus = torch.index_select(plus.eligibility_before, 1, steps)
        e_zero = torch.index_select(zero.eligibility_before, 1, steps)
        composed = compose_factors(
            0.5 * (da_plus + da_zero),
            da_plus - da_zero,
            0.5 * (e_plus + e_zero),
            e_plus - e_zero,
        )
        direct = torch.sum(
            torch.index_select(
                plus.intended_increment - zero.intended_increment, 1, steps
            ),
            dim=1,
        )
        torch.testing.assert_close(composed, direct)

    def test_norm_match_supports_step_and_condition_axes(self):
        candidate = torch.randn(4, 3, 2, 5, 5)
        target = torch.randn(4, 3, 2, 5, 5)
        matched = norm_match_trailing(candidate, target, trailing_dimensions=3)
        expected = torch.linalg.vector_norm(target.flatten(start_dim=2), dim=-1)
        observed = torch.linalg.vector_norm(matched.flatten(start_dim=2), dim=-1)
        torch.testing.assert_close(observed, expected)

    def test_batched_effective_readout_matches_standard_readout(self):
        fast_weights = self.evaluator.learn_fast_weights(FastWeightIntervention.INTACT)
        standard = readout_margin_fields(self.evaluator, fast_weights, self.geometry)
        modulation = self.net.alpha * fast_weights
        batched = readout_effective_margin_fields_batched(
            self.evaluator,
            torch.stack((modulation, modulation)),
            self.geometry,
        )
        np.testing.assert_allclose(batched[0], standard, atol=1e-6)
        np.testing.assert_allclose(batched[1], standard, atol=1e-6)

    def test_donor_pairing_is_within_exposure_retained_and_nonself(self):
        trials = self.protocol.support_trials
        subjects = 2
        retained = np.ones((trials, subjects), dtype=bool)
        exposure = np.empty((trials, subjects), dtype=np.int64)
        relations = np.empty((trials, subjects, 2), dtype=np.int64)
        for subject in range(subjects):
            for trial in range(trials):
                block = trial // len(self.protocol.support_pairs_higher_lower)
                relation = self.protocol.support_pairs_higher_lower[
                    trial % len(self.protocol.support_pairs_higher_lower)
                ]
                exposure[trial, subject] = block + 1
                relations[trial, subject] = relation
        retained[0, 0] = False
        donors = donor_indices(
            retained,
            exposure,
            relations,
            self.protocol.support_pairs_higher_lower,
        )
        for trial, subject in zip(*np.nonzero(retained)):
            donor = donors[trial, subject]
            self.assertNotEqual(donor, trial)
            self.assertTrue(retained[donor, subject])
            self.assertEqual(exposure[donor, subject], exposure[trial, subject])
        self.assertEqual(donors[0, 0], -1)

    def test_registered_sources_are_immutable(self):
        specification = load_json(
            resolve_record("benchmarks/support_factor_swap_v1.json")
        )
        validation = validate_registered_sources(specification)
        self.assertEqual(len(validation["pilot_artifacts"]), 2)

    def test_result_reports_both_pilots_and_keeps_formal_seeds_deferred(self):
        result = load_json(resolve_record("results/support_factor_swap_v1.json"))
        self.assertEqual(set(result["pilot_seeds"]), {"1901", "1902"})
        self.assertEqual(
            result["overall_diagnosis"]["formal_confirmation_status"],
            "deferred; formal seeds remain untouched",
        )
        self.assertTrue(
            result["overall_diagnosis"][
                "eligibility_identity_transfer_replicated_across_pilot_seeds"
            ]
        )
        self.assertFalse(
            result["overall_diagnosis"][
                "history_policy_attribution_competent_replicated_across_pilot_seeds"
            ]
        )


if __name__ == "__main__":
    unittest.main()
