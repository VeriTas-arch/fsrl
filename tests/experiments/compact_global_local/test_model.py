from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from fsrl.core.local_trace import ConjunctiveLocalTrace
from fsrl.experiments.compact_global_local.model import (
    CompactModelConfig,
    CompactPlasticRNN,
    CompactRecurrentSequence,
    PackedLocalTrace,
)
from fsrl.experiments.compact_global_local.optimization import margin_loss


class PackedLocalTraceTests(unittest.TestCase):
    def test_packed_trace_exactly_preserves_full_trace_reads(self):
        torch.manual_seed(7)
        cues = torch.sign(torch.randn(9, 10))
        values = torch.linspace(-1.0, 1.0, 9)
        full = ConjunctiveLocalTrace(5, device="cpu")
        packed = PackedLocalTrace(5, device="cpu")
        full_state = full.write(full.initial_state(9), cues, values)
        packed_state = packed.write(packed.initial_state(9), cues, values)

        full_raw, _ = full.read(full_state, cues.roll(1, dims=0))
        packed_raw = packed.read(packed_state, cues.roll(1, dims=0))
        torch.testing.assert_close(packed_raw, full_raw, atol=1e-6, rtol=1e-6)
        self.assertEqual(packed_state.shape, (9, 10))

    def test_reversal_negates_key(self):
        trace = PackedLocalTrace(4, device="cpu")
        pair = torch.tensor([[1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0]])
        reversed_pair = torch.cat((pair[:, 4:], pair[:, :4]), dim=1)
        torch.testing.assert_close(trace.key(reversed_pair), -trace.key(pair))


class CompactModelTests(unittest.TestCase):
    def test_layout_and_parameter_heads_are_compact(self):
        config = CompactModelConfig(cue_size=15, hidden_size=11)
        model = CompactPlasticRNN(config, device="cpu")
        self.assertEqual(config.input_size, 32)
        self.assertEqual(config.response_index, 30)
        self.assertEqual(config.evidence_index, 31)
        self.assertEqual(model.h2margin.out_features, 1)
        self.assertEqual(model.h2modulation.out_features, 1)
        self.assertFalse(hasattr(model, "h2v"))

    def test_registered_parameter_count_is_87005(self):
        config = CompactModelConfig(cue_size=15, hidden_size=200)
        model = CompactPlasticRNN(config, device="cpu")
        local = PackedLocalTrace(config.cue_size, device="cpu")
        count = sum(parameter.numel() for parameter in model.parameters()) + sum(
            parameter.numel() for parameter in local.parameters()
        )
        self.assertEqual(count, 87005)

    def test_three_steps_are_required_to_apply_eligibility(self):
        torch.manual_seed(13)
        config = CompactModelConfig(cue_size=3, hidden_size=5)
        model = CompactPlasticRNN(config, device="cpu")
        sequence = CompactRecurrentSequence(model)
        inputs = torch.zeros(3, 2, config.input_size)
        inputs[0, :, :6] = torch.tensor([[1.0, -1.0, 1.0, -1.0, 1.0, -1.0]] * 2)
        inputs[1, :, config.response_index] = 1.0
        zero_p = model.initial_fast_weights(2)
        _, _, _, _, one_step = sequence(
            inputs[:1],
            model.initial_hidden(2),
            model.initial_eligibility(2),
            zero_p,
            True,
        )
        _, _, _, _, two_steps = sequence(
            inputs[:2],
            model.initial_hidden(2),
            model.initial_eligibility(2),
            zero_p,
            True,
        )
        _, _, _, _, three_steps = sequence(
            inputs,
            model.initial_hidden(2),
            model.initial_eligibility(2),
            zero_p,
            True,
        )
        torch.testing.assert_close(one_step, zero_p)
        torch.testing.assert_close(two_steps, zero_p)
        self.assertGreater(float(three_steps.detach().abs().sum()), 0.0)

    def test_query_does_not_retain_proposed_writes(self):
        torch.manual_seed(17)
        config = CompactModelConfig(cue_size=3, hidden_size=4)
        model = CompactPlasticRNN(config, device="cpu")
        sequence = CompactRecurrentSequence(model)
        inputs = torch.randn(2, 2, config.input_size)
        initial = torch.randn(2, 4, 4)
        outputs = sequence(
            inputs,
            model.initial_hidden(2),
            model.initial_eligibility(2),
            initial,
            False,
        )
        torch.testing.assert_close(outputs[-1], initial)

    def test_single_margin_loss_equals_symmetric_two_logit_cross_entropy(self):
        margin = torch.tensor([[-2.0], [-0.25], [0.5], [3.0]])
        targets = torch.tensor([0, 1, 0, 1])
        logits = torch.cat((-0.5 * margin, 0.5 * margin), dim=1)
        torch.testing.assert_close(
            margin_loss(margin, targets),
            F.cross_entropy(logits, targets),
            atol=1e-7,
            rtol=1e-7,
        )


if __name__ == "__main__":
    unittest.main()
