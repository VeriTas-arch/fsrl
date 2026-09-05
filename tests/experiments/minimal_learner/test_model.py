import unittest

import numpy as np
import torch
import torch.nn.functional as F

from fsrl.experiments.minimal_learner.data import generic_batch
from fsrl.experiments.minimal_learner.history import score_history
from fsrl.experiments.minimal_learner.model import make_model
from fsrl.experiments.minimal_learner.protocol import specification, task_generator
from fsrl.experiments.training_strategy.batches import sample_episodes


def fixture():
    spec = specification()
    spec["task"]["cue_size"] = 4
    rng = np.random.default_rng(42)
    support = torch.tensor(rng.normal(size=(4, 2, 8)), dtype=torch.float64)
    signed = torch.tensor([[0.4, -0.7], [0.2, 0.5], [-0.8, 0.1], [0.6, -0.3]])
    signed = signed.double()
    z = torch.tensor([[1, 1], [0, 0], [1, 0], [1, 1]], dtype=torch.float64)
    local_values = signed * (z + (1 - z) * 0.4)
    query = torch.tensor(rng.normal(size=(2, 3, 8)), dtype=torch.float64)
    return spec, (support, signed, z, local_values, query)


class MinimalModelTests(unittest.TestCase):
    def test_three_or_two_scalars_and_valid_constraints(self):
        spec, args = fixture()
        for condition, count in (("score_only", 2), ("score_trace", 3)):
            model = make_model(condition, spec).double()
            self.assertEqual(sum(p.numel() for p in model.parameters()), count)
            before = [p.detach().clone() for p in model.parameters()]
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            result = model(*args)
            loss = F.softplus(-result[0]).mean()
            loss.backward()
            for p in model.parameters():
                assert p.grad is not None
                self.assertGreater(float(p.grad.abs().max()), 1e-10)
            optimizer.step()
            self.assertTrue(
                all(
                    not torch.equal(p, old)
                    for p, old in zip(model.parameters(), before, strict=True)
                )
            )
            self.assertTrue(0 < model.eta.item() < 1)
            self.assertGreater(model.global_gain.item(), 0)
            self.assertEqual(result[-1].shape[-1], 16 if count == 3 else 0)

    def test_omission_is_no_update_not_zero_target(self):
        spec, args = fixture()
        model = make_model("score_only", spec).double()
        full = model(*args)
        selected = [0, 2, 3]  # Trial 1 was omitted in BOTH subjects.
        removed = model(*(value[selected] for value in args[:4]), args[4])
        torch.testing.assert_close(full[3], removed[3], atol=0, rtol=0)
        zero_target = args[1].clone()
        zero_target[1] = 0
        admitted = args[2].clone()
        admitted[1] = 1
        different = model(args[0], zero_target, admitted, args[3], args[4])
        self.assertFalse(torch.equal(full[3], different[3]))

    def test_local_rule_reversal_readonly_query_and_reset(self):
        spec, args = fixture()
        model = make_model("score_trace", spec).double()
        output = model(*args)
        assert model.local is not None
        trace = model.local.initial_state(2)
        for cues, value in zip(args[0], args[3], strict=True):
            trace = model.local.write(trace, cues, value)
        torch.testing.assert_close(trace, output[-1], atol=0, rtol=0)
        torch.testing.assert_close(output[0], output[1] + output[2], atol=0, rtol=0)
        reversed_queries = torch.cat((args[4][..., 4:], args[4][..., :4]), dim=-1)
        torch.testing.assert_close(model(*args[:4], reversed_queries)[0], -output[0])
        for index in range(3):
            single = model(*args[:4], args[4][:, index : index + 1])
            torch.testing.assert_close(single[0][:, 0], output[0][:, index])
            torch.testing.assert_close(single[3], output[3], atol=0, rtol=0)
        torch.testing.assert_close(model(*args)[0], output[0], atol=0, rtol=0)

    def test_history_matches_autograd_and_finite_difference(self):
        spec, args = fixture()
        model = make_model("score_only", spec).double()
        support, signed, z, local_values, query = args
        signed.requires_grad_(True)
        output = model(*args)
        x = (support[..., :4] - support[..., 4:]).transpose(0, 1).numpy()
        q = (query[..., :4] - query[..., 4:]).numpy()
        history = score_history(
            x,
            signed.detach().T.numpy(),
            z.T.numpy(),
            q,
            eta=model.eta.item(),
            gain=model.global_gain.item(),
            epsilon=model.epsilon,
        )
        np.testing.assert_allclose(
            history["global_margin"], output[1].detach().numpy(), atol=1e-9, rtol=1e-7
        )
        for s in range(2):
            for query_index in range(3):
                gradient = torch.autograd.grad(
                    output[1][s, query_index], signed, retain_graph=True
                )[0]
                np.testing.assert_allclose(
                    gradient[:, s],
                    history["sensitivity"][s, :, query_index],
                    atol=1e-9,
                    rtol=1e-7,
                )
        plus, minus = signed.detach().clone(), signed.detach().clone()
        plus[0, 0] += 1e-5
        minus[0, 0] -= 1e-5
        delta = (
            model(support, plus, z, local_values, query)[1]
            - model(support, minus, z, local_values, query)[1]
        ) / 2e-5
        np.testing.assert_allclose(
            delta[0].detach(), history["sensitivity"][0, 0], atol=1e-6, rtol=1e-5
        )
        np.testing.assert_allclose(
            history["sensitivity"][:, -1],
            history["direct_sensitivity"][:, -1],
            atol=0,
            rtol=0,
        )

    def test_generic_pairing_and_original_weak_evidence(self):
        generator = task_generator()
        first = generic_batch(sample_episodes(generator, np.random.default_rng(9), 3))
        second = generic_batch(sample_episodes(generator, np.random.default_rng(9), 3))
        self.assertEqual(first.fingerprint(), second.fingerprint())
        arrays = first.arrays
        expected = arrays["signed"] * (
            arrays["retention"] + (1 - arrays["retention"]) * arrays["probabilities"]
        )
        np.testing.assert_allclose(arrays["local_evidence"], expected, atol=1e-7)
        self.assertTrue(np.any(arrays["retention"] == 0))
        self.assertTrue(np.all(np.abs(expected[arrays["retention"] == 0]) > 0))
        self.assertEqual(arrays["targets"].shape, (3, 28))


if __name__ == "__main__":
    unittest.main()
