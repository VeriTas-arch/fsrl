"""Numerical qualification before joint-training outcome exposure."""

import unittest

import numpy as np
import torch

from fsrl.experiments.relational_revision.inputs import make_panel
from fsrl.experiments.relational_revision.model import trajectory as native_trajectory
from fsrl.experiments.revision_learning.execution import PROTOCOL, loss_for
from fsrl.experiments.revision_learning.inputs import fingerprint, training_batch
from fsrl.experiments.revision_learning.model import (
    PersistentGRU,
    advance,
    forward,
    initial,
    make_model,
    read,
)
from fsrl.infra.provenance import load_json


class RevisionLearningTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_gru_independent_equation_and_query_copies(self):
        for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
            net = make_model("gru", 975001, device)
            self.assertIsInstance(net, PersistentGRU)
            x = torch.linspace(-1, 1, 76, device=device).reshape(2, 38)
            h = torch.linspace(-0.2, 0.2, 400, device=device).reshape(2, 200)
            ir, iz, inn = (x @ net.cell.weight_ih.T + net.cell.bias_ih).chunk(3, 1)
            hr, hz, hn = (h @ net.cell.weight_hh.T + net.cell.bias_hh).chunk(3, 1)
            reset, z = (ir + hr).sigmoid(), (iz + hz).sigmoid()
            expected = (1 - z) * (inn + reset * hn).tanh() + z * h
            torch.testing.assert_close(net.cell(x, h), expected)
            panel = make_panel(975002, 2, 0, "chain", "stable")
            support = torch.as_tensor(panel["support"], device=device)
            query = torch.as_tensor(panel["query"][:3], device=device)
            state = initial(net, 2, support)
            for trial in support:
                state = advance(net, trial, state)
            saved = state.detach().clone()
            margins = read(net, state, query)
            torch.testing.assert_close(state, saved, rtol=0, atol=0)
            separately = []
            for q in query:
                logits = net.readout(net.advance(q, state))
                separately.append(logits[:, 1] - logits[:, 0])
            torch.testing.assert_close(margins, torch.stack(separately, 1))
            reset_state = net.advance(support[-1], support.new_zeros(2, 200))
            self.assertGreater(float((reset_state - state).abs().max().detach()), 1e-5)

    def test_native_rollout_and_gradient_flow(self):
        for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
            net = make_model("plastic", 975003, device)
            panel = make_panel(975004, 2, 1, "chain", "revision")
            support = torch.as_tensor(panel["support"], device=device)
            query = torch.as_tensor(panel["query"], device=device)
            with torch.no_grad():
                expected = native_trajectory(net, panel, batch_size=2)["margins"]
                actual, _ = forward(net, support, query, range(24, 33))
                np.testing.assert_allclose(
                    actual.cpu().numpy(), expected, rtol=1e-5, atol=1e-6
                )
            small = {
                "support": panel["support"][:3],
                "query": panel["query"][:2],
                "signs": panel["final_sign"][:, None, :2],
                "prefixes": np.asarray([3]),
            }
            loss, _ = loss_for(net, small, 1e-4)
            loss.backward()
            for name in (
                "w",
                "alpha",
                "etaet",
                "h2DA.weight",
                "i2h.weight",
                "h2o.weight",
            ):
                gradient = dict(net.named_parameters())[name].grad
                self.assertTrue(torch.isfinite(gradient).all())
                self.assertGreater(float(gradient.abs().max()), 0)

    def test_shared_stream_and_information_boundary(self):
        spec = load_json(PROTOCOL)
        spec["training"]["batch_size"] = 2
        for step in (0, 1, 3):
            a = training_batch(975005, step, spec)
            b = training_batch(975005, step, spec)
            self.assertEqual(fingerprint(a), fingerprint(b))
            self.assertEqual(a["support"].shape[-1], 38)
            np.testing.assert_array_equal(a["support"][..., 34], a["support"][..., 37])
            self.assertFalse(a["query"][..., [33, 34, 35, 36, 37]].any())
            self.assertEqual(a["signs"].shape, (2, 1, 8))
            self.assertEqual(set(a), {"support", "query", "signs", "prefixes"})


if __name__ == "__main__":
    unittest.main()
