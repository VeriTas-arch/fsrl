"""Independent task algebra, identifiability and ABI checks before model exposure."""

import unittest

import numpy as np
import torch

from fsrl.core.sequence import RecurrentSequence
from fsrl.experiments.memory_structure.model import forward_batch
from fsrl.experiments.relational_revision import baselines, model
from fsrl.experiments.relational_revision.inputs import GROUPS, QUERIES, make_panel
from fsrl.experiments.relational_revision.statistics import endpoints, interval
from fsrl.experiments.training_strategy.batches import TensorBatch


class RevisionInputsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_same_first_evidence_cannot_identify_outlier(self):
        for history in (0, 1):
            for topology in ("chain", "star"):
                real = make_panel(970001, 12, history, topology, "revision")
                error = make_panel(970001, 12, history, topology, "outlier")
                stable = make_panel(970001, 12, history, topology, "stable")
                np.testing.assert_array_equal(
                    real["support"][:25], error["support"][:25]
                )
                np.testing.assert_array_equal(
                    real["support"][:24], stable["support"][:24]
                )
                np.testing.assert_array_equal(
                    error["final_scores"], stable["final_scores"]
                )
                self.assertFalse(
                    np.array_equal(real["support"][25:], error["support"][25:])
                )

    def test_signed_remote_crossover_and_invariant_constraints(self):
        panels = [
            make_panel(970001, 12, h, "chain", "revision", sigma=0) for h in (0, 1)
        ]
        np.testing.assert_array_equal(panels[0]["q"][24], panels[1]["q"][24])
        np.testing.assert_array_equal(panels[0]["old_scores"], panels[1]["old_scores"])
        for h, panel in enumerate(panels):
            change = panel["final_scores"] - panel["old_scores"]
            np.testing.assert_allclose(change[:, 1] - change[:, 2], 0.4)
            np.testing.assert_allclose(change[:, 0] - change[:, 3], (1 - 2 * h) * 0.4)
            for group in GROUPS[h]:
                for i in group:
                    for j in group:
                        np.testing.assert_allclose(
                            change[:, i] - change[:, j], 0, atol=1e-15
                        )
            ad = np.flatnonzero(np.all(QUERIES == (0, 3), axis=1))[0]
            self.assertTrue(np.all(panel["changed_remote"][:, ad]))

    def test_inputs_encode_observations_not_hidden_world(self):
        panel = make_panel(970001, 12, 0, "chain", "outlier")
        support = panel["support"]
        self.assertEqual(support.shape, (32, 4, 12, 38))
        np.testing.assert_array_equal(support[:, 0, :, 34], panel["q"])
        np.testing.assert_array_equal(support[:, 0, :, 37], panel["q"])
        self.assertFalse(np.any(support[:, 1:, :, [34, 37]]))
        self.assertFalse(np.any(support[..., [33, 35, 36]]))
        self.assertFalse(np.any(panel["query"][..., [33, 34, 35, 36, 37]]))
        for t in range(32):
            for b in range(12):
                i, j = panel["pairs"][t, b]
                np.testing.assert_array_equal(
                    support[t, 0, b, :15], panel["codes"][b, i]
                )
                np.testing.assert_array_equal(
                    support[t, 0, b, 15:30], panel["codes"][b, j]
                )

    def test_filter_against_information_form(self):
        panel = make_panel(970002, 3, 1, "star", "revision")
        actual = baselines.trajectory(panel, "covariance_filter")["margins"]
        for subject in range(3):
            mean, covariance = np.zeros(8), np.eye(8) * 0.25
            for t, pairs in enumerate(panel["pairs"]):
                i, j = pairs[subject]
                b = np.eye(8)[i] - np.eye(8)[j]
                precision = np.linalg.inv(covariance + 0.0005 * np.eye(8))
                natural = precision @ mean + b * panel["q"][t, subject] / 0.05**2
                covariance = np.linalg.inv(precision + np.outer(b, b) / 0.05**2)
                mean = covariance @ natural
                if t >= 23:
                    expected = (mean[QUERIES[:, 0]] - mean[QUERIES[:, 1]]) / 0.1
                    np.testing.assert_allclose(
                        actual[subject, t - 23], expected, atol=1e-10
                    )

    def test_native_rollout_and_query_immutability_cpu_cuda(self):
        panel = make_panel(970003, 2, 0, "chain", "revision")
        devices = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
        for device in devices:
            net = model.network(device=device).requires_grad_(False).eval()
            query = torch.as_tensor(panel["query"], device=device)
            batch = TensorBatch(
                support_inputs=torch.as_tensor(panel["support"], device=device),
                local_evidence=torch.as_tensor(panel["q"], device=device),
                query_inputs=query.transpose(0, 1).reshape(2, 112, 38),
                targets=torch.zeros(112, dtype=torch.long, device=device),
            )
            with torch.inference_mode():
                reference = forward_batch(net, None, RecurrentSequence(net), batch)
                expected = (
                    (reference.logits[:, 1] - reference.logits[:, 0]).reshape(56, 2).T
                )
                raw = model.trajectory(net, panel)
                np.testing.assert_allclose(
                    raw["margins"][:, -1], expected.cpu(), atol=1e-5, rtol=1e-4
                )
                weights = reference.weights.clone()
                model.probe(net, weights, query)
                torch.testing.assert_close(weights, reference.weights, atol=0, rtol=0)

    def test_endpoint_orientation_and_paired_cohort_weighting(self):
        panel = make_panel(970004, 4, 0, "chain", "revision")
        stable = np.repeat((2 * panel["old_sign"])[:, None], 9, axis=1)
        revision = np.repeat((2 * panel["final_sign"])[:, None], 9, axis=1)
        revision[:, 0] = 0
        outlier = stable.copy()
        outlier[:, 1] += 0.5
        values = endpoints(stable, revision, outlier, panel)
        np.testing.assert_allclose(
            values["revision_gain"], np.log(2) - np.logaddexp(0, -2)
        )
        self.assertTrue(np.all(values["outlier_recovery"] > 0))
        np.testing.assert_allclose(values["preservation_cost"], 0)
        self.assertTrue(np.all(values["preservation_from_before"] < 0))
        draws = np.tile(np.arange(4), (2, 10, 1))
        summary = interval(np.array([[1] * 4, [3] * 4]), draws)
        self.assertEqual(summary, {"mean": 2.0, "ci95": [2.0, 2.0]})


if __name__ == "__main__":
    unittest.main()
