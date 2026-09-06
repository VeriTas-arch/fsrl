import unittest

import numpy as np
import torch

from fsrl.experiments.adaptive_plasticity.data import (
    clustered_batch,
    model_tensors,
    occurrence_indices,
    relation_slots,
)
from fsrl.experiments.adaptive_plasticity.model import make_model
from fsrl.experiments.adaptive_plasticity.protocol import resolved_specification
from fsrl.experiments.adaptive_plasticity.reference import rollout
from fsrl.experiments.minimal_learner.data import ModelBatch


def fixture() -> ModelBatch:
    first = np.asarray([1, -1, 1], np.float32)
    second = np.asarray([-1, 1, 1], np.float32)
    third = np.asarray([1, 1, -1], np.float32)
    relations = [(first, second), (second, third)]
    cues = np.asarray(
        [np.concatenate(relations[index]) for _ in range(4) for index in range(2)]
    )[:, None]
    signed = np.asarray([[1 / 3], [2 / 3]] * 4, np.float32)
    return ModelBatch(
        {
            "support_cues": cues,
            "signed": signed,
            "retention": np.ones_like(signed),
            "probabilities": np.ones_like(signed),
            "local_evidence": np.zeros_like(signed),
            "query_cues": cues[:2].transpose(1, 0, 2),
            "targets": np.ones((1, 2)),
        }
    )


class AdaptivePlasticityModelTests(unittest.TestCase):
    def setUp(self):
        self.spec = resolved_specification()
        self.spec["task"]["cue_size"] = 3
        self.spec["task"]["max_edges"] = 2
        self.batch = fixture()

    def test_relation_slots_and_occurrences(self):
        np.testing.assert_array_equal(
            relation_slots(self.batch.arrays["support_cues"])[:, 0],
            [0, 1, 0, 1, 0, 1, 0, 1],
        )
        np.testing.assert_array_equal(
            occurrence_indices(self.batch.arrays["support_cues"])[:, 0],
            [0, 0, 1, 1, 2, 2, 3, 3],
        )

    def test_adaptive_recurrence_matches_float64_reference(self):
        for condition in ("fixed_eta_resampled", "adaptive_eta_resampled"):
            model = make_model(condition, self.spec).double()
            with torch.no_grad():
                observed = model(*model_tensors(self.batch, "cpu", torch.float64))
            expected = rollout(
                self.batch,
                eta=model.eta.item(),
                gain=model.global_gain.item(),
                epsilon=model.epsilon,
                adaptive=model.adaptive,
            )
            np.testing.assert_allclose(observed[0], expected["margins"], atol=1e-12)
            np.testing.assert_allclose(observed[1], expected["w"], atol=1e-12)

    def test_relation_and_global_are_identical_on_balanced_schedule(self):
        relation = make_model("adaptive_eta_resampled", self.spec, scheduler="relation")
        global_model = make_model(
            "adaptive_eta_resampled", self.spec, scheduler="global"
        )
        global_model.load_state_dict(relation.state_dict())
        with torch.no_grad():
            first = relation(*model_tensors(self.batch, "cpu"))
            second = global_model(*model_tensors(self.batch, "cpu"))
        for a, b in zip(first[:2], second[:2], strict=True):
            torch.testing.assert_close(a, b, atol=0, rtol=0)

    def test_clustered_schedule_preserves_content_and_breaks_scheduler_identity(self):
        uniforms = np.arange(8, dtype=float)[:, None] / 8
        clustered, reordered, permutation = clustered_batch(
            self.batch, uniforms, np.random.default_rng(3)
        )
        self.assertCountEqual(permutation[:, 0], range(8))
        np.testing.assert_array_equal(reordered, uniforms[permutation, [0]])
        original_rows = sorted(map(tuple, self.batch.arrays["support_cues"][:, 0]))
        clustered_rows = sorted(map(tuple, clustered.arrays["support_cues"][:, 0]))
        self.assertEqual(original_rows, clustered_rows)
        relation = rollout(
            clustered,
            eta=0.5,
            gain=1,
            epsilon=1e-8,
            adaptive=True,
            scheduler="relation",
        )
        global_result = rollout(
            clustered,
            eta=0.5,
            gain=1,
            epsilon=1e-8,
            adaptive=True,
            scheduler="global",
        )
        self.assertFalse(np.array_equal(relation["w"], global_result["w"]))

    def test_omission_changes_neither_weight_nor_efficacy(self):
        omitted = ModelBatch(
            {
                **self.batch.arrays,
                "retention": np.zeros_like(self.batch.arrays["retention"]),
            }
        )
        model = make_model("adaptive_eta_resampled", self.spec).double()
        with torch.no_grad():
            _, w, efficacy = model(*model_tensors(omitted, "cpu", torch.float64))
        torch.testing.assert_close(w, torch.zeros_like(w), atol=0, rtol=0)
        torch.testing.assert_close(
            efficacy,
            model.eta.expand_as(efficacy),
            atol=0,
            rtol=0,
        )

    def test_isolated_harmonic_accumulation_identity(self):
        relation = self.batch.arrays["support_cues"][:1]
        signed = np.asarray([[0.2], [-0.3], [0.6], [0.8]])
        isolated = ModelBatch(
            {
                **self.batch.arrays,
                "support_cues": np.repeat(relation, 4, axis=0),
                "signed": signed,
                "retention": np.ones_like(signed),
                "probabilities": np.ones_like(signed),
                "local_evidence": np.zeros_like(signed),
                "query_cues": relation.transpose(1, 0, 2),
                "targets": np.ones((1, 1)),
            }
        )
        eta = 0.5
        result = rollout(
            isolated,
            eta=eta,
            gain=1,
            epsilon=0,
            adaptive=True,
            with_sensitivity=True,
        )
        kappa = (1 - eta) / eta
        self.assertAlmostEqual(result["margins"][0, 0], signed.sum() / (kappa + 4))
        np.testing.assert_allclose(
            result["sensitivity"][:, 0, 0], np.full(4, 1 / (kappa + 4))
        )


if __name__ == "__main__":
    unittest.main()
