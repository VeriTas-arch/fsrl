import unittest

import numpy as np
import torch
import torch.nn.functional as F

from fsrl.experiments.minimal_learner.data import ModelBatch, generic_batch
from fsrl.experiments.minimal_learner.model import make_model as parent_model
from fsrl.experiments.minimal_learner.protocol import task_generator
from fsrl.experiments.quantized_learner.encoding import encode_batch
from fsrl.experiments.quantized_learner.protocol import (
    make_model,
    resolved_specification,
    run_directory,
    specification,
)
from fsrl.experiments.quantized_learner.reference import rollout
from fsrl.experiments.training_strategy.batches import sample_episodes
from fsrl.infra.provenance import tensor_hashes


class QuantizedModelTests(unittest.TestCase):
    def setUp(self):
        self.candidate = specification()
        self.spec = resolved_specification()
        self.codebook = np.asarray(self.candidate["encoding"]["codebook"])
        self.batch = generic_batch(
            sample_episodes(task_generator(), np.random.default_rng(973001), 3)
        )
        self.uniforms = np.random.default_rng(973002).random(
            self.batch.arrays["signed"].shape
        )

    def test_two_scalars_from_first_step_and_independent_reference(self):
        for condition in self.candidate["seeds"]["conditions"]:
            with self.subTest(condition=condition):
                model = make_model(self.spec).double()
                self.assertEqual(
                    set(dict(model.named_parameters())), {"raw_eta", "raw_global_gain"}
                )
                batch, _ = encode_batch(
                    self.batch, condition, self.uniforms, self.codebook
                )
                observed = model(*batch.tensors("cpu", dtype=torch.float64))
                expected = rollout(
                    batch.arrays,
                    eta=model.eta.item(),
                    gain=model.global_gain.item(),
                    epsilon=model.epsilon,
                )
                np.testing.assert_allclose(
                    observed[0].detach(), expected["margins"], atol=1e-9, rtol=1e-7
                )
                np.testing.assert_allclose(
                    observed[3].detach(), expected["w"], atol=1e-9, rtol=1e-7
                )
                self.assertEqual(observed[-1].numel(), 0)
                optimizer = torch.optim.Adam(
                    model.parameters(), lr=self.spec["optimization"]["learning_rate"]
                )
                before = tensor_hashes(model)
                signs = torch.as_tensor(2 * batch.arrays["targets"] - 1)
                F.softplus(-signs * observed[0]).mean().backward()
                for parameter in model.parameters():
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())
                    self.assertGreater(float(parameter.grad.abs().max()), 1e-10)
                optimizer.step()
                for name, parameter in model.named_parameters():
                    self.assertNotEqual(before[name], tensor_hashes(model)[name])
                    self.assertEqual(optimizer.state[parameter]["step"].item(), 1)

    def test_exact_path_preserves_original_score_model(self):
        original_spec = resolved_specification()
        original_spec["seeds"]["conditions"] = ["score_only"]
        original = parent_model("score_only", original_spec).double()
        model = make_model(self.spec).double()
        encoded, _ = encode_batch(self.batch, "exact", self.uniforms, self.codebook)
        for left, right in zip(
            original(*self.batch.tensors("cpu", torch.float64)),
            model(*encoded.tensors("cpu", torch.float64)),
            strict=True,
        ):
            torch.testing.assert_close(left, right, atol=0, rtol=0)

    def test_query_readonly_reset_and_no_hidden_metadata_access(self):
        model = make_model(self.spec).double()
        encoded, _ = encode_batch(
            self.batch, "persistent", self.uniforms, self.codebook
        )
        args = encoded.tensors("cpu", torch.float64)
        before = tensor_hashes(model)
        observed = model(*args)
        reversed_query = torch.cat((args[-1][..., 15:], args[-1][..., :15]), dim=-1)
        reversed_output = model(*args[:-1], reversed_query)
        torch.testing.assert_close(observed[0], -reversed_output[0], atol=0, rtol=0)
        torch.testing.assert_close(observed[3], reversed_output[3], atol=0, rtol=0)
        torch.testing.assert_close(model(*args)[0], observed[0], atol=0, rtol=0)
        bare = ModelBatch(
            {
                key: encoded.arrays[key]
                for key in (
                    "support_cues",
                    "signed",
                    "retention",
                    "local_evidence",
                    "query_cues",
                )
            }
        )
        torch.testing.assert_close(
            model(*bare.tensors("cpu", torch.float64))[0], observed[0], atol=0, rtol=0
        )
        self.assertEqual(before, tensor_hashes(model))

    def test_bad_schedule_does_not_silently_change_the_candidate(self):
        for field in ("signed", "retention"):
            arrays = {key: value.copy() for key, value in self.batch.arrays.items()}
            keys = arrays["support_pairs"][0]
            same = np.all(np.sort(keys, axis=-1) == np.sort(keys[0]), axis=-1)
            last = np.flatnonzero(same)[-1]
            arrays[field][last, 0] *= -1 if field == "signed" else 0
            if field == "retention":
                arrays[field][last, 0] = 1 - self.batch.arrays[field][last, 0]
            with self.subTest(field=field), self.assertRaises(ValueError):
                encode_batch(
                    ModelBatch(arrays), "persistent", self.uniforms, self.codebook
                )

    def test_task_stream_is_independent_of_encoding_rng_consumption(self):
        baseline = self.batch.fingerprint()
        for condition in self.candidate["seeds"]["conditions"]:
            encode_batch(self.batch, condition, self.uniforms, self.codebook)
        self.assertEqual(self.batch.fingerprint(), baseline)
        replay = generic_batch(
            sample_episodes(task_generator(), np.random.default_rng(973001), 3)
        )
        self.assertEqual(replay.fingerprint(), baseline)

    def test_training_identity_is_prospective_and_not_an_old_fit(self):
        self.assertEqual(self.spec["seeds"]["mandatory"], [2114, 2115, 2116])
        self.assertEqual(self.spec["optimization"]["total_episode_exposures"], 48000)
        for seed, condition in ((2111, "exact"), (2114, "score_only")):
            with self.assertRaises(ValueError):
                run_directory(seed, condition)


if __name__ == "__main__":
    unittest.main()
