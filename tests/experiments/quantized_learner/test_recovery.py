import unittest
from copy import deepcopy

import numpy as np
from scipy.special import logsumexp

from fsrl.experiments.minimal_learner.data import ModelBatch, generic_batch
from fsrl.experiments.minimal_learner.protocol import task_generator
from fsrl.experiments.quantized_learner.encoding import encode_batch
from fsrl.experiments.quantized_learner.recovery import (
    CONDITIONS,
    decode_choices,
    generate_choices,
    integrated_log_likelihood,
    recovery_summary,
)
from fsrl.experiments.quantized_learner.recovery_inputs import (
    ObservedDesign,
    nuisance_pool,
)
from fsrl.experiments.quantized_learner.reference import rollout
from fsrl.experiments.training_strategy.batches import sample_episodes
from fsrl.tasks.subject_encoding import SubjectEncodingState


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.batch = generic_batch(
            sample_episodes(task_generator(), np.random.default_rng(974001), 1)
        )
        self.design = ObservedDesign.from_batch(self.batch, 0)
        self.settings = {
            "eta_grid": [0.25, 0.6],
            "gain_grid": [1, 4],
            "choice_repetitions": 2,
            "nuisance_draws": 8,
            "integration_check_draws": 16,
        }
        self.codebook = np.array([-1, -1 / 3, 1 / 3, 1])
        self.options = {"temperature": 0.25, "epsilon": 1e-8}

    def test_hidden_metadata_does_not_enter_design_or_decoder(self):
        changed = {key: value.copy() for key, value in self.batch.arrays.items()}
        for key in set(changed) - {"support_cues", "signed", "query_cues"}:
            changed[key][...] = 0
        other = ObservedDesign.from_batch(ModelBatch(changed), 0)
        counts = np.zeros((12, 28), dtype=np.int64)
        outputs = [
            decode_choices(
                design,
                counts,
                np.random.default_rng(19),
                self.settings,
                self.codebook,
                **self.options,
            )
            for design in (self.design, other)
        ]
        np.testing.assert_array_equal(outputs[0][0], outputs[1][0])
        self.assertEqual(outputs[0][1], outputs[1][1])
        self.assertEqual(
            set(vars(self.design)), {"support_cues", "signed", "query_cues"}
        )
        self.assertFalse(self.design.signed.flags.writeable)
        self.assertFalse(
            np.shares_memory(self.design.signed, self.batch.arrays["signed"])
        )

    def test_prior_preserves_subject_correlations_and_stable_admission(self):
        batch, witness = nuisance_pool(self.design, np.random.default_rng(20), 16)
        relations = self.design.relations()
        np.testing.assert_array_equal(
            batch.arrays["retention"],
            batch.arrays["retention"][relations["first"]][relations["inverse"]],
        )
        for index, latent in enumerate(witness["prior_latents"]):
            state = SubjectEncodingState(
                latent[0], tuple(latent[1:-1]), latent[-1], 0.1
            )
            expected = [
                state.relation_reliability(int(i), int(j), int(d))
                for (i, j), d in zip(
                    relations["pairs"], relations["distances"], strict=True
                )
            ]
            np.testing.assert_array_equal(witness["probabilities"][index], expected)
        self.assertEqual(batch.arrays["support_cues"].shape[1], 16)
        self.assertEqual(
            set(witness),
            {
                "encoding_uniforms",
                "admission_uniforms",
                "probabilities",
                "prior_latents",
            },
        )

    def test_joint_likelihood_matches_explicit_shared_state_enumeration(self):
        batch, witness = nuisance_pool(self.design, np.random.default_rng(21), 16)
        counts = np.random.default_rng(22).integers(0, 3, size=(12, 28))
        actual = integrated_log_likelihood(
            batch,
            witness["encoding_uniforms"],
            counts,
            self.settings,
            self.codebook,
            **self.options,
        )
        for family, condition in enumerate(CONDITIONS):
            encoded, _ = encode_batch(
                batch, condition, witness["encoding_uniforms"], self.codebook
            )
            for e, eta in enumerate(self.settings["eta_grid"]):
                for g, gain in enumerate(self.settings["gain_grid"]):
                    margins = rollout(encoded.arrays, eta=eta, gain=gain, epsilon=1e-8)[
                        "margins"
                    ]
                    logits = margins / self.options["temperature"]
                    for b, budget in enumerate((8, 16)):
                        for subject in (0, 11):
                            joint = []
                            for row in logits[:budget]:
                                log_probability = sum(
                                    -np.logaddexp(0, -value) * count
                                    - np.logaddexp(0, value) * (2 - count)
                                    for value, count in zip(
                                        row, counts[subject], strict=True
                                    )
                                )
                                joint.append(log_probability)
                            expected = logsumexp(joint) - np.log(budget)
                            self.assertAlmostEqual(
                                actual[b, subject, family, 2 * e + g],
                                expected,
                                places=10,
                            )

    def test_check_budget_uses_nested_pool_not_a_new_random_draw(self):
        batch, witness = nuisance_pool(self.design, np.random.default_rng(23), 16)
        counts = np.ones((12, 28), dtype=np.int64)
        altered = {key: value.copy() for key, value in batch.arrays.items()}
        altered["retention"][:, 8:] = 0
        result = [
            integrated_log_likelihood(
                pool,
                witness["encoding_uniforms"],
                counts,
                self.settings,
                self.codebook,
                **self.options,
            )
            for pool in (batch, ModelBatch(altered))
        ]
        np.testing.assert_array_equal(result[0][0], result[1][0])
        self.assertGreater(np.max(np.abs(result[0][1] - result[1][1])), 0)

    def test_generation_has_one_state_per_setting_and_reproducible_choices(self):
        result = [
            generate_choices(
                self.design,
                np.random.default_rng(24),
                self.settings,
                self.codebook,
                **self.options,
            )
            for _ in range(2)
        ]
        self.assertEqual(result[0]["choices"].shape, (12, 2, 28))
        self.assertEqual(result[0]["w"].shape, (12, 15))
        for key in result[0]:
            np.testing.assert_array_equal(result[0][key], result[1][key])
        np.testing.assert_array_equal(
            result[0]["left_counts"], result[0]["choices"].sum(axis=1)
        )
        relations = self.design.relations()
        # Persistent's four parameter settings reuse each relation across repeats.
        values = result[0]["code_indices"][:, 4:8]
        np.testing.assert_array_equal(
            values, values[relations["first"]][relations["inverse"]]
        )

    def test_summary_integrates_shared_parameters_after_subjects(self):
        values = np.full((2, 2, 27, 3, 9), -1000.0)
        truth = np.repeat(np.arange(3), 9)
        for setting, family in enumerate(truth):
            values[:, :, setting, family, :] = -10
        result = recovery_summary(values)
        self.assertEqual(result["outcome"], "distinguishable_on_registered_screen")
        np.testing.assert_array_equal(
            result["family_winners"], np.stack((truth, truth))
        )
        np.testing.assert_allclose(
            result["family_log_scores"][:, np.arange(27), truth], -20
        )
        # Opposite best parameters in two subjects must not become one perfect fit.
        values[0, :, 0, 0] = [-1, -100, *([-1000] * 7)]
        values[1, :, 0, 0] = [-100, -1, *([-1000] * 7)]
        result = recovery_summary(values)
        expected = logsumexp(values.sum(axis=0), axis=-1) - np.log(9)
        wrong = (logsumexp(values, axis=-1) - np.log(9)).sum(axis=0)
        np.testing.assert_array_equal(result["family_log_scores"], expected)
        self.assertGreater(wrong[0, 0, 0] - expected[0, 0, 0], 90)

    def test_ties_and_budget_instability_cannot_pass(self):
        values = np.full((1, 2, 27, 3, 9), -100.0)
        tied = recovery_summary(values)
        self.assertEqual(tied["outcome"], "specificity_unresolved")
        self.assertEqual(tied["confusion"].sum(), 0)
        np.testing.assert_array_equal(tied["unidentified_counts"], np.full((2, 3), 9))
        for setting, family in enumerate(np.repeat(np.arange(3), 9)):
            values[:, :, setting, family] = -1
        values[:, 1, 0, 1] = 0
        changed = recovery_summary(values)
        self.assertEqual(changed["outcome"], "specificity_unresolved")
        self.assertFalse(changed["budget_winners_stable"])

    def test_invalid_choices_or_incomplete_pool_fail_closed(self):
        batch, witness = nuisance_pool(self.design, np.random.default_rng(25), 16)
        for bad in (-1, 0.5, 3, np.nan):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                integrated_log_likelihood(
                    batch,
                    witness["encoding_uniforms"],
                    np.full((12, 28), bad),
                    self.settings,
                    self.codebook,
                    **self.options,
                )
        settings = deepcopy(self.settings)
        settings["integration_check_draws"] = 32
        with self.assertRaises(ValueError):
            integrated_log_likelihood(
                batch,
                witness["encoding_uniforms"],
                np.zeros((12, 28)),
                settings,
                self.codebook,
                **self.options,
            )


if __name__ == "__main__":
    unittest.main()
