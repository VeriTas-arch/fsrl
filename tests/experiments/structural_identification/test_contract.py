import unittest

import numpy as np
import torch

from fsrl.analysis.behavioral import analyze_sampled_query_policy
from fsrl.experiments.adaptive_plasticity.data import model_tensors
from fsrl.experiments.cohort_diagnostic.statistics import cohort_record
from fsrl.experiments.minimal_learner.data import ModelBatch, generic_batch
from fsrl.experiments.minimal_learner.protocol import task_generator
from fsrl.experiments.structural_identification.model import (
    encode,
    make_model,
    reference,
)
from fsrl.experiments.structural_identification.observation import (
    observe,
    replicated_cycles,
    sample_choices,
)
from fsrl.experiments.structural_identification.protocol import (
    SCHEDULES,
    STRUCTURES,
    fit_order,
    model_specification,
)
from fsrl.experiments.training_strategy.batches import sample_episodes
from fsrl.experiments.training_strategy.behavior import human_references
from fsrl.tasks.protocol import ordered_pairs
from fsrl.tasks.protocol_catalog import load_registered_protocol


class StructuralContracts(unittest.TestCase):
    def test_order_manipulation_preserves_each_relation_occurrence(self):
        from fsrl.experiments.structural_identification.inputs import (
            manipulation_batch,
            schedule_variants,
        )

        rng = np.random.default_rng(827)
        batch = manipulation_batch(rng, 3)
        uniforms = rng.random(batch.arrays["signed"].shape)
        variants = schedule_variants(batch, uniforms, 828)
        base, base_uniforms = variants["balanced_K4"]
        for name in ("clustered_K4", "reverse_clustered_K4"):
            transformed, transformed_uniforms = variants[name]
            for subject in range(3):
                original_pairs = base.arrays["support_pairs"][subject]
                new_pairs = transformed.arrays["support_pairs"][subject]
                for pair in np.unique(np.sort(original_pairs, axis=1), axis=0):
                    a = (np.sort(original_pairs, axis=1) == pair).all(axis=1)
                    b = (np.sort(new_pairs, axis=1) == pair).all(axis=1)
                    np.testing.assert_array_equal(
                        base_uniforms[a, subject], transformed_uniforms[b, subject]
                    )
                    for key in ("support_cues", "signed", "retention"):
                        np.testing.assert_array_equal(
                            base.arrays[key][a, subject],
                            transformed.arrays[key][b, subject],
                        )

    def test_query_labels_are_not_model_inputs_and_reads_do_not_write(self):
        rng = np.random.default_rng(829)
        batch = generic_batch(sample_episodes(task_generator(), rng, 2))
        encoded = encode(batch, "M11", rng.random(batch.arrays["signed"].shape))
        model = make_model("decay").double()
        first = model(*model_tensors(encoded, "cpu", torch.float64))
        relabeled = ModelBatch(
            {**encoded.arrays, "targets": 1 - encoded.arrays["targets"]}
        )
        second = model(*model_tensors(relabeled, "cpu", torch.float64))
        for a, b in zip(first, second, strict=True):
            torch.testing.assert_close(a, b, atol=0, rtol=0)

    def test_recovery_integrates_shared_parameters_after_episodes(self):
        from fsrl.experiments.structural_identification.recovery import recovery_summary

        settings = {"eta_grid": [0.2, 0.8], "gain_grid": [1], "datasets_per_setting": 1}
        scores = np.full((2, 2, 8, 4, 2), -100.0)
        # Family zero can only explain each episode with a different shared eta.
        scores[0, :, :, 0] = [0, -100]
        scores[1, :, :, 0] = [-100, 0]
        scores[:, :, :, 1] = -2
        result = recovery_summary(scores, settings)
        self.assertTrue(all(row[1] == 2 for row in result["confusion_counts"][0]))

    def test_all_structures_reference_and_no_admission(self):
        rng = np.random.default_rng(821)
        batch = generic_batch(sample_episodes(task_generator(), rng, 3))
        uniforms = rng.random(batch.arrays["signed"].shape)
        physical = batch.fingerprint()
        for structure in STRUCTURES:
            encoded = encode(batch, structure, uniforms)
            self.assertEqual(batch.fingerprint(), physical)
            if structure[1] == "0":
                np.testing.assert_array_equal(encoded.arrays["retention"], 1)
            for schedule in SCHEDULES:
                with self.subTest(structure=structure, schedule=schedule):
                    model = make_model(schedule).double()
                    eta, gain = model.eta.item(), model.global_gain.item()
                    expected, state = reference(encoded, eta, gain, schedule)
                    observed, weights, efficacy = model(
                        *model_tensors(encoded, "cpu", torch.float64)
                    )
                    np.testing.assert_allclose(
                        observed.detach(), expected, atol=1e-12, rtol=1e-10
                    )
                    np.testing.assert_allclose(
                        weights.detach(), state, atol=1e-12, rtol=1e-10
                    )
                    self.assertEqual(efficacy.numel(), 0)
                    zero = ModelBatch(
                        {
                            **encoded.arrays,
                            "retention": np.zeros_like(encoded.arrays["retention"]),
                        }
                    )
                    self.assertEqual(
                        torch.count_nonzero(
                            model(*model_tensors(zero, "cpu", torch.float64))[0]
                        ),
                        0,
                    )

    def test_legacy_sampling_and_observation_parity(self):
        protocol = load_registered_protocol("liu_v2")
        rng = np.random.default_rng(822)
        # Arbitrary noncandidate logits exercise ties, cycles and exclusions.
        margins = rng.normal(0, 0.7, (25, 56))
        pairs = ordered_pairs(8)
        lookup = {pair: i for i, pair in enumerate(pairs)}
        for a, b in pairs:
            if a < b:
                margins[:, lookup[(b, a)]] = -margins[:, lookup[(a, b)]]
        choices, canonical = sample_choices(margins, protocol, 823)
        new = observe(choices, protocol, legacy_margins=canonical)
        old = analyze_sampled_query_policy(
            protocol,
            tuple(dict(zip(pairs, values, strict=True)) for values in margins),
            seed=823,
            temperature=0.25,
        )
        for a, b in zip(new["subjects"], old["subjects"], strict=True):
            for key in (
                "ranking_class",
                "majority_ties",
                "circular_triads",
                "subjective_order_high_to_low",
                "stable_error_pair_counts",
            ):
                self.assertEqual(a[key], b[key])
        refs = human_references(model_specification())
        new_record, old_record = cohort_record(new, refs), cohort_record(old, refs)
        self.assertEqual(new_record["flags"], old_record["flags"])
        for key, value in new_record["values"].items():
            self.assertAlmostEqual(value, old_record["values"][key], places=12)

    def test_common_ties_cannot_use_latent_margins(self):
        protocol = load_registered_protocol("liu_v2")
        choices = np.zeros((4, 10, 28), dtype=bool)
        choices[:, :5] = True
        common = observe(choices, protocol)
        self.assertTrue(
            all(row["ranking_class"] != "correct" for row in common["subjects"])
        )
        self.assertTrue(all(row["majority_ties"] == 28 for row in common["subjects"]))

    def test_cycle_requires_same_direction_in_both_halves(self):
        from itertools import combinations

        index = {pair: i for i, pair in enumerate(combinations(range(8), 2))}
        choices = np.ones((1, 10, 28), dtype=bool)
        choices[:, :, index[(0, 2)]] = False
        self.assertGreater(replicated_cycles(choices)[0], 0)
        choices[:, 5:, index[(0, 2)]] = True
        self.assertEqual(replicated_cycles(choices)[0], 0)

    def test_complete_fixed_rate_followup_is_prespecified(self):
        order = list(fit_order())
        self.assertEqual(len(order), 24)
        self.assertTrue(all(row[2] == "decay" for row in order[:12]))
        self.assertTrue(all(row[2] == "fixed" for row in order[12:]))


if __name__ == "__main__":
    unittest.main()
