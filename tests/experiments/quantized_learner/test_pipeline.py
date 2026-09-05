import copy
import unittest
from unittest.mock import patch

import numpy as np

from fsrl.experiments.minimal_learner.data import ModelBatch, generic_batch
from fsrl.experiments.minimal_learner.protocol import task_generator
from fsrl.experiments.quantized_learner import evaluation, training
from fsrl.experiments.quantized_learner.analysis import (
    analyze_batch,
    conditional_codes,
    query_groups,
)
from fsrl.experiments.quantized_learner.controls import (
    shuffled_teaching,
    teaching_route,
)
from fsrl.experiments.quantized_learner.decisions import recipe_decision
from fsrl.experiments.quantized_learner.encoding import (
    canonical_addresses,
    encode_batch,
)
from fsrl.experiments.quantized_learner.evidence import paired_runs, qualification_keys
from fsrl.experiments.quantized_learner.protocol import (
    make_model,
    resolved_specification,
    specification,
)
from fsrl.experiments.quantized_learner.recovery_execution import generation_shards
from fsrl.experiments.quantized_learner.verification import reconstruct_codes
from fsrl.experiments.training_strategy.batches import sample_episodes


class PipelineTests(unittest.TestCase):
    def test_generation_shards_preserve_every_array_and_design_identity(self):
        original = {
            f"{index}__value": np.array([index, index + 1], dtype=np.float64)
            for index in range(64)
        }
        shards = generation_shards(original)
        self.assertEqual(set(shards), {"generation-0", "generation-1"})
        self.assertEqual([len(row) for row in shards.values()], [32, 32])
        combined = {key: value for row in shards.values() for key, value in row.items()}
        self.assertEqual(set(combined), set(original))
        for key, value in original.items():
            np.testing.assert_array_equal(combined[key], value)

    def setUp(self):
        self.spec = resolved_specification()
        self.candidate = specification()
        self.rng = np.random.default_rng(973011)
        self.batch = generic_batch(sample_episodes(task_generator(), self.rng, 3))
        self.codebook = self.candidate["encoding"]["codebook"]
        self.uniforms = self.rng.random(self.batch.arrays["signed"].shape)

    def test_independent_tuple_code_reference_matches_every_codec(self):
        for condition in ("exact", "persistent", "resampled"):
            expected = reconstruct_codes(
                self.batch, self.uniforms, condition, self.codebook
            )
            encoded, witness = encode_batch(
                self.batch, condition, self.uniforms, self.codebook
            )
            np.testing.assert_array_equal(encoded.arrays["signed"], expected[0])
            np.testing.assert_array_equal(witness["code_indices"], expected[1])
            np.testing.assert_array_equal(witness["orientation"], expected[2])

    def test_route_preserves_block_multisets_and_reuses_donor_address(self):
        route = teaching_route(self.batch, self.rng, 4)
        keys, orientation = canonical_addresses(self.batch.arrays["support_cues"])
        retained = self.batch.arrays["retention"] == 1
        for condition in ("exact", "persistent", "resampled"):
            encoded, _ = encode_batch(
                self.batch, condition, self.uniforms, self.codebook
            )
            shifted = shuffled_teaching(encoded, route)
            for key in set(encoded.arrays) - {"signed"}:
                np.testing.assert_array_equal(encoded.arrays[key], shifted.arrays[key])
            length = len(keys) // 4
            for subject in range(3):
                donors = {}
                for block in range(4):
                    trials = np.arange(block * length, (block + 1) * length)
                    trials = trials[retained[trials, subject]]
                    a = (orientation * encoded.arrays["signed"])[trials, subject]
                    b = (orientation * shifted.arrays["signed"])[trials, subject]
                    np.testing.assert_array_equal(np.sort(a), np.sort(b))
                    for trial in trials:
                        key = keys[trial, subject]
                        donor = keys[route[trial, subject], subject]
                        self.assertEqual(donors.setdefault(key, donor), donor)

    def test_zero_admission_has_identity_route_no_cache_and_neutral_readout(self):
        batch = ModelBatch(
            {
                **self.batch.arrays,
                "retention": np.zeros_like(self.batch.arrays["retention"]),
            }
        )
        route = teaching_route(batch, self.rng, 4)
        np.testing.assert_array_equal(
            route, np.broadcast_to(np.arange(len(route))[:, None], route.shape)
        )
        encoded, witness = encode_batch(
            batch, "persistent", self.uniforms, self.codebook
        )
        np.testing.assert_array_equal(witness["cache_entries"], 0)
        model = make_model(self.spec)
        np.testing.assert_array_equal(model(*encoded.tensors("cpu"))[0].detach(), 0)

    def test_actual_analysis_interfaces_on_non_liu_fixture(self):
        model, fixed = make_model(self.spec), make_model(self.spec)
        auxiliary = {
            "encoding_uniforms": self.uniforms,
            "teaching_route": teaching_route(self.batch, self.rng, 4),
        }
        for condition in ("exact", "persistent", "resampled"):
            result = analyze_batch(
                model,
                model,
                fixed,
                fixed,
                self.batch,
                auxiliary,
                condition,
                self.codebook,
                1.0,
            )
            np.testing.assert_array_equal(
                result["outputs"]["intact"]["margins"],
                result["outputs"]["fixed_parameter"]["margins"],
            )
            np.testing.assert_array_equal(
                result["groups"]["learned"], self.batch.arrays["learned"]
            )
            self.assertEqual(result["orders"].shape, (3, 8))
            self.assertEqual(
                set(result["endpoints"]),
                {"intact", "shuffled", "z_off", "fixed_parameter"},
            )

    def test_conditional_probability_averages_after_each_state_readout(self):
        keys, _ = canonical_addresses(self.batch.arrays["support_cues"])
        z = np.zeros_like(self.batch.arrays["retention"])
        for subject in range(3):
            z[:, subject] = np.isin(keys[:, subject], np.unique(keys[:, subject])[:4])
        batch = ModelBatch({**self.batch.arrays, "retention": z})
        result = conditional_codes(
            batch,
            self.codebook,
            make_model(self.spec),
            0.25,
            2 * self.batch.arrays["targets"] - 1,
        )
        np.testing.assert_allclose(result["weights"].sum(axis=1), 1, atol=1e-12)
        self.assertTrue(np.all(result["component_counts"] <= 16))
        self.assertTrue(np.all(result["identity_errors"] < 1e-9))

    def test_all_nine_and_paired_uniform_streams_are_required(self):
        runs = {
            f"{seed}/{condition}": {
                "config": {
                    "base_stream_sha256": str(seed),
                    "uniform_stream_sha256": str(seed),
                    "initial_parameters": {"a": 1},
                }
            }
            for seed in (2114, 2115, 2116)
            for condition in ("exact", "persistent", "resampled")
        }
        paired_runs(runs)
        partial = copy.deepcopy(runs)
        partial.pop("2116/resampled")
        with self.assertRaises(RuntimeError):
            paired_runs(partial)
        runs["2115/persistent"]["config"]["uniform_stream_sha256"] = "different"
        with self.assertRaises(RuntimeError):
            paired_runs(runs)
        self.assertEqual(len(qualification_keys()), 817)

    def test_execution_guards_precede_runtime_and_any_model_work(self):
        for module, entry, guard in (
            (training, "train_all", "validate_recovery"),
            (evaluation, "evaluate_all", "validate_artifacts"),
        ):
            with (
                patch.object(module, guard, side_effect=RuntimeError("not admitted")),
                patch.object(module, "runtime") as runtime,
                self.assertRaisesRegex(RuntimeError, "not admitted"),
            ):
                getattr(module, entry)()
            runtime.assert_not_called()

    def test_partial_behavior_or_failed_binding_cannot_promote(self):
        row = {
            "decision": {
                "competence": {"learned": True, "nonlearned": True},
                "qualitative_passed": True,
                "quantitative_passed": True,
                "binding_passed": True,
            }
        }
        fits = {str(seed): copy.deepcopy(row) for seed in (2114, 2115, 2116)}
        args = ([2114, 2115, 2116], "distinguishable_on_registered_screen")
        self.assertTrue(
            recipe_decision(fits, *args)["eligible_for_unchanged_replication"]
        )
        self.assertFalse(recipe_decision(fits, *args)["main_model_promoted"])
        fits["2115"]["decision"]["quantitative_passed"] = False
        self.assertEqual(
            recipe_decision(fits, *args)["outcome"], "partial_behavioral_reproduction"
        )
        fits["2115"]["decision"]["quantitative_passed"] = True
        fits["2116"]["decision"]["binding_passed"] = False
        self.assertFalse(
            recipe_decision(fits, *args)["eligible_for_unchanged_replication"]
        )
        self.assertEqual(
            set(query_groups(self.batch)),
            {"overall", "learned", "nonlearned", "retained", "omitted"},
        )


if __name__ == "__main__":
    unittest.main()
