import copy
import unittest
from dataclasses import replace

import numpy as np
import torch
import torch.nn.functional as F

from fsrl.core.local_trace import ConjunctiveLocalTrace
from fsrl.core.relational_system import GlobalLocalRelationalSystem
from fsrl.core.sequence import RecurrentSequence
from fsrl.experiments.training_strategy.batches import (
    graph_bucket,
    prepare_batch,
    sample_episodes,
)
from fsrl.experiments.training_strategy.optimization import (
    forward_batch,
    make_optimizer,
    query_from_state,
    training_step,
)
from fsrl.experiments.training_strategy.protocol import (
    load_specification,
    phase_for_step,
    training_config,
)
from fsrl.infra.provenance import tensor_hashes
from fsrl.training.backbone import (
    build_meta_input_sequence,
    make_model_and_tasks,
    registered_excluded_signatures,
)


class PairedTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(910002)
        self.specification = load_specification()
        config = replace(
            training_config(self.specification, 910002), hidden_size=8, batch_size=2
        )
        self.config, self.backbone, self.tasks = make_model_and_tasks(
            config, device="cpu"
        )
        self.local = ConjunctiveLocalTrace(config.cue_size, device="cpu")
        self.sequence = RecurrentSequence(self.backbone)
        self.episodes = sample_episodes(self.tasks, np.random.default_rng(12345), 2)
        self.cpu_batch = prepare_batch(self.episodes)
        self.batch = self.cpu_batch.to("cpu")
        self.optimization = self.specification["optimization"]

    def test_frozen_seed_budget_and_update_schedule(self):
        self.assertEqual(self.specification["seeds"]["mandatory"], [2108, 2109, 2110])
        self.assertEqual(
            self.optimization["batch_size"] * self.optimization["total_steps"], 48000
        )
        staged = [
            phase_for_step(self.specification, "matched_staged", s) for s in range(1500)
        ]
        joint = [phase_for_step(self.specification, "joint", s) for s in range(1500)]
        self.assertEqual(staged, ["global"] * 1000 + ["local"] * 500)
        self.assertEqual(joint, ["joint"] * 1500)
        for condition, step in (("historical", 0), ("joint", -1), ("joint", 1500)):
            with self.assertRaises(ValueError):
                phase_for_step(self.specification, condition, step)

    def test_validation_split_protects_graphs_and_their_reflections(self):
        protected = registered_excluded_signatures()
        for validation in (False, True):
            episodes = sample_episodes(
                self.tasks, np.random.default_rng(4455), 8, validation=validation
            )
            for episode in episodes:
                graph = episode.graph_rank_pairs
                reflected = tuple(sorted((7 - j, 7 - i) for i, j in graph))
                self.assertEqual(graph_bucket(graph), graph_bucket(reflected))
                self.assertEqual(graph_bucket(graph) == 0, validation)
                self.assertNotIn(graph, protected)

    def test_paired_stream_fingerprints_include_probabilities_and_targets(self):
        first = np.random.default_rng(4455)
        second = np.random.default_rng(4455)
        fingerprints = []
        for _ in range(3):
            a = prepare_batch(sample_episodes(self.tasks, first, 2))
            b = prepare_batch(sample_episodes(self.tasks, second, 2))
            self.assertEqual(a.fingerprint(), b.fingerprint())
            fingerprints.append(a.fingerprint())
        self.assertEqual(len(set(fingerprints)), 3)
        altered = copy.deepcopy(self.cpu_batch)
        altered.arrays["probabilities"][0, 0] += 0.01
        self.assertNotEqual(altered.fingerprint(), self.cpu_batch.fingerprint())
        altered = copy.deepcopy(self.cpu_batch)
        altered.arrays["targets"][0] = 1 - altered.arrays["targets"][0]
        self.assertNotEqual(altered.fingerprint(), self.cpu_batch.fingerprint())

    def test_weak_local_admission_uses_original_p_not_the_binary_z(self):
        arrays = self.cpu_batch.arrays
        z = arrays["retention"]
        p = arrays["probabilities"]
        signed = arrays["signed_magnitudes"]
        local = arrays["local_evidence"]
        self.assertTrue(np.any(z == 0))
        self.assertTrue(np.any(z == 1))
        self.assertTrue(np.all((p > 0) & (p < 1)))
        np.testing.assert_allclose(local[z == 1], signed[z == 1], atol=1e-7, rtol=0)
        np.testing.assert_allclose(
            local[z == 0], (signed * p)[z == 0], atol=1e-7, rtol=0
        )
        for subject, episode in enumerate(self.episodes):
            by_relation = {}
            for trial, support in enumerate(episode.support_trials):
                key = tuple(sorted((support.left_item, support.right_item)))
                value = (z[trial, subject], p[trial, subject])
                self.assertEqual(by_relation.setdefault(key, value), value)

    def test_support_and_query_inputs_match_the_maintained_passive_interface(self):
        arrays = self.cpu_batch.arrays
        for t in range(len(self.episodes[0].support_trials)):
            pairs = arrays["support_pairs"][t]
            reference = build_meta_input_sequence(
                self.config,
                self.episodes,
                pairs[:, 0],
                pairs[:, 1],
                arrays["signed_magnitudes"][t] * arrays["retention"][t],
                num_steps=4,
                time_value=t / (len(self.episodes[0].support_trials) - 1) * (2.0 / 3.0),
                support_trial=True,
                device="cpu",
            )
            torch.testing.assert_close(
                self.batch.support_inputs[t], reference, atol=0, rtol=0
            )
        for q in range(28):
            pairs = arrays["query_pairs"][q]
            reference = build_meta_input_sequence(
                self.config,
                self.episodes,
                pairs[:, 0],
                pairs[:, 1],
                np.zeros(2),
                num_steps=2,
                time_value=2.0 / 3.0,
                support_trial=False,
                device="cpu",
            )
            torch.testing.assert_close(
                self.batch.query_inputs[:, 2 * q : 2 * q + 2], reference, atol=0, rtol=0
            )

    def forward_result(self, *, local_active=True):
        return forward_batch(
            self.backbone,
            self.local,
            self.sequence,
            self.batch,
            local_active=local_active,
            fast_weight_penalty=0.0,
        )

    def test_batched_rollout_matches_each_maintained_query_without_writeback(self):
        system = GlobalLocalRelationalSystem(self.backbone, self.local)
        state = system.initialize_episode(2)
        for t, inputs in enumerate(self.batch.support_inputs):
            state = system.support_trial(
                state,
                inputs,
                pair_cues=inputs[0, :, :30],
                local_signed_value=self.batch.local_evidence[t],
            )
        result = self.forward_result()
        torch.testing.assert_close(result.fast_weights, state.global_fast_weights)
        torch.testing.assert_close(result.local_state, state.local_trace)
        before = state.global_fast_weights.detach().clone()
        for q in range(28):
            inputs = self.batch.query_inputs[:, 2 * q : 2 * q + 2]
            query = system.query(state, inputs, pair_cues=inputs[0, :, :30])
            torch.testing.assert_close(result.logits[2 * q : 2 * q + 2], query.logits)
        torch.testing.assert_close(state.global_fast_weights, before, atol=0, rtol=0)
        global_only = self.forward_result(local_active=False)
        torch.testing.assert_close(
            global_only.logits, global_only.global_logits, atol=0, rtol=0
        )
        self.assertFalse(torch.equal(result.logits, result.global_logits))

    def test_query_ce_reaches_support_plasticity_without_a_regularizer(self):
        result = self.forward_result()
        gradients = torch.autograd.grad(
            result.query_loss,
            (
                result.fast_weights,
                result.first_support_write,
                self.backbone.h2DA.weight,
                self.local.raw_gain,
            ),
            retain_graph=True,
        )
        for gradient in gradients:
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(gradient.abs().max()), 0.0)
        detached_logits, _ = query_from_state(
            self.backbone,
            self.local,
            self.sequence,
            self.batch,
            result.fast_weights.detach(),
            result.local_state,
            local_active=True,
        )
        disconnected = torch.autograd.grad(
            F.cross_entropy(detached_logits, self.batch.targets),
            self.backbone.h2DA.weight,
            allow_unused=True,
        )[0]
        self.assertIsNone(disconnected)

    def test_combined_cross_entropy_has_the_registered_margin_sign(self):
        result = self.forward_result()
        margin = result.logits[:, 1] - result.logits[:, 0]
        target_sign = 2 * self.batch.targets - 1
        torch.testing.assert_close(
            result.query_loss, F.softplus(-target_sign * margin).mean()
        )

    def test_staged_updates_keep_rho_then_theta_exactly_frozen(self):
        optimizer = make_optimizer(self.backbone, self.local, self.optimization)
        initial_backbone = tensor_hashes(self.backbone)
        initial_local = tensor_hashes(self.local)
        for _ in range(2):
            result = training_step(
                self.backbone,
                self.local,
                self.sequence,
                self.batch,
                optimizer,
                phase="global",
                optimization=self.optimization,
            )
            torch.testing.assert_close(
                result.logits, result.global_logits, atol=0, rtol=0
            )
        frozen = tensor_hashes(self.backbone)
        self.assertNotEqual(initial_backbone, frozen)
        self.assertEqual(initial_local, tensor_hashes(self.local))
        for _ in range(2):
            training_step(
                self.backbone,
                self.local,
                self.sequence,
                self.batch,
                optimizer,
                phase="local",
                optimization=self.optimization,
            )
        self.assertEqual(frozen, tensor_hashes(self.backbone))
        self.assertNotEqual(initial_local, tensor_hashes(self.local))
        self.assertEqual(float(optimizer.state[self.backbone.w]["step"]), 2)
        self.assertEqual(float(optimizer.state[self.local.raw_gain]["step"]), 2)

    def test_joint_updates_both_parameter_groups_from_the_first_step(self):
        optimizer = make_optimizer(self.backbone, self.local, self.optimization)
        backbone_before = tensor_hashes(self.backbone)
        local_before = tensor_hashes(self.local)
        training_step(
            self.backbone,
            self.local,
            self.sequence,
            self.batch,
            optimizer,
            phase="joint",
            optimization=self.optimization,
        )
        self.assertNotEqual(backbone_before, tensor_hashes(self.backbone))
        self.assertNotEqual(local_before, tensor_hashes(self.local))
        self.assertEqual(float(optimizer.state[self.backbone.w]["step"]), 1)
        self.assertEqual(float(optimizer.state[self.local.raw_gain]["step"]), 1)
        self.assertEqual(
            [group["name"] for group in optimizer.param_groups], ["backbone", "local"]
        )
