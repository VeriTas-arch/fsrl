"""Independent stepwise checks of input, state, gradient and intervention contracts."""

import copy
import unittest
from dataclasses import replace

import numpy as np
import torch

from fsrl.core.sequence import RecurrentSequence
from fsrl.experiments.memory_structure.inputs import (
    generator,
    liu_inputs,
    prepare_shared,
)
from fsrl.experiments.memory_structure.interventions import (
    remove_relation,
    shuffle_evidence,
)
from fsrl.experiments.memory_structure.measurement import diversity
from fsrl.experiments.memory_structure.model import (
    forward_batch,
    make_model,
    objective,
    optimizer_for,
    read_queries,
    update,
)
from fsrl.experiments.memory_structure.protocol import specification
from fsrl.experiments.training_strategy.batches import prepare_batch, sample_episodes
from fsrl.infra.provenance import tensor_hashes


def fixture():
    spec = copy.deepcopy(specification())
    spec["architecture"]["hidden_size"] = 8
    spec["optimization"]["batch_size"] = 2
    episodes = sample_episodes(generator(spec), np.random.default_rng(910002), 2)
    return spec, episodes, prepare_shared(episodes)


def direct_reference(net, local, batch):
    """Call individual cells; compute the local outer-product ledger explicitly."""
    size = batch.support_inputs.shape[2]
    weights = net.initial_fast_weights(size)
    blank = batch.support_inputs.new_zeros(2, size, net.model_config.input_size)
    for trial in [blank, *batch.support_inputs.unbind()]:
        hidden = net.initial_hidden(size)
        eligibility = net.initial_eligibility(size)
        for inputs in trial.unbind():
            _, _, _, hidden, eligibility, weights = net(
                inputs, hidden, eligibility, weights
            )
    count = batch.targets.numel() // size
    query_weights = weights.repeat(count, 1, 1)
    hidden = net.initial_hidden(count * size)
    eligibility = net.initial_eligibility(count * size)
    for inputs in batch.query_inputs.unbind():
        logits, _, _, hidden, eligibility, _ = net(
            inputs, hidden, eligibility, query_weights
        )
    if local is not None:
        cue_size = local.cue_size
        state = torch.zeros(size, cue_size * cue_size)
        for trial, evidence in zip(
            batch.support_inputs, batch.local_evidence, strict=True
        ):
            left, right = trial[0, :, :cue_size], trial[0, :, cue_size : 2 * cue_size]
            key = (
                left[:, :, None] * right[:, None, :]
                - right[:, :, None] * left[:, None, :]
            ).flatten(1)
            state = state + evidence[:, None] * key / key.norm(
                dim=1, keepdim=True
            ).clamp_min(1e-8)
        left = batch.query_inputs[0, :, :cue_size]
        right = batch.query_inputs[0, :, cue_size : 2 * cue_size]
        key = (
            left[:, :, None] * right[:, None, :] - right[:, :, None] * left[:, None, :]
        ).flatten(1)
        key = key / key.norm(dim=1, keepdim=True).clamp_min(1e-8)
        correction = local.gain * (state.repeat(count, 1) * key).sum(1)
        logits = logits + torch.stack((-correction / 2, correction / 2), dim=1)
    return weights, logits


class ComputationTests(unittest.TestCase):
    def test_effective_inputs_preserve_historical_channels(self):
        _, episodes, cpu = fixture()
        original = prepare_batch(episodes).arrays
        np.testing.assert_array_equal(
            cpu.arrays["support_inputs"][..., :37], original["support_inputs"]
        )
        np.testing.assert_array_equal(
            cpu.arrays["support_inputs"][:, 0, :, 37], original["local_evidence"]
        )
        self.assertFalse(cpu.arrays["query_inputs"][..., 37].any())
        self.assertFalse(cpu.arrays["support_inputs"][:, 1:, :, 37].any())

    def test_independent_rollout_and_learning_path(self):
        spec, _, cpu = fixture()
        batch = cpu.to("cpu")
        for condition in ("dual", "single"):
            with self.subTest(condition=condition):
                net, local = make_model(spec, 910002, condition, "cpu")
                result = forward_batch(net, local, RecurrentSequence(net), batch)
                weights, logits = direct_reference(net, local, batch)
                torch.testing.assert_close(
                    result.weights, weights, atol=1e-6, rtol=1e-5
                )
                torch.testing.assert_close(result.logits, logits, atol=1e-6, rtol=1e-5)
                loss, _ = objective(result, batch, 0)
                gradient = torch.autograd.grad(
                    loss, result.first_write, retain_graph=True
                )[0]
                self.assertGreater(float(gradient.abs().sum()), 0)
                loss.backward()
                self.assertGreater(float(net.i2h.weight.grad[:, 37].abs().sum()), 0)
                if condition == "single":
                    self.assertIsNone(local)
                    self.assertIsNone(result.local_state)

    def test_paired_initialization_and_joint_parameter_updates(self):
        spec, _, cpu = fixture()
        dual, local = make_model(spec, 910002, "dual", "cpu")
        single, absent = make_model(spec, 910002, "single", "cpu")
        self.assertIsNone(absent)
        self.assertEqual(tensor_hashes(dual), tensor_hashes(single))
        before = tensor_hashes(dual)
        gain = local.raw_gain.detach().clone()
        update(
            dual,
            local,
            RecurrentSequence(dual),
            cpu.to("cpu"),
            optimizer_for(dual, local, spec),
            spec,
        )
        self.assertNotEqual(before["i2h.weight"], tensor_hashes(dual)["i2h.weight"])
        self.assertFalse(torch.equal(gain, local.raw_gain))

    def test_query_order_and_local_off_identity(self):
        spec, _, cpu = fixture()
        net, local = make_model(spec, 910002, "dual", "cpu")
        batch = cpu.to("cpu")
        sequence = RecurrentSequence(net)
        result = forward_batch(net, local, sequence, batch)
        before = result.weights.clone()
        zero, _ = read_queries(
            net,
            local,
            sequence,
            batch,
            result.weights,
            torch.zeros_like(result.local_state),
        )
        torch.testing.assert_close(zero, result.global_logits, atol=0, rtol=0)
        count = batch.targets.numel() // 2
        batch = replace(
            batch,
            query_inputs=batch.query_inputs.reshape(2, count, 2, -1)
            .flip(1)
            .reshape(2, count * 2, -1),
        )
        logits, _ = read_queries(
            net, local, sequence, batch, result.weights, result.local_state
        )
        torch.testing.assert_close(
            logits.reshape(count, 2, 2).flip(0).reshape(-1, 2), result.logits
        )
        torch.testing.assert_close(result.weights, before, atol=0, rtol=0)

    def test_source_removal_and_joint_routing(self):
        spec, _, _ = fixture()
        spec["evaluation"]["liu"]["subjects"] = 3
        protocol, cpu = liu_inputs(spec, 8)
        removed = remove_relation(cpu, protocol.support_pairs_higher_lower[0])
        mask = np.all(
            np.sort(cpu.arrays["support_pairs"], axis=-1)
            == sorted(protocol.support_pairs_higher_lower[0]),
            axis=-1,
        )
        self.assertEqual(int(mask.sum()), 12)
        self.assertFalse(removed.arrays["local_evidence"][mask].any())
        for channel in (34, 37):
            self.assertFalse(
                removed.arrays["support_inputs"][:, 0, :, channel][mask].any()
            )
        shuffled, route = shuffle_evidence(cpu, 4, 910002)
        untouched = list(range(34)) + [35, 36]
        np.testing.assert_array_equal(
            shuffled.arrays["support_inputs"][..., untouched],
            cpu.arrays["support_inputs"][..., untouched],
        )
        for subject in range(3):
            for block in range(4):
                indices = block * 8 + route[subject, block]
                np.testing.assert_array_equal(
                    shuffled.arrays["support_inputs"][
                        block * 8 : (block + 1) * 8, 0, subject
                    ][:, [34, 37]],
                    cpu.arrays["support_inputs"][indices, 0, subject][:, [34, 37]],
                )

    def test_variable_item_inputs_and_diversity(self):
        spec, _, _ = fixture()
        spec["evaluation"]["liu"]["subjects"] = 3
        for size in (6, 8, 10):
            protocol, cpu = liu_inputs(spec, size)
            self.assertEqual(cpu.arrays["support_inputs"].shape, (4 * size, 4, 3, 38))
            self.assertEqual(len(cpu.arrays["targets"]), size * (size - 1) * 3)
            self.assertEqual(protocol.n_items, size)
            row = diversity([list(range(size)), list(reversed(range(size)))], 1, 100)
            self.assertEqual(row["point"], -1)
