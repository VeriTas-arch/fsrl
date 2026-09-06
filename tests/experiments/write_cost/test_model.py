import copy
import unittest

import numpy as np
import torch
from torch import nn

from fsrl.experiments.memory_structure.inputs import generator
from fsrl.experiments.memory_structure.model import make_model
from fsrl.experiments.training_strategy.batches import sample_episodes
from fsrl.experiments.write_cost.diagnostic import joined
from fsrl.experiments.write_cost.inputs import history_pair, internal_bridge
from fsrl.experiments.write_cost.model import WriteSequence, rollout, sequences
from fsrl.experiments.write_cost.protocol import specification
from fsrl.experiments.write_cost.qualification import qualify


class PrescribedCell(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, 1))

    def forward(self, x, h, e, p):
        return x, x, x, h, e, (p + x[:, None, :]).clamp(-50, 50)


class WriteCostTests(unittest.TestCase):
    def test_lambda_zero_and_independent_equations(self):
        spec = copy.deepcopy(specification())
        spec["architecture"]["hidden_size"] = 8
        self.assertTrue(all(row["passed"] for row in qualify(spec).values()))

    def test_cancellation_is_not_free(self):
        cell = PrescribedCell()
        inputs = torch.tensor([[[2.0]], [[-2.0]]])
        p, cost = WriteSequence(cell)(
            inputs, torch.zeros(1, 1), torch.zeros(1, 1, 1), torch.zeros(1, 1, 1)
        )
        self.assertEqual(float(p.item()), 0.0)
        self.assertEqual(float(cost.item()), 4.0)
        cost.sum().backward()
        self.assertEqual(float(cell.alpha.grad.item()), 4.0)

    def test_only_actual_post_clipping_change_is_charged(self):
        p, cost = WriteSequence(PrescribedCell())(
            torch.tensor([[[10.0]]]),
            torch.zeros(1, 1),
            torch.zeros(1, 1, 1),
            torch.full((1, 1, 1), 49.0),
        )
        self.assertEqual(float(p.item()), 50.0)
        self.assertEqual(float(cost.item()), 1.0)

    def test_matched_history_preserves_target_suffix_and_clock(self):
        rng = np.random.default_rng(1423)
        task = generator(specification())
        pair = None
        while pair is None:
            pair = history_pair(sample_episodes(task, rng, 1, validation=True)[0], rng)
        novel, redundant = (row.arrays for row in pair)
        target = int(novel["target_index"])
        np.testing.assert_array_equal(
            novel["support_inputs"][target:], redundant["support_inputs"][target:]
        )
        np.testing.assert_array_equal(
            novel["support_inputs"][..., 32], redundant["support_inputs"][..., 32]
        )
        np.testing.assert_array_equal(novel["query_inputs"], redundant["query_inputs"])
        relation = set(novel["support_pairs"][target, 0])
        self.assertFalse(
            any(set(p) == relation for p in novel["support_pairs"][:target, 0])
        )
        self.assertTrue(
            any(set(p) == relation for p in redundant["support_pairs"][:target, 0])
        )
        self.assertEqual(int(novel["retention"][target, 0]), 1)
        prior = redundant["support_pairs"][:target, 0]
        index = next(i for i, pair in enumerate(prior) if set(pair) == relation)
        self.assertTrue(internal_bridge(prior, index))

    def test_joined_queries_and_cost_preserve_individual_rollouts(self):
        spec = copy.deepcopy(specification())
        spec["architecture"]["hidden_size"] = 8
        spec["task"]["max_edges"] = 7
        rng = np.random.default_rng(725)
        task = generator(spec)
        rows = []
        while len(rows) < 2:
            pair = history_pair(sample_episodes(task, rng, 1, validation=True)[0], rng)
            if pair is not None:
                rows.append(pair[0])
        net, local = make_model(spec, 113, "dual", "cpu")
        support, query = sequences(net)
        with torch.no_grad():
            batched, cost, _ = rollout(
                net, local, support, query, joined([r.arrays for r in rows]).to("cpu")
            )
            for index, row in enumerate(rows):
                single, single_cost, _ = rollout(
                    net, local, support, query, row.to("cpu")
                )
                torch.testing.assert_close(
                    batched.logits.reshape(-1, 2, 2)[:, index], single.logits
                )
                torch.testing.assert_close(cost[index], single_cost[0])
