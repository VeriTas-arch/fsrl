"""Focused state, RNG, gradient and intervention tests without scientific fits."""

import copy
import unittest

import numpy as np
import torch

from fsrl.experiments.finite_state.model import FiniteSequence, rollout, sequences
from fsrl.experiments.finite_state.protocol import specification
from fsrl.experiments.finite_state.qualification import qualify
from fsrl.experiments.memory_structure.inputs import generator, prepare_shared
from fsrl.experiments.memory_structure.model import make_model, read_queries
from fsrl.experiments.training_strategy.batches import sample_episodes


class FiniteStateTests(unittest.TestCase):
    def test_real_sequence_cost_gradient_matches_conditional_expectation(self):
        class ScalarCell(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.alpha = torch.nn.Parameter(torch.ones(1, 1))
                self.delta = torch.nn.Parameter(torch.tensor(0.125))

            def forward(self, current, hidden, eligibility, weights):
                return (
                    current,
                    current,
                    current,
                    hidden,
                    eligibility,
                    (weights + self.delta).clamp(-50, 50),
                )

        cell = ScalarCell()
        p = torch.zeros(4, 1, 1)
        u = torch.tensor([0.125, 0.375, 0.625, 0.875]).reshape(1, 4, 1, 1)
        stored, cost = FiniteSequence(cell, 201)(torch.zeros(1, 4, 1), p[:, 0], p, p, u)
        self.assertEqual(float(cost.detach().mean()), 0.125)
        self.assertEqual(float(stored.detach().mean()), 0.125)
        cost.mean().backward()
        self.assertEqual(float(cell.delta.grad), 1.0)
        self.assertEqual(float(cell.alpha.grad), 0.125)

    def test_independent_grid_cost_and_continuous_update(self):
        self.assertTrue(all(row["passed"] for row in qualify(specification()).values()))

    def test_no_shadow_state_query_mutation_or_global_rng_consumption(self):
        spec = copy.deepcopy(specification())
        spec["architecture"]["hidden_size"] = 8
        backbone, local = make_model(spec, 910005, "dual", "cpu")
        cpu = prepare_shared(
            sample_episodes(generator(spec), np.random.default_rng(15), 2)
        )
        cpu.arrays["support_inputs"] = cpu.arrays["support_inputs"][:3]
        cpu.arrays["local_evidence"] = cpu.arrays["local_evidence"][:3]
        seqs, batch = sequences(backbone, 65), cpu.to("cpu")
        state = torch.get_rng_state().clone()
        first, _, _ = rollout(backbone, local, seqs, batch, 65, 33)
        second, _, _ = rollout(backbone, local, seqs, batch, 65, 33)
        self.assertTrue(torch.equal(first.weights, second.weights))
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        indices = first.weights / (100.0 / 64)
        self.assertTrue(torch.equal(indices, indices.round()))
        saved = first.weights.clone()
        a, _ = read_queries(
            backbone, local, seqs[1], batch, first.weights, first.local_state
        )
        b, _ = read_queries(
            backbone, local, seqs[1], batch, first.weights, first.local_state
        )
        self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.equal(saved, first.weights))
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertEqual(
            set(seqs[0].__dict__) - set(torch.nn.Module().__dict__), {"states"}
        )


if __name__ == "__main__":
    unittest.main()
