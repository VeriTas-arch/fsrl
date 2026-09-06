"""Independent routing, local-evidence and participant-bootstrap checks."""

import copy
import unittest
from itertools import combinations, combinations_with_replacement

import numpy as np
import torch

from fsrl.analysis.behavioral import kendall_tau_positions
from fsrl.analysis.hodge import build_complete_graph_geometry
from fsrl.core.sequence import RecurrentSequence
from fsrl.experiments.evidence_routing.inputs import check_pair, route_batch
from fsrl.experiments.evidence_routing.interventions import shuffle_evidence
from fsrl.experiments.evidence_routing.measurement import (
    internal_preferences,
    paired_tau,
    tau_matrix,
    weighted_tau,
)
from fsrl.experiments.evidence_routing.protocol import specification
from fsrl.experiments.memory_structure.inputs import (
    generator,
    liu_inputs,
    prepare_shared,
)
from fsrl.experiments.memory_structure.model import forward_batch, make_model, objective
from fsrl.experiments.training_strategy.batches import sample_episodes


def fixture():
    spec = copy.deepcopy(specification())
    spec["architecture"]["hidden_size"] = 8
    spec["optimization"]["batch_size"] = 3
    episodes = sample_episodes(generator(spec), np.random.default_rng(910003), 3)
    return spec, prepare_shared(episodes)


class RoutingTests(unittest.TestCase):
    def test_routing_changes_only_omitted_global_input(self):
        _, base = fixture()
        self.assertGreater(check_pair(base)["weak_entries"], 0)
        original = base.fingerprint()
        isolated = route_batch(base, "isolated")
        signed, z, p = (
            base.arrays[k] for k in ("signed_magnitudes", "retention", "probabilities")
        )
        np.testing.assert_allclose(
            isolated.arrays["support_inputs"][:, 0, :, 37], signed * z, rtol=1e-6
        )
        np.testing.assert_allclose(
            isolated.arrays["local_evidence"], signed * (z + (1 - z) * p), rtol=1e-6
        )
        self.assertEqual(original, base.fingerprint())

    def test_local_state_stays_identical_while_global_state_changes(self):
        spec, base = fixture()
        results = []
        for arm in ("shared", "isolated"):
            net, local = make_model(spec, 910003, "dual", "cpu")
            batch = route_batch(base, arm).to("cpu")
            result = forward_batch(net, local, RecurrentSequence(net), batch)
            loss, _ = objective(result, batch, 0)
            loss.backward()
            self.assertGreater(float(net.i2h.weight.grad[:, 37].abs().sum()), 0)
            self.assertGreater(float(local.raw_gain.grad.abs()), 0)
            results.append(result)
        torch.testing.assert_close(
            results[0].local_state, results[1].local_state, atol=0, rtol=0
        )
        self.assertFalse(torch.equal(results[0].weights, results[1].weights))

    def test_shuffling_keeps_local_weak_values_and_common_addresses(self):
        spec, _ = fixture()
        spec["evaluation"]["liu"]["subjects"] = 3
        _, base = liu_inputs(spec, 8)
        outputs = []
        for arm in ("shared", "isolated"):
            cpu = route_batch(base, arm)
            changed, route = shuffle_evidence(cpu, 4, 910003)
            for subject in range(3):
                for block in range(4):
                    source = 8 * block + route[subject, block]
                    for key in ("local_evidence",):
                        np.testing.assert_array_equal(
                            changed.arrays[key][8 * block : 8 * (block + 1), subject],
                            cpu.arrays[key][source, subject],
                        )
                    for channel in (34, 37):
                        np.testing.assert_array_equal(
                            changed.arrays["support_inputs"][
                                8 * block : 8 * (block + 1), 0, subject, channel
                            ],
                            cpu.arrays["support_inputs"][source, 0, subject, channel],
                        )
            outputs.append(changed)
        np.testing.assert_array_equal(
            outputs[0].arrays["local_evidence"], outputs[1].arrays["local_evidence"]
        )
        self.assertTrue(
            np.any(
                outputs[1].arrays["local_evidence"]
                != outputs[1].arrays["support_inputs"][:, 0, :, 37]
            )
        )


class MeasurementTests(unittest.TestCase):
    def test_weighted_tau_matches_explicit_resampled_people(self):
        orders = np.asarray([[0, 1, 2], [1, 0, 2], [2, 1, 0], [0, 2, 1]])
        matrix = tau_matrix(orders)
        for indices in combinations_with_replacement(range(4), 4):
            counts = np.bincount(indices, minlength=4)[None, :]
            for mask in (np.ones(4, dtype=bool), np.asarray([True, False, True, True])):
                selected = [i for i in indices if mask[i]]
                measured = weighted_tau(matrix, counts * mask)[0]
                if len(selected) < 2:
                    self.assertTrue(np.isnan(measured))
                else:
                    pairs = combinations(selected, 2)
                    reference = np.mean(
                        [
                            kendall_tau_positions(
                                np.argsort(orders[i]), np.argsort(orders[j])
                            )
                            for i, j in pairs
                        ]
                    )
                    self.assertAlmostEqual(measured, reference, places=14)

    def test_paired_tau_swap_and_undefined_draws(self):
        a = np.asarray([[0, 1, 2], [0, 1, 2], [1, 0, 2], [0, 1, 2]])
        b = np.asarray([[0, 1, 2], [1, 0, 2], [2, 1, 0], [0, 2, 1]])
        mask = np.ones(4, dtype=bool)
        stats = {"samples": 1000}
        ab = paired_tau(a, b, mask, mask, 7, stats)
        ba = paired_tau(b, a, mask, mask, 7, stats)
        self.assertAlmostEqual(ab["point"], -ba["point"])
        self.assertAlmostEqual(ab["interval"]["lower"], -ba["interval"]["upper"])
        missing = paired_tau(a, b, [True, False, False, False], mask, 7, stats)
        self.assertIsNone(missing["point"])
        self.assertIsNone(missing["interval"]["upper"])
        self.assertGreater(missing["undefined_draws"], 0)

    def test_internal_errors_average_probabilities_after_orientation(self):
        spec, _ = fixture()
        spec["evaluation"]["liu"]["subjects"] = 3
        protocol, _ = liu_inputs(spec, 8)
        geometry = build_complete_graph_geometry(protocol)
        p = np.full((3, 28, 2), 0.9)
        p[:, 0] = [0.8, 0.2]
        p[:, 1] = [0.1, 0.1]
        signs = geometry.true_sign[None, :, None] * np.asarray([1, -1])
        margins = (0.25 * np.log(p / (1 - p)) * signs).reshape(3, 56)
        _, raw = internal_preferences(margins, protocol, spec, 910003)
        np.testing.assert_allclose(raw["correct_probability"], p.mean(2), atol=1e-14)
        np.testing.assert_allclose(raw["stable_error_density"], 1 / 28)
        np.testing.assert_array_equal(raw["stable_error_prevalence"], 1)
