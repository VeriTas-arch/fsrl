"""Independent pair enumeration, joint bootstrap and observation identities."""

import itertools
import unittest

import numpy as np

from fsrl.analysis.statistics import bootstrap_counts
from fsrl.experiments.observation_crossover.protocol import specification
from fsrl.experiments.observation_crossover.reporting import contrasts, order_shift
from fsrl.experiments.observation_uncertainty.inputs import encode
from fsrl.experiments.training_strategy.batches import EpisodeBatch


def brute_tau(orders, ids):
    values = []
    for a, b in itertools.combinations(ids, 2):
        first, second = orders[a].tolist(), orders[b].tolist()
        same = sum(
            (first.index(i) < first.index(j)) == (second.index(i) < second.index(j))
            for i, j in itertools.combinations(range(len(first)), 2)
        )
        values.append(2 * same / (len(first) * (len(first) - 1) / 2) - 1)
    return np.mean(values) if values else float("nan")


class CrossoverTests(unittest.TestCase):
    def test_same_population_tau_can_hide_individual_displacement(self):
        a = np.tile(np.arange(4), (3, 1))
        b = a[:, ::-1]
        self.assertEqual(brute_tau(a, range(3)), brute_tau(b, range(3)))
        np.testing.assert_array_equal(order_shift(a, b), np.ones(3))
        np.testing.assert_array_equal(order_shift(a, a), np.zeros(3))
        c = a.copy()
        c[:, :2] = c[:, 1::-1]
        np.testing.assert_allclose(order_shift(a, c), np.full(3, 1 / 6))

    def test_paired_interaction_with_duplicate_ids_and_selection(self):
        spec = specification()
        spec["statistics"] = {"samples": 80, "seed_offset": 100, "interval": 0.95}
        rng = np.random.default_rng(32)
        raw, endpoints = {}, {}
        for index, cell in enumerate(("A0", "Ae", "C0", "Ce")):
            orders = np.array([rng.permutation(4) for _ in range(5)])
            data = {"routes__full__sampled_mask": np.ones(5, dtype=bool)}
            for route in ("full", "global"):
                data[f"routes__{route}__sampled_mask"] = np.array(
                    [True, True, True, True, index != 2]
                )
                data[f"routes__{route}__sampled_orders"] = orders
                data[f"routes__{route}__internal__orders"] = orders
            raw[cell] = {"liu": data}
            endpoints[cell] = {"generic_global": rng.uniform(size=5)}
        actual = contrasts(raw, endpoints, 3, spec)
        counts = bootstrap_counts(np.random.default_rng(103), 80, 5).astype(int)
        for label, coefficient in spec["estimands"]["contrasts"].items():
            for conditional in (False, True):
                draws = []
                for weights in counts:
                    value = 0
                    for cell, c in coefficient.items():
                        data = raw[cell]["liu"]
                        ids = np.repeat(np.arange(5), weights)
                        if conditional:
                            ids = ids[data["routes__global__sampled_mask"][ids]]
                        value += c * brute_tau(
                            data["routes__global__internal__orders"], ids
                        )
                    draws.append(value)
                result = actual[label]["tau"][
                    "global_" + ("conditional" if conditional else "all77")
                ]
                if np.isfinite(draws).all():
                    np.testing.assert_allclose(
                        [result["interval"][k] for k in ("lower", "upper")],
                        np.quantile(draws, [0.025, 0.975]),
                        atol=1e-14,
                    )
                else:
                    self.assertIsNone(result["interval"]["lower"])
                    self.assertEqual(
                        result["undefined_draws"], int((~np.isfinite(draws)).sum())
                    )
            values = sum(
                c * endpoints[cell]["generic_global"] for cell, c in coefficient.items()
            )
            expected = [np.mean(np.repeat(values, count)) for count in counts]
            ce = actual[label]["CE"]["generic_global"]
            np.testing.assert_allclose(
                [ce["bootstrap"][k] for k in ("lower", "upper")],
                np.quantile(expected, [0.025, 0.975]),
                atol=1e-14,
            )

    def test_original_operator_preserves_all_archived_fields(self):
        a = {
            "local_evidence": np.array([[0.2], [-0.1]], dtype=np.float32),
            "encoding_noise_0": np.array([[1.0], [-2.0]]),
            "arbitrary_identity": np.arange(3),
        }
        cpu = EpisodeBatch(a)
        for observed in (encode(cpu, "clean", 1 / 7), encode(cpu, "noisy", 0)):
            self.assertEqual(observed.fingerprint(), cpu.fingerprint())
            for key, value in a.items():
                np.testing.assert_array_equal(observed.arrays[key], value)
        cells = specification()["design"]["cells"]
        self.assertEqual(cells["C0"]["training"], "noisy")
        self.assertEqual(cells["C0"]["observation"], "clean")


if __name__ == "__main__":
    unittest.main()
