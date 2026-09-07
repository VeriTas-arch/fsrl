"""Brute-force independent-panel bootstrap and prospective RNG boundaries."""

import itertools
import unittest

import numpy as np

from fsrl.analysis.statistics import bootstrap_counts
from fsrl.experiments.observation_replication.protocol import recipe, specification
from fsrl.experiments.observation_replication.statistics import panel_draws, panel_mean
from fsrl.experiments.observation_uncertainty.inputs import observation_seed
from tests.experiments.observation_crossover.test_estimands import brute_tau


class ReplicationTests(unittest.TestCase):
    def test_panel_mean_matches_independent_brute_force_not_pooled_tau(self):
        spec = specification()
        spec["statistics"]["samples"] = 40
        rng = np.random.default_rng(91)
        panels, expected_points, expected_draws = [], [], []
        for panel, n in enumerate((4, 5, 6)):
            raw, endpoints = {}, {}
            for cell in spec["design"]["cells"]:
                orders = np.array([rng.permutation(4) for _ in range(n)])
                raw[cell] = {"liu": {"routes__global__internal__orders": orders}}
                endpoints[cell] = {
                    "generic_global": rng.uniform(size=n + 2),
                    "liu_global": rng.uniform(size=n),
                }
            shifted = {"A": rng.uniform(size=n), "C": rng.uniform(size=n)}
            shifted["C_minus_A"] = shifted["C"] - shifted["A"]
            panels.append(panel_draws(raw, endpoints, shifted, 70 + panel, spec))
            coefficients = spec["estimands"]["contrasts"]["interaction"]
            counts = bootstrap_counts(np.random.default_rng(70 + panel), 40, n).astype(
                int
            )
            expected_points.append(
                sum(
                    c
                    * brute_tau(
                        raw[cell]["liu"]["routes__global__internal__orders"], range(n)
                    )
                    for cell, c in coefficients.items()
                )
            )
            expected_draws.append(
                [
                    sum(
                        c
                        * brute_tau(
                            raw[cell]["liu"]["routes__global__internal__orders"],
                            np.repeat(np.arange(n), count),
                        )
                        for cell, c in coefficients.items()
                    )
                    for count in counts
                ]
            )
            for metric in ("generic_global", "liu_global"):
                value = np.stack(
                    [c * endpoints[cell][metric] for cell, c in coefficients.items()]
                ).sum(0)
                cts = bootstrap_counts(
                    np.random.default_rng(70 + panel), 40, len(value)
                ).astype(int)
                brute = [np.mean(np.repeat(value, c)) for c in cts]
                np.testing.assert_allclose(
                    panels[-1][1]["interaction/" + metric], brute, atol=1e-14
                )
        result, draws = panel_mean(panels)
        key = "interaction/global_all77_tau"
        self.assertAlmostEqual(result[key]["point"], np.mean(expected_points))
        np.testing.assert_allclose(
            draws[key], np.mean(expected_draws, axis=0), atol=1e-14
        )
        np.testing.assert_allclose(
            list(result[key]["interval"].values()),
            np.quantile(np.mean(expected_draws, axis=0), [0.025, 0.975]),
            atol=1e-14,
        )

    def test_new_rng_namespaces_and_only_registered_recipe_changes(self):
        spec, parent = specification(), recipe()
        seeds = set()
        for panel in spec["design"]["panels"]:
            current = recipe(panel)
            for key in (
                "architecture",
                "optimization",
                "task",
                "observation",
                "statistics",
                "decision_contract",
            ):
                self.assertEqual(current[key], parent[key])
            row = [current["evaluation"]["generic"]["rng_seed"]]
            row += [current["evaluation"]["liu"][k] for k in spec["rng"]["liu_offsets"]]
            row += [observation_seed(100 + panel, batch_id=k) for k in (28, 32, 36, 40)]
            row += [observation_seed(200 + panel)]
            self.assertFalse(seeds.intersection(row))
            self.assertEqual(len(row), len(set(row)))
            seeds.update(row)
        self.assertFalse(
            set(spec["design"]["seeds"]).intersection(parent["seeds"]["mandatory"])
        )
        self.assertEqual(
            len(
                list(
                    itertools.product(
                        spec["design"]["seeds"],
                        spec["design"]["panels"],
                        spec["design"]["cells"],
                    )
                )
            ),
            36,
        )


if __name__ == "__main__":
    unittest.main()
