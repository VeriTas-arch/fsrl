import unittest
from itertools import combinations

import numpy as np

from fsrl.experiments.transport.sparsity_individualization import (
    ols_slope,
    outcome,
    pairwise_tau_matrix,
    weighted_pairwise_tau,
)


class SparsityIndividualizationLocalizationTests(unittest.TestCase):
    def test_ols_slope_is_paired_across_density(self):
        values = np.asarray(
            [
                [0.4, 0.8],
                [0.3, 0.6],
                [0.2, 0.4],
                [0.1, 0.2],
            ]
        )
        np.testing.assert_allclose(
            ols_slope(values, np.asarray([7, 8, 9, 10])), [-0.1, -0.2]
        )

    def test_weighted_pairwise_tau_matches_explicit_resample(self):
        positions = np.asarray(
            [
                [0, 1, 2, 3, 4, 5, 6, 7],
                [0, 1, 2, 3, 4, 5, 7, 6],
                [7, 6, 5, 4, 3, 2, 1, 0],
            ]
        )
        matrix = pairwise_tau_matrix(positions)
        counts = np.asarray([[1.0, 1.0, 1.0], [2.0, 1.0, 0.0]])
        observed = weighted_pairwise_tau(matrix, counts)
        explicit = []
        for count in counts.astype(int):
            indices = np.repeat(np.arange(3), count)
            values = [
                matrix[first, second] for first, second in combinations(range(3), 2)
            ]
            explicit.append(
                np.mean(
                    [
                        matrix[indices[first], indices[second]]
                        for first, second in combinations(range(3), 2)
                    ]
                )
            )
        self.assertEqual(len(values), 3)
        np.testing.assert_allclose(observed, explicit)

    def test_outcome_never_pools_partial_replication(self):
        def row(stable, tau):
            return {
                "flags": {
                    "stable_error_incidence_decreases": stable,
                    "all_participant_pairwise_tau_increases": tau,
                }
            }

        self.assertEqual(
            outcome([row(True, True) for _ in range(6)]),
            "DENSITY_LINKED_INDIVIDUALIZATION_CONVERGENCE",
        )
        self.assertEqual(
            outcome([row(False, True) for _ in range(6)]),
            "ORDER_CONVERGENCE_WITHOUT_REPLICATED_STABLE_ERROR_LOSS",
        )
        mixed = [row(False, False) for _ in range(6)]
        mixed[0] = row(True, True)
        self.assertEqual(
            outcome(mixed), "FAMILY_OR_BACKBONE_SPECIFIC_INDIVIDUALIZATION_CHANGE"
        )
        self.assertEqual(
            outcome([row(False, False) for _ in range(6)]),
            "NO_REPLICATED_DENSITY_LOCALIZATION",
        )


if __name__ == "__main__":
    unittest.main()
