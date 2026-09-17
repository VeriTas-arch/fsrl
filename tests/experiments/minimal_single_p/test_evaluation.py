import unittest
from itertools import combinations

import numpy as np

from fsrl.analysis.hodge import gradient_energy_fraction
from fsrl.experiments.minimal_single_p.evaluation import (
    GENERIC_GEOMETRY,
    _canonical_fields,
)


class MinimalSinglePEvaluationTests(unittest.TestCase):
    def test_canonical_orientation_and_exact_gradient_coherence(self):
        rng = np.random.default_rng(45)
        potentials = rng.normal(size=(4, 8))
        canonical_pairs = tuple(combinations(range(8), 2))
        fields = np.asarray(
            [
                [
                    potential[first] - potential[second]
                    for first, second in canonical_pairs
                ]
                for potential in potentials
            ]
        )
        pairs = np.empty((28, 4, 2), dtype=np.int64)
        margins = np.empty((4, 28))
        for query, pair in enumerate(canonical_pairs):
            for subject in range(4):
                reverse = (query + subject) % 2 == 1
                pairs[query, subject] = pair[::-1] if reverse else pair
                margins[subject, query] = (
                    -fields[subject, query] if reverse else fields[subject, query]
                )
        canonical = _canonical_fields(margins, pairs)
        np.testing.assert_array_equal(canonical, fields)
        np.testing.assert_allclose(
            gradient_energy_fraction(canonical, GENERIC_GEOMETRY), 1.0, atol=1e-12
        )


if __name__ == "__main__":
    unittest.main()
