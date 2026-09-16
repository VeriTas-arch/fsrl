import unittest

import numpy as np

from fsrl.analysis.statistics import bootstrap_counts
from fsrl.experiments.local_fidelity.evidence_access_pilot import (
    _paired as historical_paired,
)
from fsrl.experiments.local_fidelity.evidence_access_pilot import (
    _probability_metrics as historical_probability_metrics,
)
from fsrl.experiments.pl_direct_training.functional_replication.estimands import (
    cohort_outcome,
    paired_summary,
    probability_metrics,
    seed_outcome,
)
from fsrl.experiments.pl_direct_training.functional_replication.evaluation import (
    probability_endpoint_parity_error,
)


class FunctionalEstimandTests(unittest.TestCase):
    def test_endpoint_parity_restores_serialized_empty_groups(self):
        probabilities = {
            "dual_access": {
                "raw_subject_level": {
                    "retained": [None, 0.75, 0.60],
                    "omitted": [0.55, None, 0.70],
                }
            }
        }
        endpoints = {
            "dual_access": {
                "probability": {
                    "retained": np.asarray([np.nan, 0.75, 0.60]),
                    "omitted": np.asarray([0.55, np.nan, 0.70]),
                }
            }
        }
        self.assertEqual(
            probability_endpoint_parity_error(probabilities, endpoints), 0.0
        )
        endpoints["dual_access"]["probability"]["retained"][0] = 0.5
        self.assertTrue(
            np.isinf(probability_endpoint_parity_error(probabilities, endpoints))
        )

    def test_historical_probability_and_pairing_estimands_are_exact(self):
        rng = np.random.default_rng(73)
        probabilities = rng.uniform(0.05, 0.95, size=(7, 8, 2))
        retention = np.asarray(
            [
                [(subject + relation) % 3 != 0 for relation in range(8)]
                for subject in range(7)
            ]
        )
        counts = bootstrap_counts(rng, 200, 7)
        self.assertEqual(
            probability_metrics(probabilities, retention, counts, 0.95),
            historical_probability_metrics(probabilities, retention.T, counts, 0.95),
        )
        first = rng.normal(size=7)
        second = rng.normal(size=7)
        self.assertEqual(
            paired_summary(first, second, counts, 0.95),
            historical_paired(first, second, counts, 0.95),
        )

    def test_outcome_tree_never_uses_majority_vote(self):
        links = {name: True for name in ("a", "b", "c", "d")}
        passed = seed_outcome(
            integrity=True,
            competence=True,
            global_path=True,
            four_links=links,
            omitted_materiality=True,
            qualitative_behavior=True,
        )
        behavior_incomplete = seed_outcome(
            integrity=True,
            competence=True,
            global_path=True,
            four_links=links,
            omitted_materiality=True,
            qualitative_behavior=False,
        )
        alternative = seed_outcome(
            integrity=True,
            competence=True,
            global_path=True,
            four_links={**links, "d": False},
            omitted_materiality=True,
            qualitative_behavior=True,
        )
        self.assertEqual(passed, "clean_no_time_pl_functional_replication")
        self.assertEqual(
            behavior_incomplete, "mechanism_replication_behavior_incomplete"
        )
        self.assertEqual(alternative, "competent_alternative_organization")
        self.assertEqual(
            cohort_outcome(
                {
                    "3004": {"outcome": passed},
                    "3005": {"outcome": passed},
                    "3006": {"outcome": alternative},
                }
            ),
            "competent_alternative_organization",
        )


if __name__ == "__main__":
    unittest.main()
