import unittest
from types import SimpleNamespace

import numpy as np

from fsrl.experiments.local_fidelity.evidence_access_pilot import (
    access_factor,
    apply_blockwise_route,
    blockwise_derangements,
    cross_seed_decision,
    learned_probabilities,
)
from fsrl.infra.provenance import load_json
from fsrl.infra.study_registry import resolve_record


class DualEvidenceAccessPilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.specification = load_json(
            resolve_record("benchmarks/dual_evidence_access_pilot_v2_4.json")
        )

    def test_access_factor_preserves_retained_and_weakens_omitted(self):
        admission = np.asarray([1.0, 0.0, 0.0])
        reliability = np.asarray([0.2, 0.3, 0.8])
        observed = access_factor(admission, reliability)
        np.testing.assert_array_equal(observed, np.asarray([1.0, 0.3, 0.8]))

    def test_access_factor_rejects_nonbinary_global_admission(self):
        with self.assertRaisesRegex(ValueError, "binary"):
            access_factor(np.asarray([0.5]), np.asarray([0.5]))

    def test_learned_probability_reuses_frozen_component_sum_estimand(self):
        evaluator = SimpleNamespace(
            protocol=SimpleNamespace(support_pairs_higher_lower=((0, 1),), n_items=2),
            config=SimpleNamespace(bs=1),
        )
        bundle = {
            "logits": np.asarray([[-10.0, -10.0]]),
            "global_logits": np.asarray([[0.1, 0.2]]),
            "applied_local_margins": np.asarray([[0.2, 0.3]]),
        }
        observed = learned_probabilities(evaluator, bundle, 1.0)
        expected = 1.0 / (1.0 + np.exp(-np.asarray([[[0.3, -0.5]]])))
        np.testing.assert_allclose(observed, expected, atol=0.0, rtol=1e-15)

    def test_blockwise_routes_are_derangements_and_preserve_multisets(self):
        maps = blockwise_derangements(3, 2, 4, 17)
        self.assertTrue(np.all(maps != np.arange(4)[None, None]))
        values = np.arange(24, dtype=np.float64).reshape(3, 8)
        routed = apply_blockwise_route(values, maps)
        for subject in range(3):
            for block in range(2):
                start = 4 * block
                np.testing.assert_array_equal(
                    np.sort(routed[subject, start : start + 4]),
                    np.sort(values[subject, start : start + 4]),
                )

    def test_zeroed_donor_follows_the_route(self):
        maps = np.asarray([[[1, 0, 3, 2]]])
        natural = np.asarray([[10.0, 20.0, 30.0, 40.0]])
        loo = natural.copy()
        loo[0, 1] = 0.0
        routed = apply_blockwise_route(natural, maps)
        routed_loo = apply_blockwise_route(loo, maps)
        changed = np.flatnonzero(routed[0] != routed_loo[0])
        self.assertEqual(changed.tolist(), [0])
        self.assertEqual(routed_loo[0, 0], 0.0)

    def test_cross_seed_decision_does_not_pool_heterogeneous_links(self):
        flags = {name: True for name in self.specification["primary_links"]}
        seeds = {
            "2102": {"decision": {"interpretable": True, "flags": flags}},
            "2103": {
                "decision": {
                    "interpretable": True,
                    "flags": {**flags, "retained_fidelity_preservation": False},
                }
            },
        }
        result = cross_seed_decision(self.specification, seeds)
        self.assertEqual(
            result["links"]["retained_fidelity_preservation"]["status"],
            "heterogeneous_or_unresolved",
        )
        self.assertEqual(result["outcome"], "heterogeneous_or_unresolved")
        self.assertEqual(result["network_population_inference"], "not_performed")
