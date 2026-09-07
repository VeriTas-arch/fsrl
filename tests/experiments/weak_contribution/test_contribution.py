import unittest

import numpy as np

from fsrl.experiments.weak_contribution.computation import (
    input_checks,
    relation_mask,
    remove_all_weak,
)
from fsrl.experiments.weak_contribution.measurement import effects, single_endpoints
from fsrl.experiments.weak_contribution.qualification import qualify, synthetic


class WeakContributionTests(unittest.TestCase):
    def test_reconstruction_independent_local_and_global_only_removal(self):
        self.assertTrue(all(row["passed"] for row in qualify().values()))

    def test_joint_deletion_preserves_strong_local_cues_and_repeat_masks(self):
        cpu, relations = synthetic()
        changed = remove_all_weak(cpu)
        checks = input_checks(cpu, changed, relations)
        self.assertEqual(checks["zero_weak_subjects"], [0])
        self.assertEqual(checks["weak_relation_counts"], [0, 2, 1])
        np.testing.assert_array_equal(
            cpu.arrays["support_inputs"][:, :, 0],
            changed.arrays["support_inputs"][:, :, 0],
        )
        for relation in relations:
            np.testing.assert_array_equal(
                relation_mask(cpu, relation).sum(0), [4, 4, 4]
            )

    def test_weak_relations_are_averaged_within_subject_before_population(self):
        cpu, relations = synthetic()
        intact = np.zeros((3, 12))
        removed = np.broadcast_to(
            np.asarray([2.0, 5.0, 8.0])[:, None, None], (3, 3, 12)
        ).copy()
        result = single_endpoints(
            {"intact_global": intact, "single_global": removed}, cpu, relations
        )
        for group in ("direct", "remote", "nonlearned"):
            values = result[f"single_{group}_absolute_margin"]
            self.assertTrue(np.isnan(values[0]))
            np.testing.assert_array_equal(values[1:], [5.0, 5.0])

    def test_ce_and_probability_separate_beneficial_from_harmful_influence(self):
        signs = np.asarray([[1.0, -1.0]])
        intact = signs * 0.5
        removed = np.zeros((1, 2))
        useful = effects(intact, removed, signs)
        harmful = effects(-intact, removed, signs)
        self.assertTrue(np.all(useful["ce_benefit"] > 0))
        self.assertTrue(np.all(harmful["ce_benefit"] < 0))
        self.assertTrue(np.all(useful["probability_benefit"] > 0))
        self.assertTrue(np.all(harmful["probability_benefit"] < 0))
        np.testing.assert_array_equal(
            useful["absolute_margin"], harmful["absolute_margin"]
        )
