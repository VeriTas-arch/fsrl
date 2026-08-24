import unittest

import numpy as np

from fsrl.relation_trace_localization import (
    _decision,
    prototype_identity_metrics,
    prototype_rdm_similarity,
)


class RelationTraceLocalizationTests(unittest.TestCase):
    def test_same_relation_prototypes_identify_held_out_subjects(self):
        relations = 8
        subjects = 14
        features = 10
        traces = np.zeros((relations, subjects, features), dtype=np.float64)
        for relation in range(relations):
            traces[relation, :, relation] = 1.0
        retained = np.ones((relations, subjects), dtype=bool)
        metrics, _prototypes = prototype_identity_metrics(
            traces,
            retained,
            subject_folds=7,
            tolerance=1e-12,
        )
        np.testing.assert_allclose(metrics["own_prototype_cosine"], 1.0)
        np.testing.assert_allclose(metrics["other_prototype_mean_cosine"], 0.0)
        np.testing.assert_allclose(
            metrics["eight_way_identification_accuracy"], 1.0
        )

    def test_prototype_rdm_compares_geometry_across_feature_spaces(self):
        rng = np.random.default_rng(4)
        first = rng.normal(size=(7, 8, 12))
        transform = rng.normal(size=(12, 20))
        transform, _ = np.linalg.qr(transform)
        second = first @ transform
        result = prototype_rdm_similarity(first, second)
        self.assertAlmostEqual(result["mean"], 1.0)

    def test_decision_tree_distinguishes_storage_access_and_fidelity(self):
        all_present = {
            "generated_effective_write": True,
            "terminal_effective_fast_weight": True,
            "response_full_hidden": True,
            "response_hodge_residual": True,
        }
        self.assertEqual(
            _decision(all_present),
            "storage_access_present_missing_fidelity_transformation",
        )
        no_access = dict(all_present)
        no_access["response_full_hidden"] = False
        no_access["response_hodge_residual"] = False
        self.assertEqual(
            _decision(no_access),
            "persistent_storage_present_query_access_missing",
        )


if __name__ == "__main__":
    unittest.main()
