import unittest

import numpy as np

from fsrl.experiments.local_fidelity.hidden_residual import validate_registered_sources
from fsrl.experiments.local_fidelity.relation_trace import (
    prototype_identity_metrics,
    prototype_rdm_similarity,
    relation_trace_decision,
)
from fsrl.infra.provenance import load_json
from fsrl.infra.study_registry import registered_file_sha256, resolve_record


class RelationTraceLocalizationTests(unittest.TestCase):
    def test_corrected_execution_lock_preserves_registered_sources(self):
        specification = load_json(
            resolve_record("benchmarks/relation_trace_localization_v1_1.json")
        )
        self.assertEqual(
            specification["execution_contract"]["floating_reproduction_tolerance"],
            64.0 * np.finfo(np.float32).eps,
        )
        self.assertEqual(
            specification["supersedes"]["scientific_contract_changes"], "none"
        )
        validation = validate_registered_sources(specification)
        self.assertEqual(len(validation["pilot_artifacts"]), 2)

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
        np.testing.assert_allclose(metrics["eight_way_identification_accuracy"], 1.0)

    def test_prototype_rdm_compares_geometry_across_feature_spaces(self):
        rng = np.random.default_rng(4)
        first = rng.normal(size=(7, 8, 12))
        transform = rng.normal(size=(12, 20))
        transform, _ = np.linalg.qr(transform)
        second = first @ transform
        result = prototype_rdm_similarity(first, second)
        self.assertAlmostEqual(result["mean"], 1.0)

    def test_decision_tree_distinguishes_storage_access_and_fidelity(
        self,
    ):
        all_present = {
            "generated_effective_write": True,
            "terminal_effective_fast_weight": True,
            "response_full_hidden": True,
            "response_hodge_residual": True,
        }
        self.assertEqual(
            relation_trace_decision(all_present),
            "storage_access_present_missing_fidelity_transformation",
        )
        no_access = dict(all_present)
        no_access["response_full_hidden"] = False
        no_access["response_hodge_residual"] = False
        self.assertEqual(
            relation_trace_decision(no_access),
            "persistent_storage_present_query_access_missing",
        )
        query_emergent = {
            "generated_effective_write": False,
            "terminal_effective_fast_weight": False,
            "response_full_hidden": True,
            "response_hodge_residual": True,
        }
        self.assertEqual(
            relation_trace_decision(query_emergent),
            "mixed_pattern_requires_new_registered_hierarchy",
        )

    def test_registered_result_is_complete_and_source_locked(self):
        result = load_json(
            resolve_record("results/relation_trace_localization_v1_1.json")
        )
        self.assertFalse(result["formal_seed_access"])
        self.assertEqual(set(result["seed_results"]), {"1901", "1902"})
        self.assertEqual(
            result["implementation"]["sha256"],
            registered_file_sha256(
                result["implementation"]["path"],
                result["implementation"]["sha256"],
            ),
        )
        expected = {
            "generated_effective_write": False,
            "terminal_effective_fast_weight": False,
            "response_full_hidden": True,
            "response_hodge_residual": True,
        }
        self.assertEqual(
            result["overall_diagnosis"]["replicated_primary_presence"],
            expected,
        )
        for row in result["seed_results"].values():
            self.assertEqual(row["primary_presence"], expected)
            self.assertEqual(row["validation"]["query_step_0_max_abs_influence"], 0.0)
            self.assertLessEqual(
                row["validation"]["realized_minus_intended_max_abs_error"],
                row["validation"]["floating_reproduction_tolerance"],
            )
