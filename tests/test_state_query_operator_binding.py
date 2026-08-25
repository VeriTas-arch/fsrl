import unittest

import numpy as np

from fsrl.assembly_diagnostics import load_json
from fsrl.assembly_trajectory import bootstrap_counts
from fsrl.hidden_residual_audit import validate_registered_sources
from fsrl.state_query_operator_binding import (
    _decision,
    contextual_identity_metrics,
    overlap_classes,
    structured_contrasts,
)
from fsrl.study_registry import registered_file_sha256, resolve_record


class StateQueryOperatorBindingTests(unittest.TestCase):
    def test_fixed_and_cross_context_identity_for_invariant_code(self):
        identities = 8
        subjects = 14
        contexts = 8
        traces = np.zeros((identities, subjects, contexts, identities))
        for identity in range(identities):
            traces[identity, :, :, identity] = 1.0
        valid = np.ones((identities, subjects, contexts), dtype=bool)
        for cross_context in (False, True):
            metrics = contextual_identity_metrics(
                traces,
                valid,
                subject_folds=7,
                tolerance=1e-12,
                cross_context=cross_context,
            )
            np.testing.assert_allclose(metrics["own_minus_other_selectivity"], 1.0)
            np.testing.assert_allclose(
                metrics["eight_way_identification_accuracy"], 1.0
            )

    def test_cross_context_rejects_context_specific_coordinate_blocks(self):
        identities = 8
        subjects = 14
        contexts = 8
        traces = np.zeros((identities, subjects, contexts, identities * contexts))
        for identity in range(identities):
            for context in range(contexts):
                traces[identity, :, context, context * identities + identity] = 1.0
        valid = np.ones((identities, subjects, contexts), dtype=bool)
        fixed = contextual_identity_metrics(
            traces,
            valid,
            subject_folds=7,
            tolerance=1e-12,
            cross_context=False,
        )
        cross = contextual_identity_metrics(
            traces,
            valid,
            subject_folds=7,
            tolerance=1e-12,
            cross_context=True,
        )
        np.testing.assert_allclose(fixed["eight_way_identification_accuracy"], 1.0)
        np.testing.assert_allclose(cross["own_minus_other_selectivity"], 0.0)
        self.assertAlmostEqual(
            float(np.mean(cross["eight_way_identification_accuracy"])), 0.125
        )

    def test_overlap_classes_and_structured_binding_are_frozen(self):
        relations = (
            (5, 0),
            (2, 1),
            (4, 1),
            (6, 2),
            (5, 3),
            (6, 3),
            (7, 4),
            (7, 0),
        )
        overlap = overlap_classes(relations)
        values = np.ones((8, 14, 8))
        for relation in range(8):
            values[relation, :, overlap[relation] == 1] = 2.0
            values[relation, :, relation] = 3.0
        retained = np.ones((8, 14), dtype=bool)
        counts = bootstrap_counts(np.random.default_rng(3), 100, 14)
        result = structured_contrasts(values, retained, overlap, counts, interval=0.95)
        self.assertAlmostEqual(
            result["summary"]["matched_minus_shared_endpoint"]["mean"], 1.0
        )
        self.assertAlmostEqual(result["summary"]["matched_minus_disjoint"]["mean"], 2.0)

    def test_decision_tree_separates_binding_nonlinearity_and_loss(self):
        flags = {
            "operator_state_identity": True,
            "operator_binding": True,
            "hidden_state_identity": True,
            "hidden_query_identity_control": True,
        }
        self.assertEqual(
            _decision(flags),
            "query_keyed_operator_missing_fidelity_transformation",
        )
        nonlinear = dict(flags)
        nonlinear["operator_state_identity"] = False
        nonlinear["operator_binding"] = False
        self.assertEqual(
            _decision(nonlinear),
            "identity_generated_by_operating_point_nonlinearity",
        )
        lost = dict(flags)
        lost["hidden_state_identity"] = False
        self.assertEqual(
            _decision(lost), "operator_identity_lost_in_recurrent_expression"
        )

    def test_registered_sources_are_immutable(self):
        specification = load_json(resolve_record("benchmarks/state_query_operator_binding_v1.json"))
        validation = validate_registered_sources(specification)
        self.assertEqual(len(validation["pilot_artifacts"]), 2)

    def test_registered_result_is_complete_and_source_locked(self):
        result = load_json(resolve_record("results/state_query_operator_binding_v1.json"))
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
            "operator_state_identity": True,
            "cross_query_operator_state_identity": False,
            "operator_binding": True,
            "hidden_state_identity": True,
            "cross_query_hidden_state_identity": False,
            "hidden_query_identity_control": True,
        }
        self.assertEqual(
            result["overall_diagnosis"]["replicated_primary_presence"], expected
        )
        for row in result["seed_results"].values():
            self.assertEqual(row["primary_presence"], expected)
            self.assertEqual(
                row["validation"]["stable_omitted_operator_action_max_abs"], 0.0
            )
            self.assertEqual(
                row["validation"]["stable_omitted_hidden_effect_max_abs"], 0.0
            )
            self.assertLessEqual(
                row["validation"][
                    "operator_preactivation_reconstruction_max_abs_error"
                ],
                row["validation"]["floating_reproduction_tolerance"],
            )


if __name__ == "__main__":
    unittest.main()
