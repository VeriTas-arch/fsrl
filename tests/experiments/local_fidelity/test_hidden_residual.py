import unittest

import numpy as np

from fsrl.analysis.hodge import build_complete_graph_geometry
from fsrl.experiments.local_fidelity.hidden_residual import (
    cross_validated_local_direction,
    vector_hodge_components,
)
from fsrl.infra.provenance import load_json
from fsrl.infra.study_registry import registered_file_sha256, resolve_record
from fsrl.tasks.protocol import RankingProtocol


class HiddenResidualAuditTests(unittest.TestCase):
    def setUp(self):
        self.protocol = RankingProtocol(
            protocol_id="hidden-residual-test",
            item_labels=("a", "b", "c", "d"),
            true_order_high_to_low=(0, 1, 2, 3),
            support_pairs_higher_lower=((0, 1), (1, 2), (2, 3)),
            support_blocks=1,
            query_blocks=1,
            human_targets={},
        )
        self.geometry = build_complete_graph_geometry(self.protocol)

    def test_vector_hodge_components_reconstruct_and_separate_gradient(self):
        rng = np.random.default_rng(4)
        item_vectors = rng.normal(size=(2, self.protocol.n_items, 5))
        gradient_field = np.einsum("ei,bih->beh", self.geometry.incidence, item_vectors)
        gradient, residual = vector_hodge_components(gradient_field, self.geometry)
        np.testing.assert_allclose(gradient, gradient_field, atol=1e-12)
        np.testing.assert_allclose(residual, 0.0, atol=1e-12)

        arbitrary = rng.normal(size=(2, len(self.geometry.pairs), 5))
        gradient, residual = vector_hodge_components(arbitrary, self.geometry)
        np.testing.assert_allclose(gradient + residual, arbitrary, atol=1e-12)
        projected_residual, _ = vector_hodge_components(residual, self.geometry)
        np.testing.assert_allclose(projected_residual, 0.0, atol=1e-12)

    def test_cross_validated_direction_generalizes_subjects_and_relations(self):
        relations = 3
        subjects = 14
        edges = len(self.geometry.pairs)
        hidden = 6
        direct_edges = np.asarray([0, 3, 5])
        signs = np.asarray([1.0, -1.0, 1.0])
        remote_masks = tuple(
            np.asarray([edge != direct for edge in range(edges)])
            for direct in direct_edges
        )
        retained = np.ones((relations, subjects), dtype=bool)
        signal = np.asarray([1.0, -0.5, 0.25, 0.0, 0.0, 0.0])
        influence = np.zeros((relations, subjects, edges, hidden))
        for relation in range(relations):
            influence[relation, :, direct_edges[relation]] = signs[relation] * signal

        metrics = cross_validated_local_direction(
            influence,
            retained,
            direct_edges,
            signs,
            remote_masks,
            np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
            subject_folds=7,
        )
        np.testing.assert_allclose(
            metrics["local_direct_correctness"], np.linalg.norm(signal)
        )
        np.testing.assert_allclose(metrics["local_remote_absolute"], 0.0)
        np.testing.assert_allclose(metrics["output_direct_correctness"], 0.0)

    def test_registered_result_is_complete_and_source_locked(self):
        result = load_json(resolve_record("results/hidden_residual_audit_v1.json"))
        self.assertFalse(result["formal_seed_access"])
        self.assertEqual(set(result["seed_results"]), {"1901", "1902"})
        self.assertEqual(
            result["implementation"]["sha256"],
            registered_file_sha256(
                result["implementation"]["path"],
                result["implementation"]["sha256"],
            ),
        )
        for row in result["seed_results"].values():
            self.assertEqual(
                row["validation"]["stable_omitted_hidden_influence_max_abs"],
                0.0,
            )
            self.assertLessEqual(
                row["validation"]["hidden_to_logit_projection_max_abs_error"],
                row["validation"]["floating_reproduction_tolerance"],
            )
