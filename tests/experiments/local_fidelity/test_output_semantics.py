import unittest

import numpy as np

from fsrl.experiments.assembly.diagnostics import load_json
from fsrl.experiments.assembly.trajectory import build_complete_graph_geometry
from fsrl.experiments.local_fidelity.hidden_residual import validate_registered_sources
from fsrl.experiments.local_fidelity.output_semantics import (
    STAGES,
    _relation_geometry,
    classify_stage,
    classify_transition,
    decide_outcome,
    hodge_components,
    normalized_direct_correctness,
    stage_relation_metrics,
)
from fsrl.infra.study_registry import registered_file_sha256, resolve_record
from fsrl.tasks.registered_protocol import load_ranking_protocol


def _summary(lower, upper):
    return {"bootstrap": {"lower": lower, "upper": upper}}


class OperatorOutputSemanticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = load_ranking_protocol(resolve_record("benchmarks/liu_v2.json"))
        cls.geometry = build_complete_graph_geometry(cls.protocol)

    def test_hodge_removes_additive_field(self):
        potentials = np.arange(self.protocol.n_items, dtype=np.float64)
        field = self.geometry.incidence @ potentials
        gradient, residual = hodge_components(field, self.geometry)
        np.testing.assert_allclose(gradient, field, atol=1e-12)
        np.testing.assert_allclose(residual, 0.0, atol=1e-12)

        perturbed = field.copy()
        perturbed[0] += 1.0
        _gradient, residual = hodge_components(perturbed, self.geometry)
        self.assertGreater(float(np.linalg.norm(residual)), 0.0)

    def test_normalized_correctness_separates_direction_from_scale(self):
        signs = np.asarray([1.0, -1.0, 1.0])
        retained = np.ones((3, 2), dtype=bool)
        aligned = signs[:, None] * np.asarray([[1.0, 4.0]])
        rho = normalized_direct_correctness(aligned, signs, retained, tolerance=1e-12)
        np.testing.assert_allclose(rho, 1.0)
        scaled = normalized_direct_correctness(
            0.2 * aligned, signs, retained, tolerance=1e-12
        )
        np.testing.assert_allclose(scaled, rho)
        opposed = normalized_direct_correctness(
            -aligned, signs, retained, tolerance=1e-12
        )
        np.testing.assert_allclose(opposed, -1.0)

    def test_relation_geometry_freezes_direct_and_remote_edges(self):
        direct, signs, remote = _relation_geometry(self.protocol, self.geometry)
        self.assertEqual(len(direct), 8)
        self.assertEqual(set(signs), {-1.0})
        self.assertTrue(all(int(np.sum(mask)) == 15 for mask in remote))

        fields = np.zeros((8, 77, 28), dtype=np.float64)
        metrics = stage_relation_metrics(
            fields, fields, self.geometry, direct, signs, remote
        )
        self.assertEqual(metrics["remote_correctness"].shape, (8, 77))
        self.assertEqual(metrics["remote_absolute_residual"].shape, (8, 77))

    def test_directional_rules_and_outcome_tree(self):
        aligned = {
            "direct_correctness": _summary(0.1, 0.2),
            "normalized_direct_correctness_rho": _summary(0.2, 0.4),
        }
        opposed = {
            "direct_correctness": _summary(-0.2, -0.1),
            "normalized_direct_correctness_rho": _summary(-0.4, -0.2),
        }
        self.assertEqual(classify_stage(aligned), "correctness_aligned")
        self.assertEqual(classify_stage(opposed), "correctness_opposed")
        self.assertEqual(classify_transition(_summary(-0.2, -0.1)), "degradation")

        stage_status = {stage: "correctness_aligned" for stage in STAGES}
        transitions = {"J_minus_A": "unresolved", "H_minus_J": "unresolved"}
        h_status = {stage: "correctness_aligned" for stage in STAGES}
        self.assertEqual(
            decide_outcome(stage_status, transitions, h_status),
            "correct_semantics_reach_policy_amplitude_or_combination_unresolved",
        )
        h_status[STAGES[0]] = "correctness_opposed"
        self.assertEqual(
            decide_outcome(stage_status, transitions, h_status),
            "aggregate_aligned_but_H_greater_A_opposed",
        )

    def test_registered_sources_are_immutable(self):
        specification = load_json(
            resolve_record("benchmarks/operator_output_semantics_v1.json")
        )
        validation = validate_registered_sources(specification)
        self.assertEqual(len(validation["pilot_artifacts"]), 2)

    def test_registered_result_is_complete_and_source_locked(self):
        result = load_json(resolve_record("results/operator_output_semantics_v1.json"))
        self.assertFalse(result["formal_seed_access"])
        self.assertEqual(set(result["seed_results"]), {"1901", "1902"})
        self.assertEqual(
            result["implementation"]["sha256"],
            registered_file_sha256(
                result["implementation"]["path"],
                result["implementation"]["sha256"],
            ),
        )
        self.assertEqual(
            result["specification"]["sha256"],
            registered_file_sha256(
                result["specification"]["path"],
                result["specification"]["sha256"],
            ),
        )
        self.assertEqual(
            result["overall_diagnosis"]["outcome"],
            "aggregate_aligned_but_H_greater_A_opposed",
        )
        expected_stage = {stage: "correctness_aligned" for stage in STAGES}
        expected_h_greater_a = {
            STAGES[0]: "correctness_aligned",
            STAGES[1]: "correctness_aligned",
            STAGES[2]: "correctness_opposed",
        }
        for row in result["seed_results"].values():
            self.assertEqual(row["diagnosis"]["stage_status"], expected_stage)
            self.assertEqual(
                row["diagnosis"]["H_greater_A_stage_status"], expected_h_greater_a
            )
            self.assertTrue(row["diagnosis"]["stable_omission_pass"])
            tolerance = row["validation"]["floating_reproduction_tolerance"]
            self.assertLessEqual(
                row["validation"][
                    "exact_hidden_to_actual_logit_influence_max_abs_error"
                ],
                tolerance,
            )
            for name in (
                "stable_omitted_oriented_scalar_max_abs",
                "stable_omitted_field_max_abs",
                "stable_omitted_residual_max_abs",
            ):
                self.assertEqual(set(row["validation"][name].values()), {0.0})


if __name__ == "__main__":
    unittest.main()
