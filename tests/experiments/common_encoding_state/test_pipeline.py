import unittest

import numpy as np

from fsrl.experiments.common_encoding_state.cohorts import (
    ALL_ENDPOINTS,
    HUMAN_COMPOSITION,
    summarize_points,
)
from fsrl.experiments.common_encoding_state.protocol import (
    CONDITIONS,
    PARENT_SEEDS,
    RECOVERY_DATASETS,
    RHO_GRID,
    SEEDS,
)
from fsrl.experiments.common_encoding_state.recovery_execution import summarize


def recovery_rows(success=True):
    rows = []
    for rho in RHO_GRID:
        for generator in CONDITIONS:
            for dataset in range(RECOVERY_DATASETS):
                architecture = generator if success else "independent"
                rows.append(
                    {
                        "generator": generator,
                        "generating_rho": rho,
                        "dataset": dataset,
                        "architecture": architecture,
                        "rho": None if generator == "independent" else rho,
                    }
                )
    return rows


def fit_point(correct, consistent, inconsistent):
    values = {
        "learned_accuracy": 0.91,
        "nonlearned_accuracy": 0.83,
        "symbolic_distance_effect": 0.044,
        "serial_position_effect": 0.071,
        "stable_within_subject_errors": 0.928,
        "self_consistent_incorrect": consistent,
        "self_inconsistent": inconsistent,
        "correct_ranker": correct,
        "inter_subject_ranking_diversity": 0.57,
    }
    composition = np.asarray([correct, consistent, inconsistent])
    return {
        "values": {name: values[name] for name in ALL_ENDPOINTS},
        "flags": {"fixture": {"qualitative": True, "calibration": True}},
        "internal_strict_correct": correct,
        "mean_inversion_count": 1 - correct,
        "inversion_bin_fractions": [correct, 0.2, 0.3, 0.5 - correct],
        "internal_to_sampled_transition": [
            [correct, 0, 0],
            [0, consistent, inconsistent],
        ],
        "sampled_correct_all_subjects": correct,
        "loss_flow": 0,
        "rescue_flow": 0,
        "net_sampling_shift": 0,
        "ranking_composition_total_variation": float(
            0.5 * np.abs(composition - HUMAN_COMPOSITION).sum()
        ),
    }


class PipelineTests(unittest.TestCase):
    def test_recovery_selects_smallest_passing_rho_and_stops_on_failure(self):
        passed = summarize(recovery_rows())
        self.assertTrue(passed["passed"])
        self.assertEqual(passed["selected_rho"], RHO_GRID[0])
        failed = summarize(recovery_rows(success=False))
        self.assertFalse(failed["passed"])
        self.assertIsNone(failed["selected_rho"])

    def test_joint_composition_requires_episode_over_relation_in_every_fit(self):
        points = []
        for cohort in range(400):
            fits = {}
            for source, seeds in (("fixed", PARENT_SEEDS), ("trained", SEEDS)):
                for seed in seeds:
                    fits[f"{source}/{seed}/independent"] = fit_point(0.04, 0.94, 0.02)
                    fits[f"{source}/{seed}/relation_common"] = fit_point(
                        0.04, 0.94, 0.02
                    )
                    fits[f"{source}/{seed}/episode_common"] = fit_point(
                        0.08, 0.90, 0.02
                    )
            points.append({"cohort": cohort, "fits": fits})
        result = summarize_points(points)
        self.assertEqual(result["decision"]["outcome"], "common_encoding_supported")
        self.assertFalse(result["decision"]["main_model_promoted"])


if __name__ == "__main__":
    unittest.main()
