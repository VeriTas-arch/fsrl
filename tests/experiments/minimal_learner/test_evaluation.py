import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from fsrl.experiments.minimal_learner import evaluation, reporting
from fsrl.experiments.minimal_learner.data import ModelBatch, generic_batch, pair_cues
from fsrl.experiments.minimal_learner.decisions import (
    adequate,
    outcome,
    pair_analysis,
    study_outcome,
)
from fsrl.experiments.minimal_learner.model import make_model
from fsrl.experiments.minimal_learner.protocol import specification, task_generator
from fsrl.experiments.training_strategy.evaluation import json_ready, write_arrays
from fsrl.tasks.protocol import RankingProtocol, ordered_pairs


def synthetic_data():
    episode = task_generator().sample(np.random.default_rng(910201), n_edges=8)
    support = tuple(
        (episode.true_order_high_to_low[i], episode.true_order_high_to_low[j])
        for i, j in episode.graph_rank_pairs
    )
    protocol = RankingProtocol(
        "synthetic",
        tuple("ABCDEFGH"),
        episode.true_order_high_to_low,
        support,
        4,
        10,
        {},
    )
    batch = generic_batch((episode,) * 3)
    queries = np.broadcast_to(np.asarray(ordered_pairs(8))[None], (3, 56, 2))
    return protocol, ModelBatch(
        {
            **batch.arrays,
            "query_pairs": queries,
            "query_cues": pair_cues(batch.arrays["codes"], queries),
        }
    )


class MinimalEvaluationTests(unittest.TestCase):
    def test_artifact_guard_precedes_runtime_or_data(self):
        with (
            patch.object(
                evaluation, "validate_artifacts", side_effect=RuntimeError("locked")
            ),
            patch.object(evaluation, "runtime") as runtime,
            patch.object(evaluation, "liu_batch") as data,
        ):
            with self.assertRaisesRegex(RuntimeError, "locked"):
                evaluation.evaluate_all()
            runtime.assert_not_called()
            data.assert_not_called()
        with (
            patch.object(
                reporting, "validate_artifacts", side_effect=RuntimeError("locked")
            ),
            patch.object(reporting, "validate_evaluation") as evaluate,
        ):
            with self.assertRaisesRegex(RuntimeError, "locked"):
                reporting.write_report()
            evaluate.assert_not_called()

    def test_synthetic_all_controls_through_report_and_numeric_storage(self):
        spec = specification()
        spec["statistics"]["samples"] = 20
        spec["evaluation"]["generic"]["episodes"] = 4
        protocol, batch = synthetic_data()
        conditions = {}
        for condition in ("score_trace", "score_only"):
            model = make_model(condition, spec)
            with patch.object(evaluation, "liu_batch", return_value=(protocol, batch)):
                result, raw = evaluation.condition_analysis(model, model, 1, spec)
            self.assertEqual(len(result["behavior"]["flags"]), 9)
            self.assertEqual(
                set(result["endpoints"]["liu"]),
                set(spec["evaluation"]["liu"]["conditions"]),
            )
            self.assertLess(result["history"]["float64_to_float32_max_abs_error"], 1e-5)
            result.update(
                {
                    "parameters": {"eta": 0.5},
                    "cost": {
                        "global_persistent_entries": 15,
                        "local_persistent_entries": 225
                        if model.local is not None
                        else 0,
                        "trainable_parameters": 3 if model.local is not None else 2,
                        "warm_training_seconds": 1.0,
                    },
                }
            )
            conditions[condition] = result
            arrays = raw["arrays"]
            if condition == "score_only":
                for control in ("local_off", "query_shuffle", "evidence_shuffle"):
                    np.testing.assert_array_equal(
                        arrays[f"liu__bundles__{control}__logits"],
                        arrays["liu__bundles__intact__logits"],
                    )
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "raw.npz"
                write_arrays(path, arrays)
                with np.load(path, allow_pickle=False) as saved:
                    self.assertTrue(
                        all(saved[key].dtype.kind in "biuf" for key in saved.files)
                    )
        pair = pair_analysis(
            conditions["score_trace"], conditions["score_only"], 1, spec
        )
        self.assertEqual(len(pair["local_support"]), 5)
        report_result = json_ready(
            {
                "conditions": conditions,
                "pairs": {"1": pair},
                "outcome": pair["outcome"],
                "stop_rule": "stop",
            }
        )
        assert isinstance(report_result, dict)
        rendered = reporting.report_text(report_result, spec)
        self.assertIn("score_only", rendered)
        self.assertIn("history_sensitivity", rendered)
        self.assertIn("Frozen quantitative", rendered)

    def test_independent_score_only_can_win_and_missing_behavior_cannot(self):
        spec = specification()
        names = {
            "learned_accuracy",
            "nonlearned_accuracy",
            "symbolic_distance_effect",
            "serial_position_effect",
            "difficult_pair_bimodality",
            "stable_within_subject_errors",
            "self_consistent_vs_inconsistent_errors",
            "hodge_reconstructed_subjective_ranking",
            "inter_subject_ranking_diversity",
        }
        passed = {
            "behavior": {"flags": {name: {"qualitative": True} for name in names}},
            "endpoints": {
                domain: {
                    "intact": {
                        "exact_decision": {group: {"mean": 0.9} for group in groups}
                    }
                }
                for domain, groups in (
                    ("generic", ("learned", "nonlearned")),
                    ("liu", ("overall", "nonlearned")),
                )
            },
        }
        failed = copy.deepcopy(passed)
        failed["behavior"]["flags"]["stable_within_subject_errors"]["qualitative"] = (
            False
        )
        checks = {str(i): {"passed": True} for i in range(5)}
        self.assertEqual(outcome(failed, passed, checks, spec), "score_only_sufficient")
        self.assertEqual(
            outcome(passed, failed, checks, spec), "compact_dual_state_candidate"
        )
        checks["0"]["passed"] = False
        self.assertEqual(
            outcome(passed, failed, checks, spec),
            "compact_behavior_solution_mechanism_unresolved",
        )
        self.assertEqual(
            outcome(failed, failed, checks, spec), "competent_behavior_incomplete"
        )
        del passed["behavior"]["flags"]["symbolic_distance_effect"]
        self.assertFalse(adequate(passed, spec))
        with self.assertRaisesRegex(RuntimeError, "mandatory"):
            study_outcome({"2111": {"outcome": "score_only_sufficient"}}, spec)


if __name__ == "__main__":
    unittest.main()
