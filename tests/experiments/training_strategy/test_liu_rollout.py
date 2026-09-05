import unittest
from unittest.mock import patch

import numpy as np
import torch

from fsrl.analysis.hodge import build_complete_graph_geometry, normalize_potentials
from fsrl.core.config import TrainConfig
from fsrl.core.local_trace import ConjunctiveLocalTrace
from fsrl.core.plastic_rnn import RetroModulRNN
from fsrl.evaluation.contracts import FrozenEvaluationBackend
from fsrl.evaluation.frozen_fast_weight import FrozenFastWeightEvaluator
from fsrl.evaluation.relational_query import readout_relational_query_bundle
from fsrl.experiments.assembly.trajectory import exact_prefix_trajectory
from fsrl.experiments.training_strategy.evaluation import condition_analysis, json_ready
from fsrl.experiments.training_strategy.legacy_diagnostics import terminal_posterior
from fsrl.experiments.training_strategy.liu_rollout import readout_bundle, rollout_liu
from fsrl.experiments.training_strategy.protocol import load_specification
from fsrl.experiments.training_strategy.reporting import assemble_result, report_text
from fsrl.experiments.training_strategy.summaries import (
    liu_endpoints,
    mechanism_effects,
    summarize_geometry,
)
from fsrl.infra.runtime import ExecutionProfile
from fsrl.tasks.protocol import RankingProtocol, ordered_pairs


class BatchedQueryParityTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(910004)
        torch.set_num_threads(1)
        self.config = TrainConfig(hs=8, bs=2, cs=15)
        self.net = RetroModulRNN(self.config.to_model_dict(), device="cpu")
        self.local = ConjunctiveLocalTrace(15, device="cpu")
        # Non-Liu synthetic chain-plus-chord, not a formal scientific seed.
        self.protocol = RankingProtocol(
            "synthetic-query-parity",
            tuple(str(i) for i in range(8)),
            tuple(range(8)),
            tuple((i, i + 1) for i in range(7)) + ((0, 2),),
            1,
            2,
            {},
        )
        self.evaluator = FrozenFastWeightEvaluator(
            self.net,
            self.config,
            self.protocol,
            cue_seed=910004,
            support_seed=910005,
            subject_encoding_mode="stable_omission",
            subject_encoding_seed=910006,
            backend=FrozenEvaluationBackend.BATCHED_SEQUENCE,
            execution_profile=ExecutionProfile(
                device="cpu", compile=False, require_cuda=False
            ),
        )

    def test_batched_all_conditions_match_maintained_per_query_readout(self):
        weights = torch.randn(2, 8, 8) * 0.1
        local_state = torch.randn(2, 225) * 0.1
        saved_weights, saved_local = weights.clone(), local_state.clone()
        schedules = (ordered_pairs(8),) * 2
        shuffled = np.roll(np.arange(56).reshape(28, 2), 1, axis=0).reshape(56)
        shuffled = np.broadcast_to(shuffled, (2, 56)).copy()
        for local_off, global_off, indices in (
            (False, False, None),
            (True, False, None),
            (False, True, None),
            (False, False, shuffled),
        ):
            actual = readout_bundle(
                self.evaluator,
                self.local,
                weights,
                local_state,
                local_off=local_off,
                global_off=global_off,
                shuffled_indices=indices,
            )
            expected = readout_relational_query_bundle(
                self.evaluator,
                self.local,
                weights,
                local_state,
                schedules,
                local_off=local_off,
                global_off=global_off,
                shuffled_indices=indices,
            )
            for key, values in actual.items():
                with self.subTest(local_off=local_off, global_off=global_off, key=key):
                    np.testing.assert_allclose(
                        values, expected[key], atol=1e-6, rtol=1e-5
                    )
        torch.testing.assert_close(weights, saved_weights, rtol=0, atol=0)
        torch.testing.assert_close(local_state, saved_local, rtol=0, atol=0)

    def test_synthetic_rollout_covers_routing_loo_and_endpoint_adapters(self):
        result = rollout_liu(self.evaluator, self.local, load_specification())
        self.assertEqual(
            set(result["bundles"]),
            {"intact", "local_off", "P_off", "query_shuffle", "evidence_shuffle"},
        )
        for branch in result["loo"].values():
            self.assertEqual(branch.shape, (8, 2, 56))
        np.testing.assert_array_equal(
            np.sort(result["natural_local_evidence"], axis=1),
            np.sort(result["shuffled_local_evidence"], axis=1),
        )
        self.assertTrue(np.all(result["query_routing"] != np.arange(56)))
        np.testing.assert_array_equal(
            result["query_routing"][:, ::2] + 1, result["query_routing"][:, 1::2]
        )
        endpoints = liu_endpoints(
            result["bundles"], result["retention"], self.protocol, 0.25
        )
        self.assertEqual(
            set(endpoints["intact"]["probability"]),
            {"overall", "learned", "nonlearned", "retained", "omitted"},
        )
        statistics = {"samples": 100, "interval": 0.95}
        geometry = summarize_geometry(
            result["bundles"], result["loo"], self.protocol, 92, statistics
        )
        effects = mechanism_effects(endpoints, geometry, 92, statistics)
        self.assertEqual(len(effects), 10)
        self.assertEqual(
            geometry["loo_relation_subject"]["global"]["remote_absolute"].shape, (8, 2)
        )

    def test_terminal_comparator_matches_frozen_prefix_calculation(self):
        terminal = terminal_posterior(self.evaluator)
        old = exact_prefix_trajectory(
            self.evaluator,
            self.protocol,
            build_complete_graph_geometry(self.protocol),
            temperature=0.05,
        )
        np.testing.assert_allclose(
            normalize_potentials(terminal["expected_rank"]),
            old.expected_rank_potentials[-1],
            atol=1e-14,
            rtol=1e-14,
        )
        np.testing.assert_allclose(
            normalize_potentials(terminal["MAP"]),
            old.map_potentials[-1],
            atol=1e-14,
            rtol=1e-14,
        )

    def test_complete_synthetic_analysis_to_report(self):
        specification = load_specification()
        specification["statistics"]["samples"] = 100
        specification["evaluation"]["generic"]["episodes"] = 8
        metadata = {
            "task_distribution": {"liu_graph_held_out": True},
            "cost": {"warm_training_seconds": 1.0, "peak_allocated_bytes": 1024},
        }
        profile = ExecutionProfile(device="cpu", compile=False, require_cuda=False)
        with patch(
            "fsrl.experiments.training_strategy.generic_validation.PROFILE", profile
        ):
            result, raw = condition_analysis(
                self.evaluator, self.local, metadata, 2108, specification
            )
        self.assertEqual(len(result["behavior"]["flags"]), 9)
        self.assertIn("liu__loo__combined", raw["arrays"])
        self.assertIn("projection__local_off__expected_minus_MAP", raw["arrays"])
        conditions = {
            f"{seed}/{name}": json_ready(result)
            for seed in specification["seeds"]["mandatory"]
            for name in specification["seeds"]["conditions"]
        }
        with (
            patch(
                "fsrl.experiments.training_strategy.reporting.verify_matched_evaluation"
            ),
            patch(
                "fsrl.experiments.training_strategy.reporting.reference",
                return_value={},
            ),
        ):
            assembled = assemble_result(
                conditions, {"source_commit": "synthetic"}, specification
            )
        text = report_text(assembled)
        for seed in (2108, 2109, 2110):
            self.assertIn(f"## Seed {seed}", text)
        self.assertIn("single-stage", text)
        self.assertIn("symbolic_distance_effect", text)
        self.assertIn("48,000", text)
