import unittest

import numpy as np
import torch

from fsrl.analysis.behavioral import analyze_sampled_query_policy
from fsrl.core.config import TrainConfig
from fsrl.core.plastic_rnn import RetroModulRNN
from fsrl.evaluation.frozen_fast_weight import (
    FastWeightIntervention,
    FrozenFastWeightEvaluator,
)
from fsrl.experiments.transport.item_count import (
    DEFAULT_SPECIFICATION_PATH,
    cross_cell_decision,
    individualized_metrics_generic,
    protocol_for_size,
    validate_graph_contract,
    validate_n8_evaluator_interface,
)
from fsrl.infra.provenance import load_json
from fsrl.infra.study_registry import resolve_record
from fsrl.tasks.protocol import load_ranking_protocol


def _cell(*, interpretable=True, competence=True, all_pass=True):
    flags = {
        "intact_competence": competence,
        "constructive_global_structure": all_pass,
        "size_normalized_individualized_stable_structure": all_pass,
        "P_off_global_collapse": all_pass,
        "P_remote_reassembly": all_pass,
        "a_off_direct_loss": all_pass,
        "P_off_a_on_direct_nontransitive": all_pass,
        "exact_local_compression": all_pass,
    }
    return {
        "decision": {
            "interpretable": interpretable,
            "competence_passed": competence and interpretable,
            "all_primary_links_pass": all(flags.values()) and interpretable,
            "flags": flags,
        }
    }


class ItemCountTransportTests(unittest.TestCase):
    def setUp(self):
        self.specification = load_json(DEFAULT_SPECIFICATION_PATH)
        self.base = load_ranking_protocol(resolve_record("benchmarks/liu_v2.json"))

    def test_registered_cycles_reconstruct_exactly(self):
        result = validate_graph_contract(self.specification)
        self.assertTrue(result["passed"])
        self.assertEqual(
            [row["number_of_exact_minimizers"] for row in result["checks"]],
            [7, 1, 110],
        )
        self.assertEqual(
            [row["wasserstein_to_N8_target"] for row in result["checks"]],
            ["8/105", "0", "19/420"],
        )

    def test_protocols_have_registered_size_and_trial_counts(self):
        for graph in self.specification["size_matched_graph_contract"]["graphs"]:
            protocol = protocol_for_size(self.base, graph)
            self.assertEqual(protocol.n_items, graph["n_items"])
            self.assertEqual(protocol.support_trials, graph["support_trials"])
            self.assertEqual(protocol.query_trials, graph["query_trials"])
            self.assertEqual(len(protocol.support_pairs_higher_lower), protocol.n_items)
        self.assertIs(
            protocol_for_size(
                self.base,
                self.specification["size_matched_graph_contract"]["graphs"][1],
            ),
            self.base,
        )

    def test_variable_evaluator_exactly_replays_n8_constructor(self):
        config = TrainConfig(bs=3, hs=4, cs=6)
        net = RetroModulRNN(config.to_model_dict(), device="cpu")
        kwargs = {
            "cue_seed": 13,
            "support_seed": 17,
            "cue_mode": "permuted_shared",
            "subject_encoding_mode": "stable_omission",
            "subject_encoding_seed": 19,
        }
        frozen = FrozenFastWeightEvaluator(net, config, self.base, **kwargs)
        variable = FrozenFastWeightEvaluator(
            net, config, self.base, required_item_count=None, **kwargs
        )
        self.assertTrue(validate_n8_evaluator_interface(variable, frozen)["passed"])

    def test_variable_evaluator_reuses_complete_rollout_initialization(self):
        torch.manual_seed(37)
        config = TrainConfig(bs=2, hs=4, cs=6)
        net = RetroModulRNN(config.to_model_dict(), device="cpu")
        kwargs = {
            "cue_seed": 13,
            "support_seed": 17,
            "subject_encoding_mode": "stable_omission",
            "subject_encoding_seed": 19,
        }
        frozen = FrozenFastWeightEvaluator(net, config, self.base, **kwargs)
        variable = FrozenFastWeightEvaluator(
            net, config, self.base, required_item_count=None, **kwargs
        )
        frozen_weights = frozen.learn_fast_weights(FastWeightIntervention.INTACT)
        variable_weights = variable.learn_fast_weights(FastWeightIntervention.INTACT)
        torch.testing.assert_close(variable_weights, frozen_weights, rtol=0.0, atol=0.0)

        schedules = tuple(((0, 1), (1, 0)) for _ in range(config.bs))
        self.assertEqual(
            variable.readout_logits(variable_weights, schedules),
            frozen.readout_logits(frozen_weights, schedules),
        )

        n6_protocol = protocol_for_size(
            self.base,
            self.specification["size_matched_graph_contract"]["graphs"][0],
        )
        n6 = FrozenFastWeightEvaluator(
            net, config, n6_protocol, required_item_count=None, **kwargs
        )
        n6_weights = n6.initialize_fast_weights()
        n6_weights = n6.advance_support_trial(n6_weights, 0)
        self.assertEqual(n6_weights.shape, (config.bs, config.hs, config.hs))

    def test_behavior_estimator_runs_at_registered_sizes(self):
        rng = np.random.default_rng(23)
        for graph in self.specification["size_matched_graph_contract"]["graphs"][:2]:
            protocol = protocol_for_size(self.base, graph)
            logits = tuple(
                {
                    pair: float(rng.normal())
                    for first in range(protocol.n_items)
                    for second in range(first + 1, protocol.n_items)
                    for pair in ((first, second), (second, first))
                }
                for _ in range(3)
            )
            behavior = analyze_sampled_query_policy(
                protocol, logits, seed=29, temperature=0.25
            )
            self.assertEqual(len(behavior["subjects"]), 3)
            self.assertEqual(len(behavior["pairs"]), protocol.query_trials // 10)

    def test_size_normalized_stable_error_density(self):
        behavior = {
            "subjects": [
                {
                    "overall_accuracy": 0.7,
                    "ranking_class": "self_consistent_incorrect",
                    "stable_error_pair_counts": {"80": 2},
                    "subjective_order_high_to_low": [0, 1, 2, 3, 4, 5],
                },
                {
                    "overall_accuracy": 0.8,
                    "ranking_class": "self_consistent_incorrect",
                    "stable_error_pair_counts": {"80": 1},
                    "subjective_order_high_to_low": [0, 1, 2, 3, 5, 4],
                },
            ]
        }
        result = individualized_metrics_generic(
            behavior, np.random.default_rng(31), 100, 6
        )
        self.assertAlmostEqual(result["stable_error_80_pair_density"]["mean"], 0.1)
        self.assertEqual(result["eligible_noncorrect_subjects"], 2)

    def test_cross_cell_outcomes_never_pool(self):
        sizes = [6, 8, 10]
        seed_ids = [2101, 2102, 2103]
        seeds = {
            str(seed): {"sizes": {str(size): _cell() for size in sizes}}
            for seed in seed_ids
        }
        self.assertEqual(
            cross_cell_decision(seeds, sizes, seed_ids)["outcome"],
            "LIU_ITEM_COUNT_MECHANISM_TRANSPORTED",
        )
        seeds["2101"]["sizes"]["6"] = _cell(all_pass=False)
        self.assertEqual(
            cross_cell_decision(seeds, sizes, seed_ids)["outcome"],
            "ITEM_COUNT_DEPENDENT_OR_UNRESOLVED",
        )
        for seed in seed_ids:
            seeds[str(seed)]["sizes"]["6"] = _cell(all_pass=False)
            seeds[str(seed)]["sizes"]["10"] = _cell(all_pass=False)
        self.assertEqual(
            cross_cell_decision(seeds, sizes, seed_ids)["outcome"],
            "FUNCTIONAL_ASYMMETRY_NOT_ITEM_COUNT_TRANSPORTED",
        )
        seeds["2101"]["sizes"]["6"] = _cell(competence=False, all_pass=False)
        self.assertEqual(
            cross_cell_decision(seeds, sizes, seed_ids)["outcome"],
            "ITEM_COUNT_COMPETENCE_NOT_ESTABLISHED",
        )
        seeds["2101"]["sizes"]["6"] = _cell(
            interpretable=False, competence=False, all_pass=False
        )
        self.assertEqual(
            cross_cell_decision(seeds, sizes, seed_ids)["outcome"],
            "NONINTERPRETABLE_EXECUTION",
        )
