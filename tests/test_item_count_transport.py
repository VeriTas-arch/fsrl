import unittest
from unittest.mock import patch

import numpy as np

from fsrl.behavioral import analyze_sampled_query_policy
from fsrl.config import TrainConfig
from fsrl.item_count_transport import (
    DEFAULT_SPECIFICATION_PATH,
    VariableItemFrozenFastWeightEvaluator,
    analyze_size_generic_sampled_query_policy,
    cross_cell_decision,
    individualized_metrics_generic,
    load_json,
    protocol_for_size,
    validate_graph_contract,
    validate_n8_evaluator_interface,
)
from fsrl.liu_eval import FrozenFastWeightEvaluator
from fsrl.ranking_protocol import load_ranking_protocol
from fsrl.study_registry import resolve_record


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
        config = TrainConfig(bs=3, cs=6)
        kwargs = {
            "cue_seed": 13,
            "support_seed": 17,
            "cue_mode": "permuted_shared",
            "subject_encoding_mode": "stable_omission",
            "subject_encoding_seed": 19,
        }
        frozen = FrozenFastWeightEvaluator(object(), config, self.base, **kwargs)
        variable = VariableItemFrozenFastWeightEvaluator(
            object(), config, self.base, **kwargs
        )
        self.assertTrue(validate_n8_evaluator_interface(variable, frozen)["passed"])

    def test_size_generic_behavior_is_exact_at_n8_and_runs_at_n6(self):
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
            generic = analyze_size_generic_sampled_query_policy(
                protocol, logits, seed=29, temperature=0.25
            )
            self.assertEqual(len(generic["subjects"]), 3)
            self.assertEqual(len(generic["pairs"]), protocol.query_trials // 10)
            if protocol.n_items == 8:
                frozen = analyze_sampled_query_policy(
                    protocol, logits, seed=29, temperature=0.25
                )
                self.assertEqual(generic, frozen)

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

    def test_dedicated_runtime_configures_before_dispatch(self):
        with (
            patch("fsrl.item_count_runtime.configure_formal_runtime") as configure,
            patch("fsrl.item_count_transport.main", return_value=53) as workflow,
        ):
            from fsrl.item_count_runtime import main

            result = main(["--sentinel"])
        configure.assert_called_once_with()
        workflow.assert_called_once_with(["--sentinel"])
        self.assertEqual(result, 53)


if __name__ == "__main__":
    unittest.main()
