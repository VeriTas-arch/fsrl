import unittest
from unittest.mock import patch

from fsrl.presentation_order_transport import (
    DEFAULT_SPECIFICATION_PATH,
    cross_cell_decision,
    load_json,
    schedule_integrity,
    transform_schedule,
)
from fsrl.ranking_protocol import load_ranking_protocol
from fsrl.study_registry import resolve_record


def _cell(*, interpretable=True, competence=True, all_pass=True):
    flags = {
        "intact_competence": competence,
        "constructive_global_structure": all_pass,
        "individualized_stable_structure": all_pass,
        "P_off_global_collapse": all_pass,
        "P_remote_reassembly": all_pass,
        "a_off_direct_loss": all_pass,
        "P_off_a_on_direct_nontransitive": all_pass,
        "exact_order_invariant_local_algorithm": all_pass,
    }
    return {
        "decision": {
            "interpretable": interpretable,
            "competence_passed": competence and interpretable,
            "all_eight_primary_links_pass": all(flags.values()) and interpretable,
            "flags": flags,
        }
    }


class PresentationOrderTransportTests(unittest.TestCase):
    def setUp(self):
        self.specification = load_json(DEFAULT_SPECIFICATION_PATH)
        self.protocol = load_ranking_protocol(resolve_record("benchmarks/liu_v2.json"))
        self.baseline = self.protocol.support_schedule(
            __import__("numpy").random.default_rng(11)
        )

    def test_schedule_transforms_preserve_trials_and_are_deterministic(self):
        schedules = {}
        for condition in self.specification["schedule_contract"][
            "conditions_in_execution_order"
        ]:
            first = transform_schedule(self.baseline, self.protocol, condition)
            second = transform_schedule(self.baseline, self.protocol, condition)
            self.assertEqual(first, second)
            self.assertTrue(
                schedule_integrity((self.baseline,), (first,), self.protocol)["passed"]
            )
            schedules[condition] = first
        self.assertEqual(schedules["reverse"][0].left_item, self.baseline[-1].left_item)
        relations = [
            (trial.higher_item, trial.lower_item)
            for trial in schedules["relation_clustered"]
        ]
        self.assertTrue(
            all(
                len(set(relations[index : index + 4])) == 1 for index in range(0, 32, 4)
            )
        )

    def test_cross_cell_outcomes_never_pool(self):
        conditions = ["blockwise_random", "relation_clustered", "reverse"]
        seed_ids = [2101, 2102, 2103]
        seeds = {
            str(seed): {"conditions": {name: _cell() for name in conditions}}
            for seed in seed_ids
        }
        self.assertEqual(
            cross_cell_decision(seeds, conditions, seed_ids)["outcome"],
            "LIU_PRESENTATION_ORDER_MECHANISM_TRANSPORTED",
        )
        seeds["2101"]["conditions"]["reverse"] = _cell(all_pass=False)
        self.assertEqual(
            cross_cell_decision(seeds, conditions, seed_ids)["outcome"],
            "ORDER_DEPENDENT_OR_UNRESOLVED",
        )
        seeds["2101"]["conditions"]["reverse"] = _cell(competence=False, all_pass=False)
        self.assertEqual(
            cross_cell_decision(seeds, conditions, seed_ids)["outcome"],
            "PRESENTATION_ORDER_COMPETENCE_NOT_ESTABLISHED",
        )
        seeds["2101"]["conditions"]["reverse"] = _cell(
            interpretable=False, competence=False, all_pass=False
        )
        self.assertEqual(
            cross_cell_decision(seeds, conditions, seed_ids)["outcome"],
            "NONINTERPRETABLE_EXECUTION",
        )

    def test_dedicated_runtime_configures_before_dispatch(self):
        with (
            patch(
                "fsrl.presentation_order_runtime.configure_formal_runtime"
            ) as configure,
            patch(
                "fsrl.presentation_order_transport.main", return_value=43
            ) as workflow,
        ):
            from fsrl.presentation_order_runtime import main

            result = main(["--sentinel"])
        configure.assert_called_once_with()
        workflow.assert_called_once_with(["--sentinel"])
        self.assertEqual(result, 43)


if __name__ == "__main__":
    unittest.main()
