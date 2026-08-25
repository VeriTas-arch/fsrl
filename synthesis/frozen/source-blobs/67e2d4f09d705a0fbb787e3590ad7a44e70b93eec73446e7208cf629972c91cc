import unittest
from unittest.mock import patch

import numpy as np

from fsrl.evidence_sparsity_transport import (
    DEFAULT_SPECIFICATION_PATH,
    build_nested_schedules,
    cross_cell_decision,
    density_trend_metrics,
    family_protocols,
    load_json,
    validate_sparsity_contract,
)
from fsrl.ranking_protocol import load_ranking_protocol


def _cell(*, interpretable=True, competence=True, all_pass=True):
    flags = {
        "intact_competence": competence,
        "constructive_global_structure": all_pass,
        "individualized_stable_structure": all_pass,
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
            "all_eight_primary_links_pass": all(flags.values()) and interpretable,
            "flags": flags,
        }
    }


def _trend(passed=True):
    return {
        "flags": {
            "global_dependence_decreases": passed,
            "local_dependence_increases": passed,
        },
        "bidirectional_prediction_passed": passed,
    }


class EvidenceSparsityTransportTests(unittest.TestCase):
    def setUp(self):
        self.specification = load_json(DEFAULT_SPECIFICATION_PATH)
        self.base = load_ranking_protocol("benchmarks/liu_v2.json")

    def test_registered_nested_graphs_are_deterministic_and_distance_matched(self):
        result = validate_sparsity_contract(self.specification)
        self.assertTrue(result["passed"])
        self.assertTrue(result["distance_multisets_matched_between_families"])

    def test_nested_schedules_preserve_every_common_physical_trial(self):
        for family_index, family in enumerate(
            self.specification["matched_nested_graph_contract"]["families"], start=1
        ):
            protocols = family_protocols(self.base, family)
            base_schedules = tuple(
                self.base.support_schedule(np.random.default_rng(11 + subject))
                if family_index == 1
                else protocols[8].support_schedule(np.random.default_rng(11 + subject))
                for subject in range(2)
            )
            schedules = build_nested_schedules(
                base_schedules, self.base, protocols, family, family_index
            )
            self.assertEqual(schedules[8], base_schedules)
            for edge_count, protocol in protocols.items():
                self.assertTrue(
                    all(
                        len(schedule) == 4 * edge_count
                        for schedule in schedules[edge_count]
                    )
                )
                for relation in protocol.support_pairs_higher_lower:
                    reference = tuple(
                        trial
                        for trial in schedules[edge_count][0]
                        if (trial.higher_item, trial.lower_item) == relation
                    )
                    self.assertEqual(len(reference), 4)
                    for other_count, other_protocol in protocols.items():
                        if relation in other_protocol.support_pairs_higher_lower:
                            candidate = tuple(
                                trial
                                for trial in schedules[other_count][0]
                                if (trial.higher_item, trial.lower_item) == relation
                            )
                            self.assertEqual(candidate, reference)

    def test_density_slopes_use_paired_participants(self):
        cells = {}
        for edge_count in (7, 8, 9, 10):
            global_values = np.asarray([0.4, 0.2]) - 0.01 * edge_count
            local_values = np.asarray([0.0, 0.1]) + 0.02 * edge_count
            cells[str(edge_count)] = {
                "metrics": {
                    "density_dependencies": {
                        "global_dependence_all_pairs": {
                            "raw_subject": global_values.tolist()
                        },
                        "local_dependence_all_pairs": {
                            "raw_subject": local_values.tolist()
                        },
                    }
                }
            }
        specification = {
            "matched_nested_graph_contract": {
                "edge_counts_in_execution_order": [7, 8, 9, 10]
            },
            "evaluation": {
                "bootstrap_samples": 100,
                "subjects_per_cell": 2,
                "bootstrap_interval": 0.95,
            },
        }
        result = density_trend_metrics(cells, 1, 1, specification)
        self.assertAlmostEqual(
            result["summary"]["global_dependence_all_pairs"]["mean"], -0.01
        )
        self.assertAlmostEqual(
            result["summary"]["local_dependence_all_pairs"]["mean"], 0.02
        )
        self.assertTrue(result["bidirectional_prediction_passed"])

    def test_cross_cell_outcomes_never_pool(self):
        families = ["f1", "f2"]
        edge_counts = [7, 8, 9, 10]
        seed_ids = [2101, 2102, 2103]
        seeds = {
            str(seed): {
                "families": {
                    family: {
                        "densities": {
                            str(edge_count): _cell() for edge_count in edge_counts
                        },
                        "density_trend": _trend(),
                    }
                    for family in families
                }
            }
            for seed in seed_ids
        }
        self.assertEqual(
            cross_cell_decision(seeds, families, edge_counts, seed_ids)["outcome"],
            "LIU_SPARSITY_MECHANISM_TRANSPORTED",
        )
        seeds["2101"]["families"]["f1"]["densities"]["7"] = _cell(all_pass=False)
        self.assertEqual(
            cross_cell_decision(seeds, families, edge_counts, seed_ids)["outcome"],
            "SPARSITY_DEPENDENT_OR_UNRESOLVED",
        )
        seeds["2101"]["families"]["f1"]["densities"]["7"] = _cell(
            competence=False, all_pass=False
        )
        self.assertEqual(
            cross_cell_decision(seeds, families, edge_counts, seed_ids)["outcome"],
            "SPARSITY_COMPETENCE_NOT_ESTABLISHED",
        )
        seeds["2101"]["families"]["f1"]["densities"]["7"] = _cell(
            interpretable=False, competence=False, all_pass=False
        )
        self.assertEqual(
            cross_cell_decision(seeds, families, edge_counts, seed_ids)["outcome"],
            "NONINTERPRETABLE_EXECUTION",
        )

    def test_dedicated_runtime_configures_before_dispatch(self):
        with (
            patch(
                "fsrl.evidence_sparsity_runtime.configure_formal_runtime"
            ) as configure,
            patch("fsrl.evidence_sparsity_transport.main", return_value=47) as workflow,
        ):
            from fsrl.evidence_sparsity_runtime import main

            result = main(["--sentinel"])
        configure.assert_called_once_with()
        workflow.assert_called_once_with(["--sentinel"])
        self.assertEqual(result, 47)


if __name__ == "__main__":
    unittest.main()
