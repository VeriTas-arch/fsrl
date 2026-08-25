import unittest
from itertools import combinations

import numpy as np

from fsrl.experiments.transport.topology import (
    DEFAULT_SPECIFICATION_PATH,
    cross_cell_decision,
    enumerate_registered_graphs,
    graph_descriptor,
    protocol_for_graph,
    reconstruct_local_ledger,
    validate_graph_contract,
)
from fsrl.infra.provenance import load_json
from fsrl.infra.study_registry import resolve_record
from fsrl.tasks.registered_protocol import SupportTrial, load_ranking_protocol


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


class SupportTopologyTransportTests(unittest.TestCase):
    def setUp(self):
        self.specification = load_json(DEFAULT_SPECIFICATION_PATH)

    def test_registered_graphs_are_deterministic_matched_and_nonisomorphic(self):
        validation = validate_graph_contract(self.specification)
        self.assertTrue(validation["passed"])
        selected = enumerate_registered_graphs(self.specification)
        degrees = []
        for graph in self.specification["matched_graph_contract"]["graphs"]:
            edges = tuple(map(tuple, graph["rank_edges"]))
            self.assertEqual(edges, selected[graph["graph_id"]])
            descriptor = graph_descriptor(edges)
            self.assertTrue(descriptor["connected"])
            self.assertEqual(descriptor["edge_count"], 8)
            self.assertEqual(descriptor["distance_multiset"], [1, 2, 3, 3, 3, 4, 5, 7])
            degrees.append(tuple(descriptor["sorted_degree_sequence"]))
        self.assertEqual(len(set(degrees)), 3)
        self.assertNotIn((2,) * 8, degrees)

    def test_graph_protocol_changes_only_support_identity(self):
        base = load_ranking_protocol(resolve_record("benchmarks/liu_v2.json"))
        for graph in self.specification["matched_graph_contract"]["graphs"]:
            protocol = protocol_for_graph(base, graph)
            self.assertEqual(protocol.n_items, 8)
            self.assertEqual(protocol.support_trials, 32)
            self.assertEqual(protocol.query_trials, 280)
            self.assertEqual(
                protocol.true_order_high_to_low, base.true_order_high_to_low
            )
            rank = {
                item: position
                for position, item in enumerate(protocol.true_order_high_to_low)
            }
            distances = sorted(
                rank[lower] - rank[higher]
                for higher, lower in protocol.support_pairs_higher_lower
            )
            self.assertEqual(distances, [1, 2, 3, 3, 3, 4, 5, 7])

    def test_local_ledger_reconstructs_tensor_and_query_reads(self):
        rng = np.random.default_rng(7)
        codes = rng.choice([-1.0, 1.0], size=(2, 8, 5)).astype(np.float32)
        pairs = tuple(combinations(range(8), 2))

        def key(left, right):
            value = (np.outer(left, right) - np.outer(right, left)).reshape(-1)
            return value / max(np.linalg.norm(value), 1e-8)

        schedules = []
        scalars = np.empty((2, 8), dtype=np.float64)
        states = []
        reads = []
        for subject in range(2):
            trials = []
            state = np.zeros(25, dtype=np.float64)
            for index, pair in enumerate(pairs[:8]):
                left, right = pair if index % 2 == 0 else pair[::-1]
                scalar = float(rng.normal())
                scalars[subject, index] = scalar
                trials.append(SupportTrial(left, right, pair[0], pair[1], scalar, 0))
                state += scalar * key(codes[subject, left], codes[subject, right])
            schedules.append(tuple(trials))
            states.append(state)
            reads.append(
                [state @ key(codes[subject, a], codes[subject, b]) for a, b in pairs]
            )
        result = reconstruct_local_ledger(
            codes, tuple(schedules), scalars, np.asarray(states), np.asarray(reads)
        )
        self.assertLessEqual(result["tensor_state_max_abs_error"], 1e-12)
        self.assertLessEqual(result["ledger_tensor_state_max_abs_error"], 1e-12)
        self.assertLessEqual(result["all_query_raw_read_max_abs_error"], 1e-12)
        self.assertLessEqual(result["gpu_tensor_state_max_abs_error_diagnostic"], 1e-6)
        self.assertLessEqual(result["gpu_query_read_max_abs_error_diagnostic"], 1e-6)

    def test_cross_cell_outcomes_do_not_pool_graphs_or_backbones(self):
        graph_ids = ["g1", "g2", "g3"]
        seed_ids = [2101, 2102, 2103]
        seeds = {
            str(seed): {"graphs": {graph: _cell() for graph in graph_ids}}
            for seed in seed_ids
        }
        self.assertEqual(
            cross_cell_decision(seeds, graph_ids, seed_ids)["outcome"],
            "LIU_STRUCTURAL_MECHANISM_TRANSPORTED",
        )
        seeds["2101"]["graphs"]["g3"] = _cell(all_pass=False)
        self.assertEqual(
            cross_cell_decision(seeds, graph_ids, seed_ids)["outcome"],
            "TOPOLOGY_DEPENDENT_OR_UNRESOLVED",
        )
        for seed in seed_ids:
            for graph in graph_ids:
                seeds[str(seed)]["graphs"][graph] = _cell(all_pass=False)
        self.assertEqual(
            cross_cell_decision(seeds, graph_ids, seed_ids)["outcome"],
            "FUNCTIONAL_ASYMMETRY_NOT_TRANSPORTED",
        )
        seeds["2102"]["graphs"]["g2"] = _cell(competence=False, all_pass=False)
        self.assertEqual(
            cross_cell_decision(seeds, graph_ids, seed_ids)["outcome"],
            "STRUCTURAL_COMPETENCE_NOT_ESTABLISHED",
        )
        seeds["2102"]["graphs"]["g2"] = _cell(
            interpretable=False, competence=False, all_pass=False
        )
        self.assertEqual(
            cross_cell_decision(seeds, graph_ids, seed_ids)["outcome"],
            "NONINTERPRETABLE_EXECUTION",
        )


if __name__ == "__main__":
    unittest.main()
