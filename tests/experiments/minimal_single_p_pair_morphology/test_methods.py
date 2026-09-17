import unittest
from itertools import permutations

import numpy as np

from fsrl.analysis.hodge import build_complete_graph_geometry
from fsrl.experiments.minimal_single_p_pair_morphology.methods import (
    hodge_components,
    panel_stage,
    sample_pair_accuracies,
    study_outcome,
)
from fsrl.tasks.protocol import RankingProtocol


class PairMorphologyMethodTests(unittest.TestCase):
    def protocol(self):
        return RankingProtocol(
            protocol_id="test",
            item_labels=("A", "B", "C"),
            true_order_high_to_low=(0, 1, 2),
            support_pairs_higher_lower=((0, 1), (1, 2)),
            support_blocks=1,
            query_blocks=10,
            human_targets={},
        )

    def test_choice_replay_is_deterministic_and_correct(self):
        logits = {
            pair: (100.0 if pair[0] < pair[1] else -100.0)
            for pair in permutations(range(3), 2)
        }
        observed = sample_pair_accuracies(
            self.protocol(), (logits,), seed=4, temperature=0.25
        )
        np.testing.assert_array_equal(observed, np.ones((1, 3)))

    def test_hodge_components_reconstruct_and_are_orthogonal(self):
        geometry = build_complete_graph_geometry(self.protocol())
        field = np.asarray([[1.0, -0.5, 0.2]])
        gradient, residual = hodge_components(field, geometry)
        np.testing.assert_allclose(gradient + residual, field, atol=1e-12)
        np.testing.assert_allclose(np.sum(gradient * residual), 0.0, atol=1e-12)

    def test_stage_hierarchy_and_study_rule(self):
        self.assertEqual(panel_stage(14, 20, 20, 20), "direction_absent")
        self.assertEqual(panel_stage(15, 14, 20, 20), "weak_margin")
        self.assertEqual(panel_stage(15, 15, 14, 20), "latent_shape")
        self.assertEqual(panel_stage(15, 15, 15, 14), "finite_sampling")
        self.assertEqual(panel_stage(15, 15, 15, 15), "already_latent_and_sampled")
        self.assertEqual(study_outcome({"weak_margin": 40}), "weak_margin")
        self.assertEqual(
            study_outcome({"weak_margin": 39, "latent_shape": 21}),
            "mixed_or_unidentified",
        )


if __name__ == "__main__":
    unittest.main()
