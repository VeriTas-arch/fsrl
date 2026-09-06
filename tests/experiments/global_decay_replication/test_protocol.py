import unittest

from fsrl.experiments.global_decay_replication.protocol import (
    CONDITIONS,
    SEEDS,
    cohort_specification,
    resolved_specification,
    specification,
)


class GlobalDecayReplicationProtocolTests(unittest.TestCase):
    def test_frozen_matrix_and_unchanged_recipe(self):
        contract = specification()
        self.assertEqual(contract["frozen_execution"]["mandatory_seeds"], list(SEEDS))
        self.assertEqual(SEEDS, (2120, 2121, 2122))
        self.assertEqual(CONDITIONS, ("fixed_eta_resampled", "adaptive_eta_resampled"))
        spec = resolved_specification()
        self.assertEqual(spec["optimization"]["total_steps"], 1500)
        self.assertEqual(spec["optimization"]["batch_size"], 32)
        self.assertEqual(spec["task"]["support_blocks"], 4)

    def test_cohort_seed_ranges_are_disjoint(self):
        used = set()
        for index in range(400):
            settings = cohort_specification(index)["evaluation"]["liu"]
            groups = [
                [settings["cue_seed"]],
                range(settings["support_seed"], settings["support_seed"] + 77),
                [settings["subject_encoding_seed"]],
                [settings["encoding_seed"]],
                range(settings["choice_seed"], settings["choice_seed"] + 154),
            ]
            for group in groups:
                self.assertFalse(used.intersection(group))
                used.update(group)
        self.assertGreaterEqual(min(used), 5_200_000)
        self.assertLess(max(used), 5_800_000)

    def test_unregistered_cohorts_are_rejected(self):
        for index in (-1, 400):
            with self.assertRaises(ValueError):
                cohort_specification(index)


if __name__ == "__main__":
    unittest.main()
