import unittest

from fsrl.experiments.common_encoding_state.protocol import (
    CONDITIONS,
    EXECUTION_ORDER,
    RHO_GRID,
    SEEDS,
    resolved_specification,
    specification,
)


class ProtocolTests(unittest.TestCase):
    def test_candidate_and_controls_are_fixed(self):
        self.assertEqual(
            CONDITIONS, ("independent", "relation_common", "episode_common")
        )
        self.assertEqual(RHO_GRID, (0.125, 0.25, 0.5))
        self.assertEqual(SEEDS, (2123, 2124, 2125))
        self.assertEqual(set(EXECUTION_ORDER), set(SEEDS))

    def test_resolved_spec_keeps_two_scalars_and_one_stage(self):
        registered = specification()
        spec = resolved_specification()
        self.assertEqual(
            spec["optimization"]["trainable_parameters"],
            ["raw_eta", "raw_global_gain"],
        )
        self.assertEqual(spec["optimization"]["total_steps"], 1500)
        self.assertEqual(
            registered["frozen_learner"]["trainable_scalars"], ["eta0", "gamma_G"]
        )


if __name__ == "__main__":
    unittest.main()
