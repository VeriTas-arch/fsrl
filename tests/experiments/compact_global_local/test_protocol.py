from __future__ import annotations

import unittest

from fsrl.experiments.compact_global_local.protocol import load_specification


class CompactProtocolTests(unittest.TestCase):
    def test_append_only_repair_controls_trial_lengths(self):
        specification = load_specification()
        architecture = specification["architecture"]
        self.assertEqual(architecture["support_trial_steps"], 3)
        self.assertEqual(architecture["query_steps"], 2)
        self.assertEqual(architecture["blank_initial_steps"], 0)
        self.assertEqual(
            specification["active_repair"]["repair_id"],
            "compact-global-local-model-v1-repair1",
        )


if __name__ == "__main__":
    unittest.main()
