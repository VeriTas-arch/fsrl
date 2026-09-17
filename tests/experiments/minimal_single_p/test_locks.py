import unittest
from unittest.mock import patch

from fsrl.experiments.minimal_single_p import training


class MinimalSinglePLockTests(unittest.TestCase):
    def test_training_cannot_start_before_source_lock(self):
        with (
            patch.object(
                training,
                "validate_source_lock",
                side_effect=RuntimeError("source lock"),
            ),
            patch.object(training, "train_one") as train_one,
        ):
            with self.assertRaisesRegex(RuntimeError, "source lock"):
                training.train_all("C0")
            train_one.assert_not_called()


if __name__ == "__main__":
    unittest.main()
