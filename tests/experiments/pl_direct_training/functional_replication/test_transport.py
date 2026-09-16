import unittest
from itertools import combinations

import numpy as np

from fsrl.experiments.pl_direct_training.functional_replication.transport import (
    _packed_keys,
    reconstruct_packed_ledger,
)
from fsrl.tasks.protocol import SupportTrial


class FunctionalTransportTests(unittest.TestCase):
    def test_packed_ledger_reconstruction_is_exact_in_float64(self):
        rng = np.random.default_rng(101)
        item_codes = rng.normal(size=(1, 6, 15))
        keys = _packed_keys(item_codes[0])
        pairs = tuple(combinations(range(6), 2))
        schedule = tuple(
            SupportTrial(
                left_item=second if index % 2 else first,
                right_item=first if index % 2 else second,
                higher_item=first,
                lower_item=second,
                signed_magnitude=-0.2 if index % 2 else 0.2,
                block_index=0,
            )
            for index, (first, second) in enumerate(pairs[:6])
        )
        scalars = np.asarray(
            [[trial.signed_magnitude for trial in schedule]], dtype=np.float64
        )
        state = np.zeros((1, keys.shape[1]), dtype=np.float64)
        for scalar, trial in zip(scalars[0], schedule, strict=True):
            pair = tuple(sorted((trial.left_item, trial.right_item)))
            orientation = 1.0 if trial.left_item < trial.right_item else -1.0
            state[0] += scalar * orientation * keys[pairs.index(pair)]
        reads = state @ keys.T
        result = reconstruct_packed_ledger(
            item_codes, (schedule,), scalars, state, reads
        )
        self.assertLessEqual(result["tensor_state_max_abs_error"], 1e-12)
        self.assertLessEqual(result["all_query_raw_read_max_abs_error"], 1e-12)
        self.assertEqual(result["gpu_tensor_state_max_abs_error_diagnostic"], 0.0)
        self.assertLessEqual(result["gpu_query_read_max_abs_error_diagnostic"], 1e-12)


if __name__ == "__main__":
    unittest.main()
