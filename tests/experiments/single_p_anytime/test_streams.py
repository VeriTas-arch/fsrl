"""Max-first stream and prefix invariants."""

from __future__ import annotations

import unittest

import numpy as np

from fsrl.experiments.single_p_anytime.protocol import historical_recipe
from fsrl.experiments.single_p_anytime.streams import (
    generate_max_stream,
    make_generator,
    prefix_batch,
)


class AnytimeStreamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        task = historical_recipe(1)["task"]
        cls.generator = make_generator(task, support_blocks=6)

    def test_max_stream_is_regenerated_independently_of_prefix(self):
        first = generate_max_stream(
            self.generator,
            network_seed=9301,
            update=0,
            edge_count=7,
            batch_size=2,
        )
        prefix_batch(first.noisy, edge_count=7, blocks=2)
        second = generate_max_stream(
            self.generator,
            network_seed=9301,
            update=0,
            edge_count=7,
            batch_size=2,
        )
        self.assertEqual(first.task_fingerprint, second.task_fingerprint)
        self.assertEqual(first.noisy.fingerprint(), second.noisy.fingerprint())

    def test_prefix_is_an_exact_support_slice(self):
        stream = generate_max_stream(
            self.generator,
            network_seed=9301,
            update=1,
            edge_count=8,
            batch_size=2,
        )
        prefix = prefix_batch(stream.noisy, edge_count=8, blocks=3)
        np.testing.assert_array_equal(
            prefix.arrays["realized_q"], stream.noisy.arrays["realized_q"][:24]
        )
        np.testing.assert_array_equal(
            prefix.arrays["support_inputs"],
            stream.noisy.arrays["support_inputs"][:24],
        )
        self.assertEqual(stream.noisy.arrays["realized_q"].shape[0], 48)


if __name__ == "__main__":
    unittest.main()
