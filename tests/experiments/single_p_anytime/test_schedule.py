"""Structural tests for the frozen anytime chronology."""

from __future__ import annotations

import unittest

import numpy as np

from fsrl.experiments.single_p_anytime.schedule import (
    build_schedule,
    schedule_sha256,
    validate_schedule,
)


class AnytimeScheduleTests(unittest.TestCase):
    def test_every_macro_cycle_is_balanced_and_exposure_matched(self):
        summary = validate_schedule(build_schedule(3021))
        self.assertEqual(summary["shape"], [1500, 2])
        self.assertEqual(set(summary["counts"].values()), {75})
        self.assertEqual(summary["variable_exposure"], summary["fixed_exposure"])

    def test_schedule_is_deterministic_and_seed_specific(self):
        first = build_schedule(3021)
        np.testing.assert_array_equal(first, build_schedule(3021))
        self.assertNotEqual(
            schedule_sha256(first), schedule_sha256(build_schedule(3022))
        )


if __name__ == "__main__":
    unittest.main()
