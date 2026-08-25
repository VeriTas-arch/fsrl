import copy
import unittest

import numpy as np

from fsrl.experiments.human.magnitude_analysis import (
    PAIRS,
    adjusted_ols,
    analyze_profiles,
    classify_outcome,
    hodge_rank_positions,
    participant_estimands,
    profile_from_session_bundles,
    synthetic_profiles,
    validate_all_synthetic_branches,
)
from fsrl.experiments.human.magnitude_collection import (
    PROTOCOL_PATH,
    READINESS_PATH,
    build_manifest,
    build_synthetic_session_bundle,
)
from fsrl.infra.provenance import load_json


class MagnitudePlacementAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = load_json(PROTOCOL_PATH)
        cls.readiness = load_json(READINESS_PATH)

    def test_all_five_registered_outcome_branches_are_recovered(self):
        validation = validate_all_synthetic_branches(self.protocol)

        self.assertTrue(validation["passed"])
        self.assertTrue(all(validation["gates"].values()))

    def test_synthetic_profiles_recover_exact_registered_estimands(self):
        expected = {
            "metric_enters_global_order_construction": {
                "delta_flip": 0.20,
                "beta_conf": 0.0,
                "beta_learned": 0.0,
            },
            "ordinal_global_order_with_metric_confidence_modulation": {
                "delta_flip": 0.0,
                "beta_conf": 0.05,
                "beta_learned": 0.0,
            },
            "metric_retained_locally_not_globally": {
                "delta_flip": 0.0,
                "beta_conf": 0.0,
                "beta_learned": 0.05,
            },
        }
        for outcome, target in expected.items():
            profile = synthetic_profiles(outcome, self.protocol)[0]
            observed = participant_estimands(profile, self.protocol)
            for name, value in target.items():
                self.assertAlmostEqual(observed[name], value, places=12)

    def test_equivalence_has_priority_over_tiny_directional_flag(self):
        equivalent_positive = {
            "flags": {
                "directional_positive": True,
                "directional_negative": False,
                "equivalent_to_zero": True,
            }
        }
        equivalent_zero = {
            "flags": {
                "directional_positive": False,
                "directional_negative": False,
                "equivalent_to_zero": True,
            }
        }
        results = {
            "delta_flip": equivalent_positive,
            "beta_conf": equivalent_zero,
            "beta_learned": equivalent_zero,
        }

        self.assertEqual(
            classify_outcome(results), "stronger_ordinalization_or_metric_loss"
        )

    def test_effect_coded_ols_intercept_is_balanced_grand_mean(self):
        cells = [
            "A_first/C1_to_A",
            "A_first/C1_to_B",
            "B_first/C1_to_A",
            "B_first/C1_to_B",
        ] * 25
        values = np.asarray(
            [
                0.20
                + (0.03 if cell.startswith("A_first") else -0.03)
                + (0.02 if cell.endswith("C1_to_A") else -0.02)
                for cell in cells
            ]
        )
        result = adjusted_ols(values, cells, 0.10)

        self.assertAlmostEqual(result["point"], 0.20)
        self.assertAlmostEqual(result["coefficients"]["condition_order"], 0.03)
        self.assertAlmostEqual(
            result["coefficients"]["codebook_condition_mapping"], 0.02
        )
        self.assertTrue(result["flags"]["directional_positive"])
        self.assertFalse(result["flags"]["equivalent_to_zero"])

    def test_incomplete_or_below_chance_cohort_is_noninterpretable(self):
        profiles = synthetic_profiles(
            "stronger_ordinalization_or_metric_loss", self.protocol
        )
        profiles[0] = copy.deepcopy(profiles[0])
        profiles[0]["complete"] = False

        result = analyze_profiles(profiles, self.protocol)

        self.assertEqual(result["outcome"], "noninterpretable")
        self.assertEqual(result["analyzable_participants"], 99)
        self.assertEqual(result["exclusions"]["incomplete"], 1)

    def test_raw_dry_run_sessions_import_to_one_complete_profile(self):
        manifest = build_manifest(self.protocol, self.readiness)
        slot = manifest["slots"][0]
        bundles = [
            build_synthetic_session_bundle(
                slot,
                session,
                "0" * 64,
                self.protocol,
                self.readiness,
            )
            for session in slot["sessions"]
        ]

        profile = profile_from_session_bundles(bundles)

        self.assertTrue(profile["complete"])
        self.assertEqual(set(profile["pair_probability"]), {"A", "B"})
        self.assertEqual(set(profile["pair_probability"]["A"]), set(PAIRS))
        self.assertEqual(set(profile["pair_probability"]["B"]), set(PAIRS))

    def test_hodge_rank_recovers_a_perfect_complete_order(self):
        order = list(
            reversed(
                self.protocol["inherited_frozen_contract"]["assignment_A_low_to_high"]
            )
        )
        positions = {role: index for index, role in enumerate(order)}
        field = {}
        for pair in PAIRS:
            first, second = pair.split("-")
            field[pair] = float(positions[first] < positions[second])

        observed = hodge_rank_positions(field)

        self.assertEqual([int(value) for value in observed], [7, 6, 5, 4, 3, 2, 1, 0])


if __name__ == "__main__":
    unittest.main()
