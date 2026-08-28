import copy
import csv
import json
import tempfile
import unittest
from itertools import combinations
from pathlib import Path

import numpy as np

from fsrl.experiments.global_policy.amplitude_provenance import NonInterpretableEstimate
from fsrl.experiments.global_policy.comparator_adequacy import (
    DEFAULT_SPECIFICATION_PATH,
    adequacy_statistics,
    apply_protocol_repair,
    bootstrap_counts,
    decide,
    distance_profiles,
    edge_metadata,
    load_trial_cohort,
    residualize,
    row_correlations,
    validate_prerequisite,
    vector_correlation,
)
from fsrl.infra.git_provenance import verify_git_registrations
from fsrl.infra.provenance import write_json_exclusive
from fsrl.infra.study_registry import legacy_identifier, resolve_record
from fsrl.paths import REPO_ROOT
from fsrl.tasks.registered_protocol import load_ranking_protocol

ROOT = REPO_ROOT


class GlobalPolicyComparatorAdequacyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        registered = json.loads(DEFAULT_SPECIFICATION_PATH.read_text())
        cls.specification, cls.repair = apply_protocol_repair(registered)
        protocol_path = resolve_record(
            cls.specification["registered_sources"]["liu_protocol"]["path"]
        )
        cls.protocol = load_ranking_protocol(protocol_path)
        cls.metadata = edge_metadata(cls.specification, cls.protocol)

    def test_repair_changes_only_the_registered_count_vector(self):
        self.assertFalse(self.repair["scientific_estimands_changed"])
        self.assertEqual(self.repair["registered_value"], [4, 5, 4, 3, 2, 2])
        self.assertEqual(self.repair["corrected_value"], [6, 5, 2, 3, 2, 2])

    def test_allocation_prerequisite_is_the_registered_structural_only_result(self):
        prerequisite = validate_prerequisite(self.specification)
        self.assertTrue(prerequisite["passed"])
        self.assertEqual(
            prerequisite["conditional_next_step"],
            "prospective_comparator_adequacy",
        )

    def test_implementation_source_lock_is_complete(self):
        specification = json.loads(DEFAULT_SPECIFICATION_PATH.read_text())
        lock = json.loads(
            resolve_record(
                "benchmarks/global_policy_comparator_adequacy_v1.lock.json"
            ).read_text()
        )
        validation = verify_git_registrations(
            ROOT,
            "44004aa39441e075b915f499aa9d02578c78e471",
            {
                **specification["registered_sources"],
                "audit_specification": {
                    "path": legacy_identifier(DEFAULT_SPECIFICATION_PATH),
                    "sha256": lock["audit_specification_sha256"],
                },
                "protocol_repair": lock["protocol_repair"],
                **lock["implementation_sources"],
                **lock["reused_frozen_sources"],
            },
        )
        self.assertTrue(validation["passed"])
        self.assertGreaterEqual(len(validation["checks"]), 20)

    def test_edge_contract_is_exactly_the_frozen_twenty_pair_design(self):
        self.assertEqual(len(self.metadata["selected_labels"]), 20)
        self.assertEqual(self.metadata["distance_counts"].tolist(), [6, 5, 2, 3, 2, 2])
        self.assertAlmostEqual(self.metadata["distance_mean"], 2.8)
        self.assertAlmostEqual(self.metadata["distance_denominator"], 57.2)
        self.assertAlmostEqual(float(np.sum(self.metadata["distance_weights"])), 0.0)

    def test_residualization_removes_only_intercept_and_linear_distance(self):
        values = 2.0 + 3.0 * self.metadata["selected_distances"]
        residual = residualize(values, self.metadata["residualizer"])
        np.testing.assert_allclose(residual, 0.0, atol=1e-12, rtol=0.0)
        nonlinear = values + self.metadata["selected_distances"] ** 2
        nonlinear_residual = residualize(nonlinear, self.metadata["residualizer"])
        self.assertGreater(float(np.linalg.norm(nonlinear_residual)), 0.0)

    def test_vector_and_row_correlations_match(self):
        first = np.linspace(-1.0, 1.0, 20)
        second = first**3
        point = vector_correlation(first, second, 1e-12)
        rows, norms = row_correlations(first, second[None, :], 1e-12)
        self.assertAlmostEqual(point, float(rows[0]))
        self.assertGreater(float(norms[0]), 1e-12)

    def test_degenerate_pair_vector_is_noninterpretable(self):
        with self.assertRaises(NonInterpretableEstimate):
            vector_correlation(np.ones(20), np.arange(20), 1e-12)

    def test_distance_profiles_keep_all_six_registered_levels(self):
        values = np.broadcast_to(self.metadata["selected_distances"], (3, 20)).astype(
            np.float64
        )
        profiles = distance_profiles(values, self.metadata)
        np.testing.assert_allclose(
            profiles,
            np.broadcast_to(np.arange(1.0, 7.0), (3, 6)),
            atol=0.0,
            rtol=0.0,
        )

    def test_bootstrap_counts_are_deterministic_and_reuse_all_subjects(self):
        specification = copy.deepcopy(self.specification)
        specification["statistical_contract"]["bootstrap_samples"] = 13
        first = bootstrap_counts(specification, 7)
        second = bootstrap_counts(specification, 7)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (13, 7))
        np.testing.assert_array_equal(np.sum(first, axis=1), np.full(13, 7))

    def test_all_four_substantive_decisions_are_mutually_exclusive(self):
        expected = {
            (True, True): "comparator_externally_adequate",
            (False, False): "comparator_externally_inadequate",
            (True, False): "distance_adequate_pair_inadequate",
            (False, True): "pair_adequate_distance_inadequate",
        }
        for (distance, pair), outcome in expected.items():
            statistics = {
                "primary": {
                    "distance": {"adequate": distance},
                    "pair": {"adequate": pair},
                }
            }
            self.assertEqual(decide(statistics, True)["outcome"], outcome)
        self.assertEqual(decide({}, False)["outcome"], "noninterpretable")

    def test_synthetic_statistics_use_one_human_bootstrap_and_fixed_posterior(self):
        specification = copy.deepcopy(self.specification)
        specification["statistical_contract"]["bootstrap_samples"] = 101
        distances = self.metadata["selected_distances"]
        pair_pattern = 0.015 * np.sin(np.arange(20))
        base = 0.58 + 0.025 * distances + pair_pattern
        subject_offset = np.linspace(-0.02, 0.02, 9)[:, None]
        half_noise = 0.003 * np.cos(np.arange(20))[None, :]
        odd = base + subject_offset + half_noise
        even = base + subject_offset - half_noise
        human = {
            "full": 0.5 * (odd + even),
            "odd": odd,
            "even": even,
            "participant_labels": [f"synthetic:{index}" for index in range(9)],
            "cohort_slices": {
                "preregistered": slice(0, 5),
                "replication": slice(5, 9),
            },
            "historical_all_pair_human_slope": 0.0,
        }
        posterior_rows = np.broadcast_to(base, (7, 20)).copy()
        posterior = {
            "subject_probability": posterior_rows,
            "cohort_probability": np.mean(posterior_rows, axis=0),
            "subject_slopes": posterior_rows @ self.metadata["distance_weights"],
            "historical_track_b_posterior_slope": 0.0,
        }
        statistics, integrity, raw = adequacy_statistics(
            specification, human, posterior, self.metadata
        )
        self.assertEqual(integrity["bootstrap_samples"], 101)
        self.assertEqual(len(raw["bootstrap"]["counts"]), 101)
        self.assertEqual(
            statistics["primary"]["pair"]["interval_rule"],
            "lower90_at_least_threshold",
        )
        self.assertEqual(len(statistics["secondary"]["distance_profile"]), 6)

    def test_synthetic_trial_loader_requires_every_pair_in_every_block(self):
        pairs = tuple(combinations(range(8), 2))
        pair_to_index = {pair: index for index, pair in enumerate(pairs)}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trials.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=(
                        "id",
                        "trial",
                        "block",
                        "film_choose_index",
                        "film_index_1",
                        "film_index_2",
                        "r_or_w",
                    ),
                )
                writer.writeheader()
                for subject in range(1, 41):
                    trial = 0
                    for block in range(1, 11):
                        for first, second in pairs:
                            trial += 1
                            writer.writerow(
                                {
                                    "id": subject,
                                    "trial": trial,
                                    "block": block,
                                    "film_choose_index": second + 1,
                                    "film_index_1": first + 1,
                                    "film_index_2": second + 1,
                                    "r_or_w": 1,
                                }
                            )
            values, labels = load_trial_cohort(path, "preregistered", pair_to_index, 8)
        self.assertEqual(values.shape, (40, 10, 28))
        self.assertEqual(len(labels), 40)
        self.assertTrue(np.all(values == 1.0))

    def test_exclusive_writer_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            write_json_exclusive(path, {"status": "first"})
            with self.assertRaises(FileExistsError):
                write_json_exclusive(path, {"status": "second"})
            self.assertEqual(json.loads(path.read_text()), {"status": "first"})
