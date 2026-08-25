import copy
import tempfile
import unittest
from pathlib import Path

import jsonschema

from fsrl.experiments.human.magnitude_collection import (
    PROTOCOL_PATH,
    RAW_SCHEMA_PATH,
    READINESS_PATH,
    REPAIR_PATH,
    build_manifest,
    build_synthetic_session_bundle,
    bytes_sha256,
    canonical_json_bytes,
    load_gzip_json,
    read_locked_session_bundle,
    render_trial_svg,
    run_readiness,
    validate_codebooks,
    validate_manifest,
    validate_session_bundle,
    write_gzip_json_exclusive,
    write_session_bundle_exclusive,
)
from fsrl.infra.provenance import file_sha256, load_json


class MagnitudePlacementCollectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = load_json(PROTOCOL_PATH)
        cls.readiness = load_json(READINESS_PATH)
        cls.raw_schema = load_json(RAW_SCHEMA_PATH)
        cls.manifest = build_manifest(cls.protocol, cls.readiness)

    def test_binary_codebooks_are_equal_weight_equidistant_and_dihedrally_unique(self):
        result = validate_codebooks(self.protocol)

        self.assertTrue(result["passed"])
        self.assertTrue(all(result["gates"].values()))
        self.assertEqual(result["cross_hamming_min"], 4)
        self.assertEqual(result["cross_hamming_max"], 12)
        self.assertEqual(
            result["distance_matrices"]["C1"],
            result["distance_matrices"]["C2"],
        )

    def test_full_120_slot_manifest_is_complete_and_deterministic(self):
        validation = validate_manifest(self.manifest, self.protocol, self.readiness)
        repeated = build_manifest(self.protocol, self.readiness)

        self.assertTrue(validation["passed"])
        self.assertTrue(all(validation["gates"].values()))
        self.assertEqual(validation["support_trials"], 7680)
        self.assertEqual(validation["query_trials"], 67200)
        self.assertEqual(
            bytes_sha256(canonical_json_bytes(self.manifest)),
            bytes_sha256(canonical_json_bytes(repeated)),
        )

    def test_renderer_is_label_free_and_frame_hash_is_locked(self):
        trial = self.manifest["slots"][0]["sessions"][0]["support_trials"][0]
        svg = render_trial_svg(trial, self.protocol, self.readiness)

        self.assertNotIn("<text", svg)
        self.assertNotIn("aria-label", svg)
        self.assertNotIn("data-role", svg)
        self.assertIn('shape-rendering="crispEdges"', svg)
        self.assertEqual(bytes_sha256(svg.encode("utf-8")), trial["renderer_sha256"])

    def test_gzip_manifest_serialization_is_reproducible_and_exclusive(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.json.gz"
            second = Path(directory) / "second.json.gz"
            write_gzip_json_exclusive(first, self.manifest)
            write_gzip_json_exclusive(second, self.manifest)

            self.assertEqual(file_sha256(first), file_sha256(second))
            self.assertEqual(load_gzip_json(first), self.manifest)
            with self.assertRaises(FileExistsError):
                write_gzip_json_exclusive(first, self.manifest)

    def test_synthetic_raw_bundle_matches_schema_manifest_and_write_once_lock(self):
        slot = self.manifest["slots"][0]
        session = slot["sessions"][0]
        manifest_hash = "0" * 64
        bundle = build_synthetic_session_bundle(
            slot,
            session,
            manifest_hash,
            self.protocol,
            self.readiness,
        )
        validation = validate_session_bundle(
            bundle,
            slot,
            session,
            self.protocol,
            self.readiness,
            self.raw_schema,
        )

        self.assertTrue(validation["passed"])
        self.assertTrue(all(validation["gates"].values()))
        jsonschema.Draft202012Validator.check_schema(self.raw_schema)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "session.json"
            written = write_session_bundle_exclusive(path, bundle)
            self.assertEqual(read_locked_session_bundle(path), bundle)
            self.assertEqual(written["sha256"], file_sha256(path))
            with self.assertRaises(FileExistsError):
                write_session_bundle_exclusive(path, bundle)

    def test_readiness_runner_rejects_non_synthetic_participant_id(self):
        slot = self.manifest["slots"][0]
        session = slot["sessions"][0]
        bundle = build_synthetic_session_bundle(
            slot,
            session,
            "0" * 64,
            self.protocol,
            self.readiness,
        )
        changed = copy.deepcopy(bundle)
        changed["participant_id"] = "MPBP-001"
        jsonschema.Draft202012Validator(self.raw_schema).validate(changed)

        validation = validate_session_bundle(
            changed,
            slot,
            session,
            self.protocol,
            self.readiness,
            self.raw_schema,
        )

        self.assertFalse(validation["passed"])
        self.assertFalse(validation["gates"]["synthetic_participant_only"])

    def test_readiness_runner_exercises_all_gates_but_keeps_collection_no_go(self):
        result = run_readiness(
            self.manifest,
            "0" * 64,
            self.protocol,
            self.readiness,
            load_json(REPAIR_PATH),
            self.raw_schema,
        )

        self.assertEqual(result["implementation_status"], "pass")
        self.assertEqual(result["collection_status"], "NO_GO")
        self.assertTrue(all(result["gates"].values()))
        self.assertFalse(result["dry_run"]["human_data_used"])
        self.assertEqual(set(result["external_go_requirements"].values()), {"pending"})
