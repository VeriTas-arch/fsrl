import unittest

from fsrl.infra.provenance import file_sha256
from fsrl.infra.record_catalog import (
    catalog_record,
    record_reference,
    resolve_record_id,
)


class RecordCatalogApiTests(unittest.TestCase):
    def test_stable_record_id_resolves_current_materialization(self):
        record_id = (
            "study.support_topology_transport."
            "benchmarks_liu_support_topology_transport_v1_json"
        )
        path = resolve_record_id(record_id)
        reference = record_reference(record_id)
        self.assertEqual(
            reference["repository_path"],
            "studies/support_topology_transport/records/benchmarks/"
            "liu_support_topology_transport_v1.json",
        )
        self.assertEqual(
            reference["materialized_identity"]["sha256"], file_sha256(path)
        )

    def test_historical_and_materialized_identities_are_explicit(self):
        record_id = "study.assembly_diagnostics.benchmarks_assembly_diagnostics_v1_json"
        reference = record_reference(record_id)
        self.assertNotEqual(
            reference["registered_identity"]["sha256"],
            reference["materialized_identity"]["sha256"],
        )
        self.assertEqual(
            reference["materialized_identity"]["sha256"],
            file_sha256(resolve_record_id(record_id)),
        )

    def test_git_blob_only_record_does_not_fake_a_current_path(self):
        record = catalog_record(
            "study.development_qualification."
            "checkpoints_dev_v2_seed1801_step1000_net_dat"
        )
        with self.assertRaisesRegex(FileNotFoundError, "not materialized"):
            resolve_record_id(record["record_id"])

    def test_unknown_record_id_fails_without_path_fallback(self):
        with self.assertRaisesRegex(KeyError, "unknown registered record ID"):
            resolve_record_id("study.unknown.results_missing_json")
