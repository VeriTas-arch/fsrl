import unittest

from fsrl.infra.record_catalog import resolve_record_id
from fsrl.tasks.holdouts import registered_holdout_signatures
from fsrl.tasks.protocol_catalog import (
    PROTOCOL_DOCUMENT_IDS,
    PROTOCOL_RECORD_IDS,
    load_registered_protocol,
    protocol_path,
)


class ProtocolCatalogTests(unittest.TestCase):
    def test_protocols_resolve_by_stable_logical_id(self):
        for protocol_id, record_id in PROTOCOL_RECORD_IDS.items():
            with self.subTest(protocol_id=protocol_id):
                self.assertEqual(
                    protocol_path(protocol_id), resolve_record_id(record_id)
                )
                self.assertEqual(
                    load_registered_protocol(protocol_id).protocol_id,
                    PROTOCOL_DOCUMENT_IDS[protocol_id],
                )

    def test_holdout_set_is_derived_from_explicit_protocol_ids(self):
        signatures = registered_holdout_signatures()
        self.assertEqual(len(signatures), 2)
        self.assertEqual(
            signatures, registered_holdout_signatures(("liu_v1", "liu_v2"))
        )

    def test_unknown_protocol_never_falls_back_to_v1(self):
        with self.assertRaisesRegex(KeyError, "unknown registered protocol"):
            load_registered_protocol("liu_latest")
