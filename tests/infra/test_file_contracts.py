import gzip
import io
import json
import tempfile
import unittest
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

from fsrl.infra.file_contracts import (
    classify_path,
    dataset_file,
    find_unmanifested_run_roots,
    load_dataset_manifest,
    stable_record_id,
    validate_dataset_manifest,
    validate_payload,
    validate_run_manifest,
)
from fsrl.infra.record_catalog import CATALOG_PATH, check_record_catalog
from fsrl.paths import EXTERNAL_DATA_ROOT
from tools.provenance.backfill_run_manifests_v1 import run as backfill_runs
from tools.provenance.materialize_historical_file_views_v1 import (
    deterministic_float_npz,
    deterministic_gzip,
)


class FileContractTests(unittest.TestCase):
    def test_unmanifested_run_roots_are_not_invisible(self):
        with tempfile.TemporaryDirectory() as directory:
            runs_root = Path(directory) / "artifacts" / "runs"
            legacy_root = runs_root / "legacy-workflow"
            legacy_root.mkdir(parents=True)
            (legacy_root / "result.json").write_text("{}\n", encoding="utf-8")

            workflow_root = runs_root / "prospective-workflow"
            manifested = workflow_root / "execution-1"
            manifested.mkdir(parents=True)
            (manifested / "run.json").write_text("{}\n", encoding="utf-8")
            missing = workflow_root / "execution-2"
            missing.mkdir()
            (missing / "metrics.json").write_text("{}\n", encoding="utf-8")

            observed = find_unmanifested_run_roots(runs_root)

            self.assertEqual(observed, [legacy_root, missing])

    def test_compound_formats_and_legacy_checkpoint_are_explicit(self):
        self.assertEqual(classify_path("record.schema.json")["format"], "json_schema")
        self.assertEqual(classify_path("record.json.gz")["format"], "gzip_json")
        checkpoint = classify_path("net.dat")
        self.assertTrue(checkpoint["legacy_format"])
        self.assertEqual(checkpoint["prospective_suffix"], ".pth")
        self.assertEqual(classify_path("model.pth")["format"], "pytorch_state_dict")
        self.assertEqual(classify_path("program.pt")["format"], "pytorch_program")

    def test_stable_record_id_uses_the_immutable_legacy_locator(self):
        self.assertEqual(
            stable_record_id("study", "example", "results/example_v1.json"),
            "study.example.results_example_v1_json",
        )
        with self.assertRaisesRegex(ValueError, "safe relative"):
            stable_record_id("study", "example", "../result.json")

    def test_strict_json_and_checkpoint_envelopes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            invalid = root / "invalid.json"
            invalid.write_text('{"value": NaN}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-finite"):
                validate_payload(invalid)

            checkpoint = root / "model.pth"
            with zipfile.ZipFile(checkpoint, "w") as archive:
                archive.writestr("model/data.pkl", b"state")
            self.assertTrue(validate_payload(checkpoint)["passed"])

    def test_historical_views_are_deterministic_and_reversible(self):
        source = b'{"values": [1.0, 2.0]}\n'
        first = deterministic_gzip(source)
        self.assertEqual(first, deterministic_gzip(source))
        self.assertEqual(gzip.decompress(first), source)

        values = np.asarray([0.25, -0.125, 10.0], dtype=np.float64)
        first_npz = deterministic_float_npz(values)
        self.assertEqual(first_npz, deterministic_float_npz(values))
        with np.load(io.BytesIO(first_npz), allow_pickle=False) as archive:
            np.testing.assert_array_equal(archive["values"], values)

    def test_liu_dataset_manifest_owns_source_identity(self):
        path = EXTERNAL_DATA_ROOT / "liu2026" / "dataset.toml"
        result = validate_dataset_manifest(path)
        self.assertTrue(result["passed"], result["errors"])
        self.assertEqual(result["files"], 6)
        manifest = load_dataset_manifest(path)
        preregistered = dataset_file(manifest, "preregistered")
        self.assertEqual(preregistered["rows"], 11200)
        self.assertEqual(preregistered["participants"], 40)

    def test_registered_record_catalog_is_current_and_complete(self):
        result = check_record_catalog(CATALOG_PATH)
        self.assertTrue(result["passed"], result)
        catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
        historical = [
            record for record in catalog["records"] if record.get("origin") != "native"
        ]
        native_count = len(catalog["records"]) - len(historical)
        self.assertEqual(len(historical), 213)
        self.assertEqual(result["entries"], 213 + native_count)
        self.assertEqual(catalog["registered_record_count"], 211 + native_count)
        self.assertEqual(catalog["retired_asset_count"], 2)
        self.assertEqual(
            Counter(record["normalization"]["status"] for record in historical),
            {
                "already_conformant": 193,
                "historical_gzip_view": 19,
                "historical_pth_view": 1,
            },
        )

    def test_legacy_run_backfill_is_additive_and_rerunnable(self):
        with tempfile.TemporaryDirectory() as directory:
            repository_root = Path(directory)
            runs_root = repository_root / "artifacts" / "runs"
            run_root = runs_root / "example-workflow"
            run_root.mkdir(parents=True)
            source = run_root / "result.json"
            source.write_text('{"decision": "preserved"}\n', encoding="utf-8")
            original = source.read_bytes()

            first = backfill_runs(
                apply=True,
                runs_root=runs_root,
                repository_root=repository_root,
            )
            second = backfill_runs(
                apply=False,
                runs_root=runs_root,
                repository_root=repository_root,
            )
            self.assertTrue(first["passed"], first)
            self.assertEqual(first["written"], 1)
            self.assertTrue(second["passed"], second)
            self.assertEqual(source.read_bytes(), original)
            manifest = json.loads((run_root / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["file_count"], 1)
            self.assertEqual(manifest["files"][0]["path"], "result.json")
            self.assertEqual(
                manifest["conversion_contract"]["ownership"], "not_inferred"
            )
            validation = validate_run_manifest(run_root / "run.json")
            self.assertTrue(validation["passed"], validation["errors"])

            prospective = runs_root / "prospective" / "execution-1"
            prospective.mkdir(parents=True)
            (prospective / "run.json").write_text(
                '{"document_type": "fsrl.run_manifest"}\n', encoding="utf-8"
            )
            third = backfill_runs(
                apply=False,
                runs_root=runs_root,
                repository_root=repository_root,
            )
            self.assertTrue(third["passed"], third)
            self.assertEqual(third["manifests"], 1)


if __name__ == "__main__":
    unittest.main()
