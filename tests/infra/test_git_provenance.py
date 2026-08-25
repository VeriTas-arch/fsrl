import tempfile
import unittest
from pathlib import Path

from fsrl.infra.git_provenance import git_blob_sha256, verify_git_registrations
from fsrl.paths import REPO_ROOT

ROOT = REPO_ROOT
HISTORICAL_COMMIT = "44004aa39441e075b915f499aa9d02578c78e471"


class GitProvenanceTests(unittest.TestCase):
    def test_historical_blob_is_independent_of_working_tree(self):
        expected = "ce16de6a6ce4e7e2d8ec9a39e9e85fa93ed94199d4e0f1ac5c818a62a0ce10fe"
        self.assertEqual(
            git_blob_sha256(ROOT, HISTORICAL_COMMIT, "fsrl/formal_runtime.py"),
            expected,
        )

    def test_registration_mismatch_fails_closed(self):
        with self.assertRaisesRegex(RuntimeError, "historical Git source lock failed"):
            verify_git_registrations(
                ROOT,
                HISTORICAL_COMMIT,
                {
                    "formal_runtime": {
                        "path": "fsrl/formal_runtime.py",
                        "sha256": "0" * 64,
                    }
                },
            )

    def test_commit_and_path_must_be_exact_and_repository_relative(self):
        with self.assertRaisesRegex(RuntimeError, "full commit"):
            git_blob_sha256(ROOT, "44004aa", "fsrl/formal_runtime.py")
        with self.assertRaisesRegex(RuntimeError, "repository-relative"):
            git_blob_sha256(ROOT, HISTORICAL_COMMIT, "../formal_runtime.py")

    def test_repository_without_blob_fails_closed(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            self.assertRaisesRegex(RuntimeError, "unavailable"),
        ):
            git_blob_sha256(
                Path(directory), HISTORICAL_COMMIT, "fsrl/formal_runtime.py"
            )


if __name__ == "__main__":
    unittest.main()
