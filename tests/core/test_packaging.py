import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

from fsrl.paths import REPO_ROOT


class PackagingTests(unittest.TestCase):
    def test_wheel_contains_code_and_required_internal_contracts_only(self):
        with tempfile.TemporaryDirectory(prefix="fsrl-wheel-") as temporary:
            root = Path(temporary)
            source = root / "source"
            distribution = root / "dist"
            source.mkdir()
            distribution.mkdir()
            shutil.copy2(REPO_ROOT / "pyproject.toml", source)
            shutil.copy2(REPO_ROOT / "README.md", source)
            shutil.copytree(
                REPO_ROOT / "fsrl",
                source / "fsrl",
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
            environment = {**os.environ, "PIP_NO_CACHE_DIR": "1"}
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "wheel",
                    "--no-deps",
                    "--no-build-isolation",
                    "--wheel-dir",
                    str(distribution),
                    str(source),
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=60,
                env=environment,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            wheels = list(distribution.glob("*.whl"))
            self.assertEqual(len(wheels), 1)
            with zipfile.ZipFile(wheels[0]) as archive:
                names = set(archive.namelist())
                installed = root / "installed"
                archive.extractall(installed)

            import_check = subprocess.run(
                [
                    sys.executable,
                    "-P",
                    "-c",
                    (
                        "import fsrl, fsrl.analysis, fsrl.core, fsrl.evaluation, "
                        "fsrl.experiments, fsrl.infra, fsrl.tasks, fsrl.training, "
                        "fsrl.workflows; print(fsrl.__file__)"
                    ),
                ],
                cwd=root,
                check=False,
                capture_output=True,
                text=True,
                env={
                    **environment,
                    "PYTHONNOUSERSITE": "1",
                    "PYTHONPATH": str(installed),
                },
            )
            self.assertEqual(import_check.returncode, 0, import_check.stderr)
            self.assertIn(str(installed), import_check.stdout)

        required = {
            "fsrl/experiments/transport/contracts/item_count_within_cell_v1.json",
            "fsrl/experiments/transport/contracts/topology_within_cell_v1.json",
        }
        self.assertTrue(required <= names)
        forbidden = ("studies/", "synthesis/", "workflows/", "data/", "artifacts/")
        self.assertFalse(any(name.startswith(forbidden) for name in names))
