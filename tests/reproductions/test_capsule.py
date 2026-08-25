import unittest
from pathlib import Path

from fsrl.paths import REPO_ROOT
from reproductions.relational_learning_2024 import figures
from reproductions.relational_learning_2024.cli import parse_args
from reproductions.relational_learning_2024.verify import verify_manifest

ROOT = REPO_ROOT
CAPSULE = ROOT / "reproductions" / "relational_learning_2024"


class ReproductionCapsuleTests(unittest.TestCase):
    def test_upstream_and_supplied_checkpoints_match_manifest(self):
        result = verify_manifest()
        self.assertTrue(result["passed"])
        self.assertEqual(result["capsule_id"], "relational_learning_2024")
        self.assertEqual(result["files"], 11)

    def test_default_figure_checkpoint_resolves_inside_capsule(self):
        path = figures.resolve_model_path(None)
        self.assertEqual(path, CAPSULE / "checkpoints" / "net_active.dat")

    def test_training_outputs_default_to_ignored_artifacts(self):
        parsed = parse_args([])
        output = Path(parsed.output_dir)
        self.assertTrue(output.is_relative_to(ROOT / "artifacts"))

    def test_old_flat_reproduction_paths_are_absent(self):
        for path in (
            ROOT / "archive",
            ROOT / "addons",
            ROOT / "net_active.dat",
            ROOT / "net_passive.dat",
            ROOT / "eval_figures.py",
            ROOT / "simple_neo.py",
        ):
            self.assertFalse(path.exists(), path)
