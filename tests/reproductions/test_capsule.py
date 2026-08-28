import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from fsrl.core import RetroModelConfig, RetroModulRNN
from fsrl.core.config import TrainConfig
from fsrl.paths import REPO_ROOT
from fsrl.training.checkpoints import load_checkpoint_state
from reproductions.relational_learning_2024 import figures
from reproductions.relational_learning_2024.cli import parse_args
from reproductions.relational_learning_2024.training import save_checkpoint
from reproductions.relational_learning_2024.verify import verify_manifest

ROOT = REPO_ROOT
CAPSULE = ROOT / "reproductions" / "relational_learning_2024"


class ReproductionCapsuleTests(unittest.TestCase):
    def test_upstream_and_supplied_checkpoints_match_manifest(self):
        result = verify_manifest()
        self.assertTrue(result["passed"])
        self.assertEqual(result["capsule_id"], "relational_learning_2024")
        self.assertEqual(result["files"], 11)
        self.assertEqual(result["views"], 3)

    def test_checkpoint_views_are_byte_identical_and_canonical_loadable(self):
        for check in verify_manifest()["view_checks"]:
            source = CAPSULE / check["source"]
            view = CAPSULE / check["view"]
            self.assertEqual(source.read_bytes(), view.read_bytes())
            state_dict, source_format, compatibility_mode = load_checkpoint_state(
                view, device="cpu"
            )
            self.assertTrue(state_dict)
            self.assertEqual(source_format, "pytorch_state_dict")
            self.assertEqual(compatibility_mode, "canonical")

    def test_default_figure_checkpoint_resolves_inside_capsule(self):
        path = figures.resolve_model_path(None)
        self.assertEqual(path, CAPSULE / "checkpoints" / "net_active.pth")

    def test_training_outputs_default_to_ignored_artifacts(self):
        parsed = parse_args([])
        output = Path(parsed.output_dir)
        self.assertTrue(output.is_relative_to(ROOT / "artifacts"))

    def test_maintained_training_writes_only_canonical_binary_formats(self):
        config = TrainConfig(rngseed=7, bs=2, hs=4, cs=3)
        model_config = RetroModelConfig(
            input_size=config.inputsize,
            hidden_size=config.hs,
            output_size=config.outputsize,
            batch_size=config.bs,
        )
        torch.manual_seed(89)
        net = RetroModulRNN(model_config, device="cpu")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            save_checkpoint(config, net, output, [0.1, 0.2, 0.3])
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {"net.pth", "training_metrics.npz"},
            )
            with np.load(output / "training_metrics.npz", allow_pickle=False) as data:
                np.testing.assert_array_equal(data["episode"], [0])
                np.testing.assert_allclose(data["test_reward"], [0.1])

    def test_model_directory_rejects_legacy_dat(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy = root / "net.dat"
            legacy.write_bytes(b"legacy")
            with self.assertRaises(FileNotFoundError):
                figures.resolve_model_path_from_dir(root)
            with self.assertRaisesRegex(ValueError, "must end in .pth"):
                figures.resolve_model_path(legacy)
            canonical = root / "net.pth"
            canonical.write_bytes(b"canonical")
            self.assertEqual(figures.resolve_model_path_from_dir(root), canonical)

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
