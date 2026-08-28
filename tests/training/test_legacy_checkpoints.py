import tempfile
import unittest
from pathlib import Path

import torch

from fsrl.core import RetroModelConfig, RetroModulRNN
from fsrl.training.legacy_checkpoints import (
    convert_legacy_checkpoint,
    legacy_checkpoint_format,
    load_frozen_retro_checkpoint,
    load_legacy_checkpoint_state,
    load_legacy_retro_checkpoint,
    resolve_frozen_checkpoint_path,
)


class LegacyCheckpointBoundaryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(97)
        self.config = RetroModelConfig(
            input_size=17, hidden_size=5, output_size=2, batch_size=2
        )
        self.net = RetroModulRNN(self.config, device="cpu")

    def test_legacy_boundary_accepts_only_dat(self):
        self.assertEqual(
            legacy_checkpoint_format("net.dat"),
            ("legacy_pytorch_state_dict", "legacy_read_only"),
        )
        with self.assertRaisesRegex(ValueError, "must end in .dat"):
            legacy_checkpoint_format("net.pth")

    def test_legacy_loader_reports_explicit_compatibility(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "net.dat"
            torch.save(self.net.state_dict(), path)
            state_dict, source_format, compatibility_mode = (
                load_legacy_checkpoint_state(path, device="cpu")
            )
            loaded, train_config, info = load_legacy_retro_checkpoint(
                path, batch_size=3, device="cpu"
            )
        self.assertEqual(source_format, "legacy_pytorch_state_dict")
        self.assertEqual(compatibility_mode, "legacy_read_only")
        self.assertEqual(info.source_format, source_format)
        self.assertEqual(info.compatibility_mode, compatibility_mode)
        self.assertEqual(train_config.bs, 3)
        for name, value in state_dict.items():
            self.assertTrue(torch.equal(value, loaded.state_dict()[name]))

    def test_frozen_resolver_prefers_pth_and_falls_back_to_dat(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy = root / "net.dat"
            legacy.write_bytes(b"legacy")
            self.assertEqual(resolve_frozen_checkpoint_path(root), legacy)
            canonical = root / "net.pth"
            canonical.write_bytes(b"canonical")
            self.assertEqual(resolve_frozen_checkpoint_path(root), canonical)

    def test_frozen_loader_dispatches_both_explicit_formats(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for suffix, expected_mode in (
                (".pth", "canonical"),
                (".dat", "legacy_read_only"),
            ):
                path = root / f"net{suffix}"
                torch.save(self.net.state_dict(), path)
                _, _, info = load_frozen_retro_checkpoint(
                    path, batch_size=2, device="cpu"
                )
                self.assertEqual(info.compatibility_mode, expected_mode)

    def test_one_way_conversion_is_byte_identical_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "net.dat"
            target = root / "net.pth"
            torch.save(self.net.state_dict(), source)
            result = convert_legacy_checkpoint(source, target)
            self.assertEqual(source.read_bytes(), target.read_bytes())
            self.assertEqual(
                result["transformation"], "byte_identity_extension_normalization"
            )
            with self.assertRaisesRegex(FileExistsError, "refuses overwrite"):
                convert_legacy_checkpoint(source, target)
