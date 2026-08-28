import tempfile
import unittest
from pathlib import Path

import torch

from fsrl.core import RetroModelConfig, RetroModulRNN
from fsrl.training.checkpoints import (
    checkpoint_format,
    convert_legacy_checkpoint,
    load_checkpoint_state,
    load_retro_checkpoint,
)


class CheckpointBoundaryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(83)
        self.config = RetroModelConfig(
            input_size=17, hidden_size=5, output_size=2, batch_size=2
        )
        self.net = RetroModulRNN(self.config, device="cpu")

    def test_pth_is_canonical_and_dat_is_read_only_legacy(self):
        self.assertEqual(
            checkpoint_format("net.pth"), ("pytorch_state_dict", "canonical")
        )
        self.assertEqual(
            checkpoint_format("net.dat"),
            ("legacy_pytorch_state_dict", "legacy_read_only"),
        )
        with self.assertRaisesRegex(ValueError, "must end in"):
            checkpoint_format("net.pt")

    def test_canonical_and_legacy_suffixes_normalize_to_same_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            canonical = root / "net.pth"
            legacy = root / "net.dat"
            torch.save(self.net.state_dict(), canonical)
            torch.save(self.net.state_dict(), legacy)
            canonical_state, canonical_format, canonical_mode = load_checkpoint_state(
                canonical, device="cpu"
            )
            legacy_state, legacy_format, legacy_mode = load_checkpoint_state(
                legacy, device="cpu"
            )
            for name in canonical_state:
                self.assertTrue(torch.equal(canonical_state[name], legacy_state[name]))
            self.assertEqual(canonical_format, "pytorch_state_dict")
            self.assertEqual(canonical_mode, "canonical")
            self.assertEqual(legacy_format, "legacy_pytorch_state_dict")
            self.assertEqual(legacy_mode, "legacy_read_only")

    def test_loader_reports_format_without_changing_model_state(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "net.pth"
            torch.save(self.net.state_dict(), path)
            loaded, train_config, info = load_retro_checkpoint(
                path, batch_size=3, device="cpu"
            )
        self.assertEqual(info.schema_version, 1)
        self.assertEqual(info.source_format, "pytorch_state_dict")
        self.assertEqual(info.compatibility_mode, "canonical")
        self.assertEqual(train_config.bs, 3)
        self.assertEqual(loaded.model_config.hidden_size, self.config.hidden_size)
        for name, value in self.net.state_dict().items():
            self.assertTrue(torch.equal(value, loaded.state_dict()[name]))

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

    def test_checkpoint_payload_must_be_plain_tensor_state_dict(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "net.pth"
            torch.save({"state_dict": self.net.state_dict()}, path)
            with self.assertRaisesRegex(TypeError, "tensor state_dict"):
                load_checkpoint_state(path, device="cpu")
