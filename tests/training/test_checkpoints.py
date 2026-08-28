import tempfile
import unittest
from pathlib import Path

import torch

from fsrl.core import RetroModelConfig, RetroModulRNN
from fsrl.training.checkpoints import (
    checkpoint_format,
    load_checkpoint_state,
    load_retro_checkpoint,
    resolve_checkpoint_path,
)


class CheckpointBoundaryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(83)
        self.config = RetroModelConfig(
            input_size=17, hidden_size=5, output_size=2, batch_size=2
        )
        self.net = RetroModulRNN(self.config, device="cpu")

    def test_current_boundary_accepts_only_pth(self):
        self.assertEqual(
            checkpoint_format("net.pth"), ("pytorch_state_dict", "canonical")
        )
        with self.assertRaisesRegex(ValueError, "must end in"):
            checkpoint_format("net.dat")
        with self.assertRaisesRegex(ValueError, "must end in"):
            checkpoint_format("net.pt")

    def test_current_loader_rejects_dat_even_when_payload_is_valid(self):
        with tempfile.TemporaryDirectory() as directory:
            legacy = Path(directory) / "net.dat"
            torch.save(self.net.state_dict(), legacy)
            with self.assertRaisesRegex(ValueError, "must end in .pth"):
                load_checkpoint_state(legacy, device="cpu")

    def test_current_resolver_never_falls_back_to_dat(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "net.dat").write_bytes(b"legacy")
            self.assertEqual(resolve_checkpoint_path(root), root / "net.pth")

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

    def test_checkpoint_payload_must_be_plain_tensor_state_dict(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "net.pth"
            torch.save({"state_dict": self.net.state_dict()}, path)
            with self.assertRaisesRegex(TypeError, "tensor state_dict"):
                load_checkpoint_state(path, device="cpu")
