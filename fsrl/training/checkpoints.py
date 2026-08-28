"""Versioned checkpoint loading shared by training and evaluation."""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from fsrl.core.config import ADDINPUT, TrainConfig
from fsrl.core.model_config import RetroModelConfig
from fsrl.infra.provenance import file_sha256
from fsrl.infra.runtime import default_device

from ..core.plastic_rnn import RetroModulRNN


@dataclass(frozen=True)
class CheckpointInfo:
    schema_version: int
    path: str
    sha256: str
    source_format: str
    compatibility_mode: str
    hidden_size: int
    cue_size: int
    input_size: int


def checkpoint_sha256(path: Path | str) -> str:
    return file_sha256(path)


def checkpoint_format(path: Path | str) -> tuple[str, str]:
    """Classify canonical and read-only legacy checkpoint suffixes."""

    suffix = Path(path).suffix.lower()
    if suffix == ".pth":
        return "pytorch_state_dict", "canonical"
    if suffix == ".dat":
        return "legacy_pytorch_state_dict", "legacy_read_only"
    raise ValueError("checkpoint must end in .pth or legacy read-only .dat")


def resolve_checkpoint_path(directory: Path | str, basename: str = "net") -> Path:
    """Prefer a canonical checkpoint and fall back to one legacy input."""

    root = Path(directory)
    canonical = root / f"{basename}.pth"
    legacy = root / f"{basename}.dat"
    if canonical.is_file():
        return canonical
    if legacy.is_file():
        return legacy
    return canonical


def load_checkpoint_state(
    path: Path | str, *, device: str | torch.device | None = None
) -> tuple[dict[str, torch.Tensor], str, str]:
    """Normalize a canonical or legacy checkpoint to one state-dict boundary."""

    source = Path(path)
    source_format, compatibility_mode = checkpoint_format(source)
    execution_device = torch.device(device or default_device())
    payload = torch.load(source, map_location=execution_device, weights_only=True)
    if not isinstance(payload, Mapping) or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in payload.items()
    ):
        raise TypeError("checkpoint payload must be a tensor state_dict")
    state_dict = dict(payload)
    required = {"i2h.weight", "h2o.weight", "w", "alpha"}
    missing = sorted(required - state_dict.keys())
    if missing:
        raise ValueError(f"checkpoint state_dict is missing keys: {missing}")
    return state_dict, source_format, compatibility_mode


def convert_legacy_checkpoint(source: Path | str, target: Path | str) -> dict[str, Any]:
    """Materialize a one-way, byte-identical ``.dat`` to ``.pth`` view."""

    source_path = Path(source)
    target_path = Path(target)
    source_format, compatibility_mode = checkpoint_format(source_path)
    if source_format != "legacy_pytorch_state_dict":
        raise ValueError("legacy checkpoint conversion requires a .dat source")
    if target_path.suffix.lower() != ".pth":
        raise ValueError("legacy checkpoint conversion target must end in .pth")
    if target_path.exists() or target_path.is_symlink():
        raise FileExistsError(f"checkpoint conversion refuses overwrite: {target_path}")
    load_checkpoint_state(source_path, device="cpu")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_path, target_path)
    source_hash = checkpoint_sha256(source_path)
    target_hash = checkpoint_sha256(target_path)
    if source_hash != target_hash:
        raise RuntimeError("legacy checkpoint conversion changed source bytes")
    return {
        "passed": True,
        "source": str(source_path.resolve()),
        "target": str(target_path.resolve()),
        "source_format": source_format,
        "target_format": "pytorch_state_dict",
        "compatibility_mode": compatibility_mode,
        "transformation": "byte_identity_extension_normalization",
        "sha256": target_hash,
        "bytes": target_path.stat().st_size,
    }


def load_training_provenance(checkpoint: Path, checkpoint_hash: str) -> dict:
    metadata_path = checkpoint.parent / "config.json"
    if not metadata_path.is_file():
        return {"present": False}
    with metadata_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    registered_hash = metadata.get("checkpoint", {}).get("sha256")
    return {
        "present": True,
        "path": str(metadata_path.resolve()),
        "checkpoint_sha_matches": registered_hash == checkpoint_hash,
        "task_distribution": metadata.get("task_distribution"),
    }


def load_retro_checkpoint(
    path: Path | str,
    batch_size: int,
    *,
    device: str | torch.device | None = None,
) -> tuple[RetroModulRNN, TrainConfig, CheckpointInfo]:
    path = Path(path)
    execution_device = torch.device(device or default_device())
    state_dict, source_format, compatibility_mode = load_checkpoint_state(
        path, device=execution_device
    )
    hidden_size, input_size = state_dict["i2h.weight"].shape
    cue_remainder = int(input_size) - (1 + ADDINPUT + 2)
    if cue_remainder <= 0 or cue_remainder % 2:
        raise ValueError(f"Checkpoint input size {input_size} has no valid cue size")
    cue_size = cue_remainder // 2
    config = TrainConfig(
        bs=batch_size,
        hs=int(hidden_size),
        cs=cue_size,
        nbcues_min=8,
        nbcues_max=8,
    )
    model_config = RetroModelConfig(
        input_size=int(input_size),
        hidden_size=int(hidden_size),
        output_size=int(state_dict["h2o.weight"].shape[0]),
        batch_size=batch_size,
    )
    net = RetroModulRNN(model_config, device=execution_device)
    net.load_state_dict(state_dict)
    net.eval()
    info = CheckpointInfo(
        schema_version=1,
        path=str(path.resolve()),
        sha256=checkpoint_sha256(path),
        source_format=source_format,
        compatibility_mode=compatibility_mode,
        hidden_size=int(hidden_size),
        cue_size=cue_size,
        input_size=int(input_size),
    )
    return net, config, info
