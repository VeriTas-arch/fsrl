"""Explicit read-only adapters for historical ``.dat`` checkpoints."""

from __future__ import annotations

import shutil
from pathlib import Path

import torch

from fsrl.core.config import TrainConfig
from fsrl.infra.runtime import default_device

from ..core.plastic_rnn import RetroModulRNN
from .checkpoints import (
    CheckpointInfo,
    _load_tensor_state_dict,
    _restore_retro_checkpoint,
    checkpoint_sha256,
    load_retro_checkpoint,
)


def legacy_checkpoint_format(path: Path | str) -> tuple[str, str]:
    """Classify one historical checkpoint without widening the current API."""

    if Path(path).suffix.lower() == ".dat":
        return "legacy_pytorch_state_dict", "legacy_read_only"
    raise ValueError("legacy checkpoints must end in .dat")


def resolve_frozen_checkpoint_path(
    directory: Path | str, basename: str = "net"
) -> Path:
    """Prefer a derived ``.pth`` view and fall back to frozen ``.dat`` evidence."""

    root = Path(directory)
    canonical = root / f"{basename}.pth"
    legacy = root / f"{basename}.dat"
    if canonical.is_file():
        return canonical
    if legacy.is_file():
        return legacy
    return canonical


def load_legacy_checkpoint_state(
    path: Path | str, *, device: str | torch.device | None = None
) -> tuple[dict[str, torch.Tensor], str, str]:
    """Load a tensor state dict through the explicit historical boundary."""

    source = Path(path)
    source_format, compatibility_mode = legacy_checkpoint_format(source)
    execution_device = torch.device(device or default_device())
    state_dict = _load_tensor_state_dict(source, execution_device)
    return state_dict, source_format, compatibility_mode


def load_legacy_retro_checkpoint(
    path: Path | str,
    batch_size: int,
    *,
    device: str | torch.device | None = None,
) -> tuple[RetroModulRNN, TrainConfig, CheckpointInfo]:
    """Restore one historical ``.dat`` checkpoint for frozen replay only."""

    source = Path(path)
    execution_device = torch.device(device or default_device())
    state_dict, source_format, compatibility_mode = load_legacy_checkpoint_state(
        source, device=execution_device
    )
    return _restore_retro_checkpoint(
        source,
        state_dict,
        batch_size,
        execution_device=execution_device,
        source_format=source_format,
        compatibility_mode=compatibility_mode,
    )


def load_frozen_retro_checkpoint(
    path: Path | str,
    batch_size: int,
    *,
    device: str | torch.device | None = None,
) -> tuple[RetroModulRNN, TrainConfig, CheckpointInfo]:
    """Load canonical views or historical inputs for frozen evaluation."""

    source = Path(path)
    if source.suffix.lower() == ".pth":
        return load_retro_checkpoint(source, batch_size, device=device)
    return load_legacy_retro_checkpoint(source, batch_size, device=device)


def convert_legacy_checkpoint(
    source: Path | str, target: Path | str
) -> dict[str, object]:
    """Materialize a one-way, byte-identical ``.dat`` to ``.pth`` view."""

    source_path = Path(source)
    target_path = Path(target)
    source_format, compatibility_mode = legacy_checkpoint_format(source_path)
    if target_path.suffix.lower() != ".pth":
        raise ValueError("legacy checkpoint conversion target must end in .pth")
    if target_path.exists() or target_path.is_symlink():
        raise FileExistsError(f"checkpoint conversion refuses overwrite: {target_path}")
    load_legacy_checkpoint_state(source_path, device="cpu")
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
