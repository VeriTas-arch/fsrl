"""Versioned checkpoint loading shared by training and evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from fsrl.core.config import ADDINPUT, TrainConfig
from fsrl.infrastructure.provenance import file_sha256
from fsrl.infrastructure.runtime import default_device

from ..core.plastic_rnn import RetroModulRNN


@dataclass(frozen=True)
class CheckpointInfo:
    path: str
    sha256: str
    hidden_size: int
    cue_size: int
    input_size: int


def checkpoint_sha256(path: Path | str) -> str:
    return file_sha256(path)


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
    try:
        state_dict = torch.load(path, map_location=execution_device, weights_only=True)
    except TypeError:
        state_dict = torch.load(path, map_location=execution_device)
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
    net = RetroModulRNN(config.to_model_dict(), device=execution_device)
    net.load_state_dict(state_dict)
    net.eval()
    info = CheckpointInfo(
        path=str(path.resolve()),
        sha256=checkpoint_sha256(path),
        hidden_size=int(hidden_size),
        cue_size=cue_size,
        input_size=int(input_size),
    )
    return net, config, info
