"""Canonical ``.pth`` checkpoint loading for maintained execution."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch

from fsrl.core.config import ADDINPUT, TrainConfig
from fsrl.core.model_config import RetroModelConfig
from fsrl.infra.provenance import file_sha256
from fsrl.infra.runtime import default_device
from fsrl.tasks.holdouts import registered_holdout_signatures

from ..core.plastic_rnn import RetroModulRNN
from .backbone import COMPILED_TRAINING_EXECUTION

FORMAL_CONFIRMATION_ID = "liu-neural-constructive-ranking-confirmation-v1"


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
    """Classify one canonical checkpoint and reject compatibility formats."""

    suffix = Path(path).suffix.lower()
    if suffix == ".pth":
        return "pytorch_state_dict", "canonical"
    raise ValueError("current checkpoints must end in .pth")


def resolve_checkpoint_path(directory: Path | str, basename: str = "net") -> Path:
    """Return the canonical checkpoint path without legacy fallback."""

    return Path(directory) / f"{basename}.pth"


def _load_tensor_state_dict(
    source: Path, execution_device: torch.device
) -> dict[str, torch.Tensor]:
    """Load and validate the shared tensor-only state-dict envelope."""

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
    return state_dict


def load_checkpoint_state(
    path: Path | str, *, device: str | torch.device | None = None
) -> tuple[dict[str, torch.Tensor], str, str]:
    """Load one canonical checkpoint through the tensor-only boundary."""

    source = Path(path)
    source_format, compatibility_mode = checkpoint_format(source)
    execution_device = torch.device(device or default_device())
    state_dict = _load_tensor_state_dict(source, execution_device)
    return state_dict, source_format, compatibility_mode


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


def validate_meta_checkpoint(checkpoint: Path, specification: dict, seed: int) -> dict:
    """Validate a registered meta-training checkpoint without loading weights."""

    metadata_path = checkpoint.parent / "config.json"
    with metadata_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    expected_training = dict(specification["training"])
    expected_training.pop("seeds")
    expected_training.pop("checkpoint_selection")
    expected_training["seed"] = seed
    if metadata["training"] != expected_training:
        raise RuntimeError(
            f"seed {seed} checkpoint training configuration is not registered"
        )
    if (
        specification["confirmation_id"] == FORMAL_CONFIRMATION_ID
        and metadata.get("execution") != COMPILED_TRAINING_EXECUTION
    ):
        raise RuntimeError(
            f"seed {seed} checkpoint was not trained with the registered compiler"
        )
    if metadata["completed_outer_steps"] != specification["training"]["outer_steps"]:
        raise RuntimeError(f"seed {seed} checkpoint is not the fixed final step")
    observed_signatures = {
        tuple(tuple(pair) for pair in signature)
        for signature in metadata["task_distribution"].get(
            "held_out_rank_graph_signatures", []
        )
    }
    if observed_signatures != set(registered_holdout_signatures()):
        raise RuntimeError(
            f"seed {seed} training did not exclude both Liu graph signatures"
        )
    observed_hash = file_sha256(checkpoint)
    if metadata["checkpoint"]["sha256"] != observed_hash:
        raise RuntimeError(f"seed {seed} checkpoint hash does not match metadata")
    return metadata


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
    return _restore_retro_checkpoint(
        path,
        state_dict,
        batch_size,
        execution_device=execution_device,
        source_format=source_format,
        compatibility_mode=compatibility_mode,
    )


def _restore_retro_checkpoint(
    path: Path,
    state_dict: dict[str, torch.Tensor],
    batch_size: int,
    *,
    execution_device: torch.device,
    source_format: str,
    compatibility_mode: str,
) -> tuple[RetroModulRNN, TrainConfig, CheckpointInfo]:
    """Restore a validated state dict for canonical or explicit legacy callers."""

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
