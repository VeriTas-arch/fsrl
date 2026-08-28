"""Typed configuration boundary for the checkpoint-compatible plastic RNN."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RetroModelConfig:
    """Canonical model dimensions, independent of legacy training key names."""

    input_size: int
    hidden_size: int
    output_size: int
    batch_size: int

    def __post_init__(self) -> None:
        for name, value in (
            ("input_size", self.input_size),
            ("hidden_size", self.hidden_size),
            ("output_size", self.output_size),
            ("batch_size", self.batch_size),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")

    @classmethod
    def from_legacy_mapping(cls, config: Mapping[str, Any]) -> RetroModelConfig:
        """Adapt the historical ``inputsize/hs/outputsize/bs`` mapping once."""

        required = {
            "inputsize": "input_size",
            "hs": "hidden_size",
            "outputsize": "output_size",
            "bs": "batch_size",
        }
        missing = [key for key in required if key not in config]
        if missing:
            raise KeyError(f"legacy model config is missing keys: {missing}")
        return cls(
            **{target: int(config[source]) for source, target in required.items()}
        )

    def to_legacy_mapping(self) -> dict[str, int]:
        """Return the one-way adapter required by historical consumers."""

        return {
            "inputsize": self.input_size,
            "hs": self.hidden_size,
            "outputsize": self.output_size,
            "bs": self.batch_size,
        }

    @property
    def cue_size(self) -> int:
        """Derive the relational cue width from the frozen input ABI."""

        remainder = self.input_size - 7
        if remainder <= 0 or remainder % 2:
            raise ValueError(
                f"input_size {self.input_size} does not encode the relational ABI"
            )
        return remainder // 2
