"""Named input-channel layout shared by ranking task adapters."""

from __future__ import annotations

from dataclasses import dataclass

ADDITIONAL_INPUTS = 4
ACTION_COUNT = 2
RESPONSE_STEP = 1
EVIDENCE_AUXILIARY_OFFSET = 3


@dataclass(frozen=True)
class RelationalInputLayout:
    cue_size: int
    action_count: int = ACTION_COUNT
    additional_inputs: int = ADDITIONAL_INPUTS

    def __post_init__(self) -> None:
        if self.cue_size < 1:
            raise ValueError("cue_size must be positive")
        if self.action_count < 1:
            raise ValueError("action_count must be positive")
        if self.additional_inputs != ADDITIONAL_INPUTS:
            raise ValueError("the v1 checkpoint ABI requires four auxiliary inputs")

    @property
    def pair_cue_width(self) -> int:
        return 2 * self.cue_size

    @property
    def stimulus_width(self) -> int:
        return self.pair_cue_width + 1

    @property
    def bias_index(self) -> int:
        return self.stimulus_width

    @property
    def time_index(self) -> int:
        return self.stimulus_width + 1

    @property
    def reward_index(self) -> int:
        return self.stimulus_width + 2

    @property
    def evidence_index(self) -> int:
        return self.stimulus_width + EVIDENCE_AUXILIARY_OFFSET

    @property
    def action_start(self) -> int:
        return self.stimulus_width + self.additional_inputs

    @property
    def input_size(self) -> int:
        return self.action_start + self.action_count

    def validate_width(self, width: int) -> None:
        if width != self.input_size:
            raise ValueError(
                f"input width {width} does not match the v1 ABI {self.input_size}"
            )
