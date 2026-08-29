"""First-class implementation of the confirmed global/local state organization."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch
from torch import nn

from .inputs import RESPONSE_STEP
from .local_trace import ConjunctiveLocalTrace
from .plastic_rnn import RetroModulRNN
from .state import RelationalEpisodeState


class RelationalIntervention(str, Enum):
    INTACT = "intact"
    GLOBAL_OFF = "global_off"
    LOCAL_OFF = "local_off"


@dataclass(frozen=True)
class RelationalQueryReadout:
    logits: torch.Tensor
    global_logits: torch.Tensor
    raw_local_margin: torch.Tensor
    local_gain: torch.Tensor
    local_correction: torch.Tensor
    policy_residual: torch.Tensor


class GlobalLocalRelationalSystem(nn.Module):
    """Compose the frozen global fast-weight state with the direct local trace.

    This class changes no registered equation.  It gives the already confirmed
    ``P_T`` and ``L_T`` organization one maintained state and rollout API.
    """

    def __init__(
        self,
        backbone: RetroModulRNN,
        local_trace: ConjunctiveLocalTrace,
        *,
        response_step: int = RESPONSE_STEP,
    ) -> None:
        super().__init__()
        if backbone.model_config.cue_size != local_trace.cue_size:
            raise ValueError("backbone and local trace use different cue sizes")
        if response_step < 0:
            raise ValueError("response_step must be non-negative")
        self.backbone = backbone
        self.local = local_trace
        self.response_step = int(response_step)

    @property
    def cue_size(self) -> int:
        return self.backbone.model_config.cue_size

    def initialize_episode(
        self, batch_size: int, *, blank_steps: int = 2
    ) -> RelationalEpisodeState:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if blank_steps < 0:
            raise ValueError("blank_steps must be non-negative")
        recurrent = self.backbone.initial_state(batch_size)
        blank = recurrent.hidden.new_zeros(
            batch_size, self.backbone.model_config.input_size
        )
        hidden = recurrent.hidden
        eligibility = recurrent.eligibility
        fast_weights = recurrent.fast_weights
        for _ in range(blank_steps):
            _, _, _, hidden, eligibility, fast_weights = self.backbone(
                blank, hidden, eligibility, fast_weights
            )
        return RelationalEpisodeState(
            global_fast_weights=fast_weights,
            local_trace=self.local.initial_state(batch_size),
        )

    def support_trial(
        self,
        state: RelationalEpisodeState,
        input_sequence: torch.Tensor,
        *,
        pair_cues: torch.Tensor,
        local_signed_value: torch.Tensor,
        global_write: bool = True,
        local_write: bool = True,
    ) -> RelationalEpisodeState:
        self._validate_sequence(state, input_sequence, pair_cues)
        hidden = self.backbone.initial_hidden(input_sequence.shape[1])
        eligibility = self.backbone.initial_eligibility(input_sequence.shape[1])
        fast_weights = state.global_fast_weights
        local_state = state.local_trace
        if local_write:
            local_state = self.local.write(local_state, pair_cues, local_signed_value)
        for inputs in input_sequence.unbind():
            _, _, _, hidden, eligibility, proposed = self.backbone(
                inputs, hidden, eligibility, fast_weights
            )
            if global_write:
                fast_weights = proposed
        return RelationalEpisodeState(fast_weights, local_state)

    def query(
        self,
        state: RelationalEpisodeState,
        input_sequence: torch.Tensor,
        *,
        pair_cues: torch.Tensor,
        intervention: RelationalIntervention = RelationalIntervention.INTACT,
    ) -> RelationalQueryReadout:
        self._validate_sequence(state, input_sequence, pair_cues)
        intervention = RelationalIntervention(intervention)
        if input_sequence.shape[0] <= self.response_step:
            raise ValueError("query sequence does not contain the response step")
        batch_size = input_sequence.shape[1]
        hidden = self.backbone.initial_hidden(batch_size)
        eligibility = self.backbone.initial_eligibility(batch_size)
        fast_weights = state.global_fast_weights
        if intervention == RelationalIntervention.GLOBAL_OFF:
            fast_weights = torch.zeros_like(fast_weights)
        response_logits = None
        response_inputs = None
        pre_response_hidden = None
        for step, inputs in enumerate(input_sequence.unbind()):
            if step == self.response_step:
                response_inputs = inputs
                pre_response_hidden = hidden
            logits, _, _, hidden, eligibility, _ = self.backbone(
                inputs, hidden, eligibility, fast_weights
            )
            if step == self.response_step:
                response_logits = logits
        if response_logits is None:
            raise RuntimeError("response logits were not produced")
        if response_inputs is None or pre_response_hidden is None:
            raise RuntimeError("response state was not captured")
        gain_override = None
        if intervention == RelationalIntervention.LOCAL_OFF:
            gain_override = response_logits.new_zeros(batch_size, 1)
        corrected, raw, gain, correction = self.local(
            response_logits,
            state.local_trace,
            pair_cues,
            gain_override=gain_override,
        )
        return RelationalQueryReadout(
            logits=corrected,
            global_logits=response_logits,
            raw_local_margin=raw,
            local_gain=gain,
            local_correction=correction,
            policy_residual=self._policy_residual(
                response_inputs, pre_response_hidden, fast_weights
            ),
        )

    def _policy_residual(
        self,
        inputs: torch.Tensor,
        hidden: torch.Tensor,
        fast_weights: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = inputs.shape[0]
        hidden_size = self.backbone.model_config.hidden_size
        hidden_column = hidden.view(batch_size, hidden_size, 1)
        baseline = (
            self.backbone.i2h(inputs).view(batch_size, hidden_size, 1)
            + torch.matmul(self.backbone.w, hidden_column)
        ).view(batch_size, hidden_size)
        drive = torch.matmul(self.backbone.alpha * fast_weights, hidden_column).view(
            batch_size, hidden_size
        )
        baseline_hidden = torch.tanh(baseline)
        exact_increment = torch.tanh(baseline + drive) - baseline_hidden
        linear_increment = (1.0 - baseline_hidden.square()) * drive
        margin = (self.backbone.h2o.weight[1] - self.backbone.h2o.weight[0]).view(1, -1)
        exact_policy = torch.sum(margin * exact_increment, dim=1, keepdim=True)
        linear_policy = torch.sum(margin * linear_increment, dim=1, keepdim=True)
        return linear_policy - exact_policy

    def _validate_sequence(
        self,
        state: RelationalEpisodeState,
        input_sequence: torch.Tensor,
        pair_cues: torch.Tensor,
    ) -> None:
        if input_sequence.ndim != 3:
            raise ValueError("input_sequence must be [steps, batch, input]")
        batch_size = input_sequence.shape[1]
        if input_sequence.shape[2] != self.backbone.model_config.input_size:
            raise ValueError("input sequence does not match backbone input width")
        if state.global_fast_weights.shape[0] != batch_size:
            raise ValueError("global state batch does not match input sequence")
        if state.local_trace.shape[0] != batch_size:
            raise ValueError("local state batch does not match input sequence")
        if pair_cues.shape != (batch_size, 2 * self.cue_size):
            raise ValueError("pair_cues has the wrong shape")
