from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from torch import nn

from fsrl.infra.runtime import default_device

from .model_config import RetroModelConfig
from .state import PlasticRNNState


class RetroModulRNN(nn.Module):
    """RNN with neuromodulated recurrent plasticity.

    ``et`` is the Hebbian eligibility trace. ``pw`` is the within-episode
    plastic recurrent weight matrix. Only ``pw`` changes during an episode.
    """

    def __init__(
        self,
        config: RetroModelConfig | Mapping[str, Any],
        *,
        device: str | torch.device | None = None,
    ):
        super().__init__()
        if isinstance(config, RetroModelConfig):
            model_config = config
        else:
            model_config = RetroModelConfig.from_legacy_mapping(config)

        nbda = 2
        self.model_config = model_config
        self.execution_device = torch.device(device or default_device())
        self.activ = torch.tanh
        self.i2h = torch.nn.Linear(
            model_config.input_size, model_config.hidden_size
        ).to(self.execution_device)
        self.w = torch.nn.Parameter(
            (
                (1.0 / np.sqrt(model_config.hidden_size))
                * (
                    2.0 * torch.rand(model_config.hidden_size, model_config.hidden_size)
                    - 1.0
                )
            ).to(self.execution_device),
            requires_grad=True,
        )
        self.alpha = torch.nn.Parameter(
            (
                0.01
                * (
                    2.0 * torch.rand(model_config.hidden_size, model_config.hidden_size)
                    - 1.0
                )
            ).to(self.execution_device),
            requires_grad=True,
        )
        self.etaet = torch.nn.Parameter(
            (0.7 * torch.ones(1)).to(self.execution_device), requires_grad=True
        )
        self.DAmult = torch.nn.Parameter(
            (1.0 * torch.ones(1)).to(self.execution_device), requires_grad=True
        )
        self.h2DA = torch.nn.Linear(model_config.hidden_size, nbda).to(
            self.execution_device
        )
        self.h2o = torch.nn.Linear(
            model_config.hidden_size, model_config.output_size
        ).to(self.execution_device)
        self.h2v = torch.nn.Linear(model_config.hidden_size, 1).to(
            self.execution_device
        )

    def forward(self, inputs, hidden, et, pw):
        batch_size = inputs.shape[0]
        hidden_size = self.model_config.hidden_size
        assert pw.shape[0] == hidden.shape[0] == et.shape[0] == batch_size

        hactiv = self.activ(
            self.i2h(inputs).view(batch_size, hidden_size, 1)
            + torch.matmul(
                (self.w + torch.mul(self.alpha, pw)),
                hidden.view(batch_size, hidden_size, 1),
            )
        ).view(batch_size, hidden_size)

        activout = self.h2o(hactiv)
        valueout = self.h2v(hactiv)

        daout2 = torch.tanh(self.h2DA(hactiv))
        daout = self.DAmult * (daout2[:, 0] - daout2[:, 1])[:, None]

        pw = pw + daout.view(batch_size, 1, 1) * et
        torch.clip_(pw, min=-50.0, max=50.0)

        deltaet = torch.bmm(
            hactiv.view(batch_size, hidden_size, 1),
            hidden.view(batch_size, 1, hidden_size),
        )
        deltaet = torch.tanh(deltaet)
        et = (1 - self.etaet) * et + self.etaet * deltaet

        return activout, valueout, daout, hactiv, et, pw

    def initial_hidden(self, batch_size: int) -> torch.Tensor:
        return self.w.new_zeros(batch_size, self.model_config.hidden_size)

    def initial_eligibility(self, batch_size: int) -> torch.Tensor:
        return self.w.new_zeros(
            batch_size, self.model_config.hidden_size, self.model_config.hidden_size
        )

    def initial_fast_weights(self, batch_size: int) -> torch.Tensor:
        return self.w.new_zeros(
            batch_size, self.model_config.hidden_size, self.model_config.hidden_size
        )

    def initial_state(self, batch_size: int) -> PlasticRNNState:
        """Return the canonical typed recurrent state."""

        return PlasticRNNState(
            hidden=self.initial_hidden(batch_size),
            eligibility=self.initial_eligibility(batch_size),
            fast_weights=self.initial_fast_weights(batch_size),
        )
