import numpy as np
import torch
from torch import nn

from ..runtime import default_device
from .state import PlasticRNNState


class RetroModulRNN(nn.Module):
    """RNN with neuromodulated recurrent plasticity.

    ``et`` is the Hebbian eligibility trace. ``pw`` is the within-episode
    plastic recurrent weight matrix. Only ``pw`` changes during an episode.
    """

    def __init__(self, config, *, device: str | torch.device | None = None):
        super().__init__()
        for paramname in ["outputsize", "inputsize", "hs", "bs"]:
            if paramname not in config:
                raise KeyError("Must provide missing key in config: " + paramname)

        nbda = 2
        self.GG = config
        self.execution_device = torch.device(device or default_device())
        self.activ = torch.tanh
        self.i2h = torch.nn.Linear(config["inputsize"], config["hs"]).to(
            self.execution_device
        )
        self.w = torch.nn.Parameter(
            (
                (1.0 / np.sqrt(config["hs"]))
                * (2.0 * torch.rand(config["hs"], config["hs"]) - 1.0)
            ).to(self.execution_device),
            requires_grad=True,
        )
        self.alpha = torch.nn.Parameter(
            (0.01 * (2.0 * torch.rand(config["hs"], config["hs"]) - 1.0)).to(
                self.execution_device
            ),
            requires_grad=True,
        )
        self.etaet = torch.nn.Parameter(
            (0.7 * torch.ones(1)).to(self.execution_device), requires_grad=True
        )
        self.DAmult = torch.nn.Parameter(
            (1.0 * torch.ones(1)).to(self.execution_device), requires_grad=True
        )
        self.h2DA = torch.nn.Linear(config["hs"], nbda).to(self.execution_device)
        self.h2o = torch.nn.Linear(config["hs"], config["outputsize"]).to(
            self.execution_device
        )
        self.h2v = torch.nn.Linear(config["hs"], 1).to(self.execution_device)

    def forward(self, inputs, hidden, et, pw):
        batch_size = inputs.shape[0]
        hidden_size = self.GG["hs"]
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

    def initialZeroET(self, batch_size):
        return self.w.new_zeros(batch_size, self.GG["hs"], self.GG["hs"])

    def initialZeroPlasticWeights(self, batch_size):
        return self.w.new_zeros(batch_size, self.GG["hs"], self.GG["hs"])

    def initialZeroState(self, batch_size):
        return self.w.new_zeros(batch_size, self.GG["hs"])

    def initial_state(self, batch_size: int) -> PlasticRNNState:
        """Return the typed equivalent of the three historical zero-state calls."""

        return PlasticRNNState(
            hidden=self.initialZeroState(batch_size),
            eligibility=self.initialZeroET(batch_size),
            fast_weights=self.initialZeroPlasticWeights(batch_size),
        )
