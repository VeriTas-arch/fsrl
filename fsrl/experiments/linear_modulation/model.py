"""Single affine modulation; preserve all other equations and common initialization."""

import torch
from torch import nn

from fsrl.core.model_config import RetroModelConfig
from fsrl.core.plastic_rnn import RetroModulRNN
from fsrl.experiments.finite_state.model import sequences
from fsrl.experiments.training_strategy.locks import verify_reference
from fsrl.infra.provenance import load_json, tensor_hashes


class LinearModulationRNN(RetroModulRNN):
    def __init__(self, config, *, device=None):
        super().__init__(config, device=device)
        weight = self.DAmult.detach() * self.h2DA.weight.detach().diff(dim=0).neg()
        bias = self.DAmult.detach() * self.h2DA.bias.detach().diff().neg()
        # Meta construction consumes no RNG; common parameter initialization is exact.
        head = nn.Linear(self.model_config.hidden_size, 1, device="meta")
        head.weight = nn.Parameter(weight.clone())
        head.bias = nn.Parameter(bias.clone())
        self.h2DA = head
        del self.DAmult

    def forward(self, inputs, hidden, et, pw):
        batch_size = inputs.shape[0]
        hidden_size = self.model_config.hidden_size
        hactiv = self.activ(
            self.i2h(inputs).view(batch_size, hidden_size, 1)
            + torch.matmul(
                self.w + self.alpha * pw, hidden.view(batch_size, hidden_size, 1)
            )
        ).view(batch_size, hidden_size)
        activout, valueout = self.h2o(hactiv), self.h2v(hactiv)
        modulation = self.h2DA(hactiv)
        pw = pw + modulation.view(batch_size, 1, 1) * et
        torch.clip_(pw, min=-50.0, max=50.0)
        delta = torch.bmm(
            hactiv.view(batch_size, hidden_size, 1),
            hidden.view(batch_size, 1, hidden_size),
        )
        et = (1 - self.etaet) * et + self.etaet * torch.tanh(delta)
        return activout, valueout, modulation, hactiv, et, pw


def make_model(spec, seed, condition="single", device="cuda"):
    assert condition == "single"
    torch.manual_seed(seed)
    return LinearModulationRNN(
        RetroModelConfig(
            spec["architecture"]["input_size"],
            spec["architecture"]["hidden_size"],
            2,
            spec["optimization"]["batch_size"],
        ),
        device=device,
    ), None


def common_hashes(hashes):
    return {
        k: v
        for k, v in hashes.items()
        if k not in ("h2DA.weight", "h2DA.bias", "DAmult")
    }


def load_model(seed, model, spec):
    backbone, local = make_model(spec, seed)
    metadata = load_json(verify_reference(model["result.json"]))
    backbone.load_state_dict(
        torch.load(
            verify_reference(model["net.pth"]),
            weights_only=True,
            map_location="cuda",
        )
    )
    if tensor_hashes(backbone) != metadata["final_backbone"]:
        raise RuntimeError("loaded affine model differs from locked weights")
    backbone.requires_grad_(False).eval()
    return backbone, local, sequences(backbone, None, compiled=True)
