"""Native single-P learner and a nonplastic GRU with persistent hidden state."""

import numpy as np
import torch
from torch import nn

from fsrl.core.sequence import RecurrentSequence
from fsrl.experiments.relational_revision.model import network, probe


class PersistentGRU(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.cell = nn.GRUCell(38, 200, device=device)
        self.readout = nn.Linear(200, 2, device=device)

    def advance(self, inputs, state):
        for x in inputs.unbind():
            state = self.cell(x, state)
        return state


def make_model(kind, seed, device="cuda"):
    torch.manual_seed(seed)
    if kind == "plastic":
        return network(device=device, seed=seed)
    if kind == "gru":
        return PersistentGRU(device)
    raise ValueError(kind)


def initial(net, count, like):
    blank = like.new_zeros(2, count, 38)
    if isinstance(net, PersistentGRU):
        return net.advance(blank, like.new_zeros(count, 200))
    return RecurrentSequence(net)(
        blank,
        net.initial_hidden(count),
        net.initial_eligibility(count),
        net.initial_fast_weights(count),
        True,
    )[-1]


def advance(net, inputs, state):
    if isinstance(net, PersistentGRU):
        return net.advance(inputs, state)
    count = state.shape[0]
    return RecurrentSequence(net)(
        inputs,
        net.initial_hidden(count),
        net.initial_eligibility(count),
        state,
        True,
    )[-1]


def read(net, state, query):
    if not isinstance(net, PersistentGRU):
        return probe(net, state, query)
    count = state.shape[0]
    result = []
    for chunk in query.split(8):
        copies = len(chunk)
        inputs = chunk.transpose(0, 1).reshape(2, copies * count, 38)
        h = net.advance(inputs, state.repeat(copies, 1))
        logits = net.readout(h)
        result.append((logits[:, 1] - logits[:, 0]).reshape(copies, count).T)
    return torch.cat(result, 1)


def forward(net, support, query, prefixes):
    state = initial(net, support.shape[2], support)
    outputs = []
    for t, inputs in enumerate(support.unbind(), 1):
        state = advance(net, inputs, state)
        if t in prefixes:
            outputs.append(read(net, state, query))
    return torch.stack(outputs, 1), state


def trajectory(net, panel, batch_size=16):
    device = next(net.parameters()).device
    outputs = []
    for start in range(0, panel["support"].shape[2], batch_size):
        support = torch.as_tensor(
            panel["support"][:, :, start : start + batch_size], device=device
        )
        query = torch.as_tensor(
            panel["query"][:, :, start : start + batch_size], device=device
        )
        margins, _ = forward(net, support, query, range(24, 33))
        outputs.append(margins.detach().cpu().numpy())
    return {"margins": np.concatenate(outputs)}
