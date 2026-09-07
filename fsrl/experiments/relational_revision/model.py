"""Native affine cell, trial resets and write-disabled prefix probes."""

import numpy as np
import torch

from fsrl.core.model_config import RetroModelConfig
from fsrl.core.sequence import RecurrentSequence
from fsrl.experiments.linear_modulation.model import LinearModulationRNN


def network(checkpoint=None, device="cuda", seed=970001):
    torch.manual_seed(seed)
    net = LinearModulationRNN(RetroModelConfig(38, 200, 2, 32), device=device)
    if checkpoint is not None:
        net.load_state_dict(
            torch.load(checkpoint, map_location=device, weights_only=True)
        )
    return net


def probe(net, weights, inputs):
    """Native queries in query-major order; never change the support state."""
    count = weights.shape[0]
    seq = RecurrentSequence(net)
    result = []
    for chunk in inputs.split(8):
        copies = len(chunk)
        query = chunk.transpose(0, 1).reshape(2, copies * count, 38)
        logits, *_ = seq(
            query,
            net.initial_hidden(copies * count),
            net.initial_eligibility(copies * count),
            weights.repeat(copies, 1, 1),
            False,
        )
        result.append((logits[:, 1] - logits[:, 0]).reshape(copies, count).T)
    return torch.cat(result, dim=1)


def trajectory(net, panel, batch_size=32):
    device = net.w.device
    collected, budgets = [], []
    sequence = RecurrentSequence(net)
    for start in range(0, panel["support"].shape[2], batch_size):
        support = torch.as_tensor(
            panel["support"][:, :, start : start + batch_size], device=device
        )
        query = torch.as_tensor(
            panel["query"][:, :, start : start + batch_size], device=device
        )
        count = support.shape[2]
        hidden = net.initial_hidden(count)
        eligibility = net.initial_eligibility(count)
        weights = net.initial_fast_weights(count)
        _, _, _, _, _, weights = sequence(
            support.new_zeros(2, count, 38), hidden, eligibility, weights, True
        )
        margins, amounts = [], []
        for t, inputs in enumerate(support.unbind()):
            previous = weights
            _, _, _, _, _, weights = sequence(
                inputs, hidden, eligibility, weights, True
            )
            amounts.append(
                ((weights - previous) * net.alpha).square().sum((1, 2)).sqrt()
            )
            if t >= 23:
                margins.append(probe(net, weights, query))
        collected.append(torch.stack(margins, 1).detach().cpu().numpy())
        budgets.append(torch.stack(amounts, 1).detach().cpu().numpy())
    return {"margins": np.concatenate(collected), "write_norm": np.concatenate(budgets)}
