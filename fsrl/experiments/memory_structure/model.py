"""Common plastic RNN with an optional, actually absent, local store."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from fsrl.core.local_trace import ConjunctiveLocalTrace
from fsrl.core.model_config import RetroModelConfig
from fsrl.core.plastic_rnn import RetroModulRNN


@dataclass
class Rollout:
    logits: torch.Tensor
    global_logits: torch.Tensor
    weights: torch.Tensor
    local_state: torch.Tensor | None
    first_write: torch.Tensor


def make_model(spec: dict, seed: int, condition: str, device: str):
    torch.manual_seed(seed)
    backbone = RetroModulRNN(
        RetroModelConfig(
            spec["architecture"]["input_size"],
            spec["architecture"]["hidden_size"],
            2,
            spec["optimization"]["batch_size"],
        ),
        device=device,
    )
    local = None
    if condition == "dual":
        local = ConjunctiveLocalTrace(
            spec["task"]["cue_size"],
            initial_gain=spec["optimization"]["initial_local_gain"],
            device=device,
        )
    elif condition != "single":
        raise ValueError("unknown memory condition")
    return backbone, local


def read_queries(backbone, local, sequence, batch, weights, state, route=None):
    subjects = weights.shape[0]
    count = batch.targets.numel() // subjects
    logits, _, _, _, _, _ = sequence(
        batch.query_inputs,
        backbone.initial_hidden(count * subjects),
        backbone.initial_eligibility(count * subjects),
        weights.repeat(count, 1, 1),
        False,
    )
    if local is None:
        return logits, logits
    cues = batch.query_inputs[0, :, : 2 * local.cue_size]
    if route is not None:
        cues = cues.reshape(count, subjects, -1).transpose(0, 1)
        subjects_index = torch.arange(subjects, device=cues.device)[:, None]
        cues = cues[subjects_index, route].transpose(0, 1).reshape(count * subjects, -1)
    combined, _, _, _ = local(logits, state.repeat(count, 1), cues)
    return combined, logits


def forward_batch(backbone, local, sequence, batch) -> Rollout:
    subjects = batch.support_inputs.shape[2]
    hidden = backbone.initial_hidden(subjects)
    eligibility = backbone.initial_eligibility(subjects)
    weights = backbone.initial_fast_weights(subjects)
    blank = batch.support_inputs.new_zeros(
        2, subjects, backbone.model_config.input_size
    )
    _, _, _, _, _, weights = sequence(blank, hidden, eligibility, weights, True)
    state = None if local is None else local.initial_state(subjects)
    first = weights
    for index, inputs in enumerate(batch.support_inputs.unbind()):
        _, _, _, _, _, weights = sequence(inputs, hidden, eligibility, weights, True)
        if local is not None:
            state = local.write(
                state, inputs[0, :, : 2 * local.cue_size], batch.local_evidence[index]
            )
        if index == 0:
            first = weights
    logits, global_logits = read_queries(
        backbone, local, sequence, batch, weights, state
    )
    return Rollout(logits, global_logits, weights, state, first)


def objective(result: Rollout, batch, penalty: float):
    ce = F.cross_entropy(result.logits, batch.targets)
    return ce + penalty * result.weights.square().mean(), ce


def optimizer_for(backbone, local, spec: dict):
    opt = spec["optimization"]
    groups = [
        {"params": list(backbone.parameters()), "lr": opt["backbone_learning_rate"]}
    ]
    if local is not None:
        groups.append({"params": [local.raw_gain], "lr": opt["local_learning_rate"]})
    return torch.optim.Adam(groups, betas=(0.9, 0.999), eps=1e-8)


def update(backbone, local, sequence, batch, optimizer, spec: dict):
    optimizer.zero_grad(set_to_none=True)
    result = forward_batch(backbone, local, sequence, batch)
    loss, ce = objective(result, batch, spec["optimization"]["fast_weight_penalty"])
    loss.backward()
    clip = spec["optimization"]["gradient_clip"]
    torch.nn.utils.clip_grad_norm_(backbone.parameters(), clip, error_if_nonfinite=True)
    if local is not None:
        torch.nn.utils.clip_grad_norm_([local.raw_gain], clip, error_if_nonfinite=True)
    optimizer.step()
    return float(loss.detach()), float(ce.detach())
