"""Unchanged continuous recurrent computation with an actually optional local store."""

import torch

from fsrl.experiments.memory_structure.model import Rollout, objective, read_queries


def rollout(backbone, local, seqs, batch, states, rounding_seed):
    subjects = batch.support_inputs.shape[2]
    hidden = backbone.initial_hidden(subjects)
    eligibility = backbone.initial_eligibility(subjects)
    weights = backbone.initial_fast_weights(subjects)
    blank = batch.support_inputs.new_zeros(
        2, subjects, backbone.model_config.input_size
    )
    _, _, _, _, _, weights = seqs[1](blank, hidden, eligibility, weights, True)
    state = None if local is None else local.initial_state(subjects)
    rng = torch.Generator(device=weights.device).manual_seed(rounding_seed)
    writes = []
    first = weights
    for index, inputs in enumerate(batch.support_inputs.unbind(0)):
        shape = (inputs.shape[0], *weights.shape)
        if states is None:
            uniforms = weights.new_zeros(())
        else:
            uniforms = torch.rand(shape, generator=rng, device=weights.device)
        weights, amount = seqs[0](inputs, hidden, eligibility, weights, uniforms)
        writes.append(amount)
        if local is not None:
            state = local.write(
                state, inputs[0, :, : 2 * local.cue_size], batch.local_evidence[index]
            )
        if index == 0:
            first = weights
    logits, global_logits = read_queries(
        backbone, local, seqs[1], batch, weights, state
    )
    per_trial = torch.stack(writes)
    cost = per_trial.sum(0) / (
        batch.support_inputs.shape[0] * batch.support_inputs.shape[1]
    )
    return Rollout(logits, global_logits, weights, state, first), cost, per_trial


def update(backbone, local, seqs, batch, optimizer, spec, states, coefficient, draw):
    optimizer.zero_grad(set_to_none=True)
    result, cost, _ = rollout(backbone, local, seqs, batch, states, draw)
    base, ce = objective(result, batch, spec["optimization"]["fast_weight_penalty"])
    loss = base + coefficient * cost.mean()
    loss.backward()
    limit = spec["optimization"]["gradient_clip"]
    torch.nn.utils.clip_grad_norm_(
        backbone.parameters(), limit, error_if_nonfinite=True
    )
    if local is not None:
        torch.nn.utils.clip_grad_norm_([local.raw_gain], limit, error_if_nonfinite=True)
    optimizer.step()
    return {
        "loss": float(loss.detach()),
        "ce": float(ce.detach()),
        "cost": float(cost.detach().mean()),
    }
