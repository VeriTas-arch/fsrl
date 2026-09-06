"""Fixed and experience-dependent normalized metric-error learners."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from fsrl.core.local_trace import inverse_softplus

from .protocol import CONDITIONS


class AdaptiveScoreLearner(nn.Module):
    def __init__(
        self,
        cue_size: int,
        *,
        condition: str,
        scheduler: str = "relation",
        initial_eta: float,
        initial_global_gain: float,
        epsilon: float,
        max_relations: int,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        if condition not in CONDITIONS:
            raise ValueError("unregistered adaptive-plasticity condition")
        if scheduler not in {"relation", "global"}:
            raise ValueError("scheduler must be relation or global")
        if not 0 < initial_eta < 1 or epsilon <= 0 or max_relations < 1:
            raise ValueError("invalid score-learner initialization")
        self.cue_size = cue_size
        self.condition = condition
        self.scheduler = scheduler
        self.epsilon = epsilon
        self.max_relations = max_relations
        self.raw_eta = nn.Parameter(
            torch.tensor([math.log(initial_eta / (1 - initial_eta))], device=device)
        )
        self.raw_global_gain = nn.Parameter(
            torch.tensor([inverse_softplus(initial_global_gain)], device=device)
        )

    @property
    def eta(self) -> torch.Tensor:
        return torch.sigmoid(self.raw_eta)

    @property
    def global_gain(self) -> torch.Tensor:
        return F.softplus(self.raw_global_gain)

    @property
    def adaptive(self) -> bool:
        return self.condition == "adaptive_eta_resampled"

    def forward(
        self,
        support_cues: torch.Tensor,
        signed: torch.Tensor,
        retention: torch.Tensor,
        relation_slots: torch.Tensor,
        query_cues: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        subjects = support_cues.shape[1]
        w = support_cues.new_zeros(subjects, self.cue_size)
        efficacy = self.eta.expand(subjects, self.max_relations).clone()
        relation_count = relation_slots.max(dim=0).values + 1
        for t, cues in enumerate(support_cues.unbind()):
            x = cues[:, : self.cue_size] - cues[:, self.cue_size :]
            if not self.adaptive:
                step_eta = self.eta
            elif self.scheduler == "relation":
                slots = relation_slots[t, :, None]
                step_eta = efficacy.gather(1, slots).squeeze(1)
            else:
                completed_rounds = torch.div(t, relation_count, rounding_mode="floor")
                step_eta = self.eta / (1 + completed_rounds * self.eta)
            error = signed[t] - (w * x).sum(dim=-1)
            update = (
                step_eta
                * retention[t]
                * error
                / (self.epsilon + x.square().sum(dim=-1))
            )
            w = w + update[:, None] * x
            if self.adaptive and self.scheduler == "relation":
                next_eta = step_eta / (1 + step_eta)
                change = retention[t] * (next_eta - step_eta)
                efficacy = efficacy.scatter_add(1, slots, change[:, None])
        q = query_cues[..., : self.cue_size] - query_cues[..., self.cue_size :]
        margins = self.global_gain * (w[:, None] * q).sum(dim=-1)
        visible_efficacy = (
            efficacy
            if self.adaptive and self.scheduler == "relation"
            else support_cues.new_empty(subjects, 0)
        )
        return margins, w, visible_efficacy


def make_model(
    condition: str, spec: dict, device: str = "cpu", *, scheduler: str = "relation"
) -> AdaptiveScoreLearner:
    settings = spec["optimization"]
    return AdaptiveScoreLearner(
        spec["task"]["cue_size"],
        condition=condition,
        scheduler=scheduler,
        initial_eta=settings["initial_eta"],
        initial_global_gain=settings["initial_global_gain"],
        epsilon=spec["model"]["epsilon"],
        max_relations=spec["task"]["max_edges"],
        device=device,
    )


def load_model(
    config: dict,
    spec: dict,
    device: str = "cuda",
    *,
    scheduler: str = "relation",
) -> AdaptiveScoreLearner:
    from fsrl.infra.provenance import tensor_hashes

    model = make_model(config["condition"], spec, device, scheduler=scheduler)
    model.load_state_dict(
        {
            key: torch.tensor(value, dtype=torch.float32, device=device)
            for key, value in config["raw_parameters"].items()
        }
    )
    if tensor_hashes(model) != config["final_parameters"]:
        raise RuntimeError("loaded adaptive-plasticity parameters differ")
    return model.requires_grad_(False).eval()
