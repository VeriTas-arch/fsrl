"""Prefix and exhaustive-probe trajectory diagnostics."""

from __future__ import annotations

from itertools import combinations, product

import numpy as np
import torch

from fsrl.experiments.clean_single_p.model import AffineSingleP, AffineSinglePSequence
from fsrl.experiments.training_strategy.batches import EpisodeBatch

from .estimands import canonical_field, fixed_effect_slope, hodge
from .rollouts import read_ordered, support_trajectory


def subset_batch(cpu: EpisodeBatch, selected: np.ndarray) -> EpisodeBatch:
    arrays = {}
    subjects = cpu.arrays["item_codes"].shape[0]
    for name, value in cpu.arrays.items():
        if name in {"support_inputs"}:
            arrays[name] = value[:, :, selected].copy()
        elif name in {
            "local_evidence",
            "signed_magnitudes",
            "retention",
            "probabilities",
            "support_pairs",
        }:
            arrays[name] = value[:, selected].copy()
        elif name == "query_inputs":
            queries = value.shape[1] // subjects
            arrays[name] = (
                value.reshape(value.shape[0], queries, subjects, value.shape[-1])[
                    :, :, selected
                ]
                .copy()
                .reshape(value.shape[0], queries * len(selected), value.shape[-1])
            )
        elif name == "targets":
            arrays[name] = value.reshape(-1, subjects)[:, selected].copy().reshape(-1)
        elif value.ndim and value.shape[0] == subjects:
            arrays[name] = value[selected].copy()
        elif value.ndim >= 2 and value.shape[1] == subjects:
            arrays[name] = value[:, selected].copy()
        else:
            arrays[name] = value.copy()
    return EpisodeBatch(arrays)


def selected_generic(cpu: EpisodeBatch) -> EpisodeBatch:
    indices = np.argsort(cpu.arrays["episode_indices"])[:2]
    return subset_batch(cpu, indices)


def _probe_inputs(codes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pairs = tuple(combinations(range(8), 2))
    evidence = (-4.0 / 7.0, -1.0 / 7.0, 1.0 / 7.0, 4.0 / 7.0)
    probes = tuple(product(pairs, evidence))
    inputs = np.zeros((4, len(probes), 32), dtype=np.float32)
    for index, ((left, right), value) in enumerate(probes):
        inputs[0, index, :15] = codes[left]
        inputs[0, index, 15:30] = codes[right]
        inputs[0, index, 31] = value
        inputs[1, index, 30] = 1.0
    times = np.full((4, len(probes), 1), 1.0 / 3.0, dtype=np.float32)
    return inputs, times


def _probe_susceptibility(
    model: AffineSingleP,
    weights: torch.Tensor,
    codes: np.ndarray,
    baseline_field: np.ndarray,
) -> tuple[float, float]:
    inputs, times = _probe_inputs(codes)
    count = inputs.shape[1]
    repeated = weights.expand(count, -1, -1).clone()
    sequence = AffineSinglePSequence(model)
    kwargs = (
        {"time_values": torch.from_numpy(times).to("cuda")}
        if model.retains_time
        else {}
    )
    _, _, _, _, changed = sequence(
        torch.from_numpy(inputs).to("cuda"),
        model.initial_hidden(count),
        model.initial_eligibility(count),
        repeated,
        True,
        **kwargs,
    )
    alpha = model.alpha
    write = torch.linalg.vector_norm(alpha * (changed - repeated), dim=(-2, -1)).mean()
    code_batch = np.broadcast_to(codes, (count, *codes.shape)).copy()
    ordered, pairs = read_ordered(model, changed, code_batch)
    fields = canonical_field(ordered, pairs)
    functional = np.sqrt(np.mean((fields - baseline_field[None, :]) ** 2, axis=1))
    return float(write.cpu()), float(np.mean(functional))


def _canonical_correct_signs(cpu: EpisodeBatch) -> np.ndarray:
    subjects = cpu.arrays["item_codes"].shape[0]
    targets = cpu.arrays["targets"].reshape(-1, subjects).T
    pairs = np.asarray(cpu.arrays["query_pairs"]).transpose(1, 0, 2)
    signs = np.empty((subjects, 28), dtype=np.float64)
    canonical_pairs = tuple(combinations(range(8), 2))
    for episode in range(subjects):
        lookup = {
            tuple(map(int, pair)): 2.0 * float(target) - 1.0
            for pair, target in zip(pairs[episode], targets[episode], strict=True)
        }
        for index, (left, right) in enumerate(canonical_pairs):
            if (left, right) in lookup:
                signs[episode, index] = lookup[(left, right)]
            elif (right, left) in lookup:
                signs[episode, index] = -lookup[(right, left)]
            else:
                raise ValueError("generic query set omits a canonical item pair")
    return signs


def trajectory_rows(model: AffineSingleP, cpu: EpisodeBatch) -> dict[str, np.ndarray]:
    states, writes, modulations = support_trajectory(model, cpu)
    correct_signs = _canonical_correct_signs(cpu)
    rows = {
        "episode": [],
        "prefix": [],
        "maturity": [],
        "write_susceptibility": [],
        "functional_susceptibility": [],
        "potential_alignment_final": [],
        "potential_norm": [],
        "residual_fraction": [],
        "mean_absolute_margin": [],
        "mean_correct_signed_margin": [],
        "natural_modulation_magnitude": [],
        "natural_write_norm": [],
        "natural_effective_write": [],
    }
    fields, potentials = [], []
    for state in states:
        ordered, pairs = read_ordered(model, state, cpu.arrays["item_codes"])
        field = canonical_field(ordered, pairs)
        fields.append(field)
        potentials.append(hodge(field)["potentials"])
    final = potentials[-1]
    for prefix, (state, field, potential) in enumerate(
        zip(states, fields, potentials, strict=True)
    ):
        residual = hodge(field)["residual_fraction"]
        numerator = np.sum(potential * final, axis=1)
        denominator = np.linalg.norm(potential, axis=1) * np.linalg.norm(final, axis=1)
        alignment = numerator / (denominator + 1e-12)
        for episode in range(len(field)):
            write_s, functional_s = _probe_susceptibility(
                model,
                state[episode : episode + 1],
                cpu.arrays["item_codes"][episode],
                field[episode],
            )
            rows["episode"].append(episode)
            rows["prefix"].append(prefix)
            rows["maturity"].append(float(np.linalg.norm(potential[episode])))
            rows["write_susceptibility"].append(write_s)
            rows["functional_susceptibility"].append(functional_s)
            rows["potential_alignment_final"].append(float(alignment[episode]))
            rows["potential_norm"].append(float(np.linalg.norm(potential[episode])))
            rows["residual_fraction"].append(float(residual[episode]))
            rows["mean_absolute_margin"].append(float(np.mean(np.abs(field[episode]))))
            rows["mean_correct_signed_margin"].append(
                float(np.mean(correct_signs[episode] * field[episode]))
            )
            rows["natural_modulation_magnitude"].append(
                np.nan
                if prefix == len(states) - 1
                else float(torch.abs(modulations[prefix][episode]).cpu())
            )
            rows["natural_write_norm"].append(
                np.nan
                if prefix == len(states) - 1
                else float(torch.linalg.vector_norm(writes[prefix][episode]).cpu())
            )
            rows["natural_effective_write"].append(
                np.nan
                if prefix == len(states) - 1
                else float(
                    torch.linalg.vector_norm(
                        model.alpha * writes[prefix][episode]
                    ).cpu()
                )
            )
    return {
        key: np.asarray(value, dtype=np.float64 if key != "episode" else np.int64)
        for key, value in rows.items()
    }


def regression_summary(
    panels: list[dict[str, np.ndarray]], *, seed: int, draws: int
) -> dict:
    point = np.mean(
        [
            fixed_effect_slope(
                panel["functional_susceptibility"], panel["maturity"], panel["prefix"]
            )
            for panel in panels
        ]
    )
    rng = np.random.default_rng(seed)
    distribution = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        slopes = []
        for panel in panels:
            episodes = np.unique(panel["episode"])
            selected = rng.choice(episodes, size=len(episodes), replace=True)
            indices = np.concatenate(
                [np.flatnonzero(panel["episode"] == episode) for episode in selected]
            )
            slopes.append(
                fixed_effect_slope(
                    panel["functional_susceptibility"][indices],
                    panel["maturity"][indices],
                    panel["prefix"][indices],
                )
            )
        distribution[draw] = np.mean(slopes)
    return {
        "point": float(point),
        "interval": {
            "lower": float(np.quantile(distribution, 0.025)),
            "upper": float(np.quantile(distribution, 0.975)),
        },
    }


__all__ = ["regression_summary", "selected_generic", "subset_batch", "trajectory_rows"]
