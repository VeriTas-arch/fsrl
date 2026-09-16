"""Stage-3 probes evaluated from frozen direct-authority prefix states."""

from __future__ import annotations

from itertools import combinations, product

import numpy as np
import torch

from fsrl.experiments.single_p_time_role.estimands import fixed_effect_slope, hodge

from .direct import apply_support_probe, read_ordered


def _probe_inputs(codes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    probes = tuple(
        product(
            combinations(range(8), 2),
            (-4.0 / 7.0, -1.0 / 7.0, 1.0 / 7.0, 4.0 / 7.0),
        )
    )
    inputs = np.zeros((4, len(probes), 32), dtype=np.float32)
    for index, ((left, right), value) in enumerate(probes):
        inputs[0, index, :15] = codes[left]
        inputs[0, index, 15:30] = codes[right]
        inputs[0, index, 31] = value
        inputs[1, index, 30] = 1.0
    times = np.full((4, len(probes), 1), 1.0 / 3.0, dtype=np.float32)
    return inputs, times


def _canonical_field(ordered: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    lookup = {
        tuple(map(int, pair)): float(value)
        for pair, value in zip(pairs, ordered, strict=True)
    }
    return np.asarray(
        [
            (lookup[(left, right)] - lookup[(right, left)]) / 2.0
            for left, right in combinations(range(8), 2)
        ],
        dtype=np.float64,
    )


def _correct_signs(cpu) -> np.ndarray:
    subjects = cpu.arrays["item_codes"].shape[0]
    targets = cpu.arrays["targets"].reshape(-1, subjects).T
    pairs = np.asarray(cpu.arrays["query_pairs"]).transpose(1, 0, 2)
    signs = np.empty((subjects, 28), dtype=np.float64)
    canonical = tuple(combinations(range(8), 2))
    for episode in range(subjects):
        lookup = {
            tuple(map(int, pair)): 2.0 * float(target) - 1.0
            for pair, target in zip(pairs[episode], targets[episode], strict=True)
        }
        for index, (left, right) in enumerate(canonical):
            signs[episode, index] = (
                lookup[(left, right)]
                if (left, right) in lookup
                else -lookup[(right, left)]
            )
    return signs


def _probe_susceptibility(model, weight, codes, baseline):
    inputs, times = _probe_inputs(codes)
    repeated = weight.expand(inputs.shape[1], -1, -1).clone()
    before = repeated.clone()
    changed = apply_support_probe(model, inputs, times, repeated)
    write = torch.linalg.vector_norm(
        model.alpha * (changed - before), dim=(-2, -1)
    ).mean()
    code_batch = np.broadcast_to(codes, (inputs.shape[1], *codes.shape)).copy()
    ordered, pairs = read_ordered(model, changed, code_batch)
    fields = np.stack(
        [
            _canonical_field(row, pair_row)
            for row, pair_row in zip(ordered, pairs, strict=True)
        ]
    )
    functional = np.sqrt(np.mean((fields - baseline[None, :]) ** 2, axis=1))
    return float(write.cpu()), float(np.mean(functional))


def trajectory_rows(model, cpu, states: np.ndarray, modulations: np.ndarray) -> dict:
    state_tensor = torch.from_numpy(states).to("cuda")
    correct_signs = _correct_signs(cpu)
    fields, potentials = [], []
    for state in state_tensor:
        ordered, pairs = read_ordered(model, state, cpu.arrays["item_codes"])
        field = np.stack(
            [
                _canonical_field(row, pair_row)
                for row, pair_row in zip(ordered, pairs, strict=True)
            ]
        )
        fields.append(field)
        potentials.append(hodge(field)["potentials"])
    final = potentials[-1]
    rows = {
        name: []
        for name in (
            "episode",
            "prefix",
            "maturity",
            "write_susceptibility",
            "functional_susceptibility",
            "potential_alignment_final",
            "potential_norm",
            "residual_fraction",
            "mean_absolute_margin",
            "mean_correct_signed_margin",
            "natural_modulation_magnitude",
            "natural_write_norm",
            "natural_effective_write",
        )
    }
    for prefix, (state, field, potential) in enumerate(
        zip(state_tensor, fields, potentials, strict=True)
    ):
        residual = hodge(field)["residual_fraction"]
        numerator = np.sum(potential * final, axis=1)
        denominator = np.linalg.norm(potential, axis=1) * np.linalg.norm(final, axis=1)
        for episode in range(len(field)):
            write_s, functional_s = _probe_susceptibility(
                model,
                state[episode : episode + 1],
                cpu.arrays["item_codes"][episode],
                field[episode],
            )
            terminal = prefix == len(state_tensor) - 1
            delta = (
                None if terminal else state_tensor[prefix + 1, episode] - state[episode]
            )
            rows["episode"].append(episode)
            rows["prefix"].append(prefix)
            rows["maturity"].append(float(np.linalg.norm(potential[episode])))
            rows["write_susceptibility"].append(write_s)
            rows["functional_susceptibility"].append(functional_s)
            rows["potential_alignment_final"].append(
                float(numerator[episode] / (denominator[episode] + 1e-12))
            )
            rows["potential_norm"].append(float(np.linalg.norm(potential[episode])))
            rows["residual_fraction"].append(float(residual[episode]))
            rows["mean_absolute_margin"].append(float(np.mean(np.abs(field[episode]))))
            rows["mean_correct_signed_margin"].append(
                float(np.mean(correct_signs[episode] * field[episode]))
            )
            rows["natural_modulation_magnitude"].append(
                np.nan if terminal else float(abs(modulations[prefix, episode]))
            )
            rows["natural_write_norm"].append(
                np.nan if terminal else float(torch.linalg.vector_norm(delta).cpu())
            )
            rows["natural_effective_write"].append(
                np.nan
                if terminal
                else float(torch.linalg.vector_norm(model.alpha * delta).cpu())
            )
    return {
        key: np.asarray(value, dtype=np.int64 if key == "episode" else np.float64)
        for key, value in rows.items()
    }


def regression_summary(
    panels: list[dict[str, np.ndarray]], *, seed: int, draws: int
) -> dict:
    point = np.mean(
        [
            fixed_effect_slope(
                row["functional_susceptibility"], row["maturity"], row["prefix"]
            )
            for row in panels
        ]
    )
    rng = np.random.default_rng(seed)
    distribution = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        slopes = []
        for row in panels:
            episodes = np.unique(row["episode"])
            selected = rng.choice(episodes, size=len(episodes), replace=True)
            indices = np.concatenate(
                [np.flatnonzero(row["episode"] == episode) for episode in selected]
            )
            slopes.append(
                fixed_effect_slope(
                    row["functional_susceptibility"][indices],
                    row["maturity"][indices],
                    row["prefix"][indices],
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


__all__ = ["regression_summary", "trajectory_rows"]
