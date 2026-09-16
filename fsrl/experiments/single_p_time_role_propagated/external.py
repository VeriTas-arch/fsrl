"""Direct external-output reconstruction without an adapter arithmetic path."""

from __future__ import annotations

import gc

import numpy as np
import torch

from fsrl.experiments.clean_single_p.protocol import inherited_recipe
from fsrl.experiments.finite_state.liu import primary_analysis
from fsrl.experiments.memory_structure.inputs import size_protocol
from fsrl.experiments.memory_structure.interventions import (
    remove_relation,
    shuffle_evidence,
)
from fsrl.experiments.single_p_time_role.estimands import canonical_field
from fsrl.experiments.training_strategy.evaluation import flatten_arrays
from fsrl.experiments.write_cost.evaluation import summarize_generic

from .direct import load_cpu, read_ordered, read_original, support_trajectory

CELLS = {
    "A0": {"training": "clean", "observation": "clean"},
    "Ae": {"training": "clean", "observation": "noisy"},
    "C0": {"training": "noisy", "observation": "clean"},
    "Ce": {"training": "noisy", "observation": "noisy"},
}


def _storage(weights: torch.Tensor, alpha: torch.Tensor) -> dict[str, np.ndarray]:
    values = weights.detach().cpu().numpy()
    effective = (weights * alpha).detach().cpu().numpy()
    return {
        "P_abs_mean": np.abs(values).mean((1, 2)),
        "P_abs_max": np.abs(values).max((1, 2)),
        "A_abs_mean": np.abs(effective).mean((1, 2)),
        "A_abs_max": np.abs(effective).max((1, 2)),
        "zero_fraction": (values == 0).mean((1, 2)),
        "boundary_fraction": (np.abs(values) == 50).mean((1, 2)),
    }


def _rollout(model, cpu) -> tuple[torch.Tensor, np.ndarray, np.ndarray, np.ndarray]:
    states, _, _, total_write = support_trajectory(model, cpu)
    terminal = states[-1]
    margins = read_original(model, terminal, cpu).astype(np.float64)
    cost = (
        (
            total_write
            / (
                cpu.arrays["support_inputs"].shape[0]
                * cpu.arrays["support_inputs"].shape[1]
            )
        )
        .cpu()
        .numpy()
    )
    return terminal, margins, cost, total_write.cpu().numpy()


def _field(model, weights: torch.Tensor, cpu) -> np.ndarray:
    ordered, pairs = read_ordered(model, weights, cpu.arrays["item_codes"])
    return canonical_field(ordered, pairs)


def _generic(
    model,
    inputs: dict[str, dict],
    observation: str,
    recipe: dict,
    keep_p: bool,
    sample_seed: int,
):
    collected: dict[str, list[np.ndarray]] = {}
    fields = []
    internal = {}
    for name, record in sorted(inputs.items()):
        if not name.startswith("test-"):
            continue
        cpu = load_cpu(record, observation, recipe)
        terminal, margins, cost, total_write = _rollout(model, cpu)
        signs = (2 * cpu.arrays["targets"] - 1).reshape(-1, len(cost)).T
        row = {
            "margins": margins,
            "global_margins": margins.copy(),
            "signs": signs,
            "learned": cpu.arrays["learned"],
            "cost": cost.astype(np.float64),
            "total_write": total_write.astype(np.float64),
            "episode_indices": cpu.arrays["episode_indices"],
            **_storage(terminal, model.alpha),
        }
        for key, value in row.items():
            collected.setdefault(key, []).append(value)
        fields.append(_field(model, terminal, cpu))
        if keep_p:
            selected = np.argsort(cpu.arrays["episode_indices"])[:2]
            states, _, modulations, _ = support_trajectory(model, cpu)
            full_prefix = torch.stack(states).cpu().numpy()
            full_modulations = torch.stack(modulations).cpu().numpy()
            internal[name] = {
                "terminal_P": terminal.cpu().numpy(),
                "full_prefix_P": full_prefix,
                "prefix_P": full_prefix[:, selected],
                "natural_modulation": full_modulations[:, selected],
                "selected_positions": selected.astype(np.int64),
                "selected_episode_indices": cpu.arrays["episode_indices"][selected],
                "source_batch_size": np.asarray(
                    len(cpu.arrays["item_codes"]), dtype=np.int64
                ),
                "source_support_shape": np.asarray(
                    cpu.arrays["support_inputs"].shape, dtype=np.int64
                ),
                "source_query_shape": np.asarray(
                    cpu.arrays["query_inputs"].shape, dtype=np.int64
                ),
            }
        del terminal
    arrays = {key: np.concatenate(value) for key, value in collected.items()}
    summary = summarize_generic(arrays, recipe, sample_seed)
    global_arrays = {**arrays, "margins": arrays["global_margins"]}
    summary["global"] = summarize_generic(global_arrays, recipe, sample_seed)
    arrays["global_ce"] = global_arrays["ce"]
    return arrays, summary, np.concatenate(fields), internal


def _liu(model, record: dict, observation: str, recipe: dict, keep_p: bool):
    cpu = load_cpu(record, observation, recipe)
    task = size_protocol(recipe, 8)
    terminal, intact, cost, total_write = _rollout(model, cpu)
    zero = torch.zeros_like(terminal)
    p_off = read_original(model, zero, cpu).astype(np.float64)
    shuffled, route = shuffle_evidence(
        cpu, task.support_blocks, recipe["evaluation"]["liu"]["evidence_shuffle_seed"]
    )
    shuffled_terminal, shuffled_margins, _, _ = _rollout(model, shuffled)
    del shuffled_terminal
    removed = []
    for relation in task.support_pairs_higher_lower:
        changed_terminal, changed_margins, _, _ = _rollout(
            model, remove_relation(cpu, relation)
        )
        removed.append(changed_margins)
        del changed_terminal
    raw = {
        "bundles": {
            "intact": {"logits": intact},
            "local_off": {"logits": intact.copy()},
            "P_off": {"logits": p_off},
            "evidence_shuffle": {"logits": shuffled_margins},
        },
        "removed": np.stack(removed),
        "removed_global": np.stack(removed),
        "evidence_route": route,
        "cost": cost,
        "total_write": total_write,
        "storage": _storage(terminal, model.alpha),
    }
    internal = (
        {
            "terminal_P": terminal.cpu().numpy(),
            "source_batch_size": np.asarray(
                len(cpu.arrays["item_codes"]), dtype=np.int64
            ),
            "source_support_shape": np.asarray(
                cpu.arrays["support_inputs"].shape, dtype=np.int64
            ),
            "source_query_shape": np.asarray(
                cpu.arrays["query_inputs"].shape, dtype=np.int64
            ),
        }
        if keep_p
        else {}
    )
    field = _field(model, terminal, cpu)
    return cpu, raw, field, internal


def evaluate_unit(
    model,
    inputs: dict[str, dict],
    *,
    observation: str,
    panel: int,
    seed: int,
    keep_p: bool,
) -> tuple[dict, dict, dict]:
    recipe = inherited_recipe(panel)
    generic_raw, generic_result, generic_fields, generic_internal = _generic(
        model, inputs, observation, recipe, keep_p, seed + panel * 1000000
    )
    liu_cpu, liu_raw, liu_field, liu_internal = _liu(
        model, inputs["liu-8"], observation, recipe, keep_p
    )
    liu_result, liu_analysis, _ = primary_analysis(
        liu_raw,
        liu_cpu,
        size_protocol(recipe, 8),
        recipe,
        seed + panel * 1000000,
        generic_result,
    )
    external = {
        "generic": generic_raw,
        "liu": flatten_arrays({**liu_raw, **liu_analysis}),
        "stage1": {"generic_fields": generic_fields, "liu_fields": liu_field},
    }
    result = {"generic": generic_result, "liu": liu_result}
    internal = {"generic": generic_internal, "liu": liu_internal}
    gc.collect()
    torch.cuda.empty_cache()
    return external, result, internal


__all__ = ["CELLS", "evaluate_unit"]
