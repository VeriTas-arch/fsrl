"""Analytic propagation of the frozen cross-path margin error budget."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fsrl.analysis.hodge import build_complete_graph_geometry
from fsrl.experiments.memory_structure.inputs import size_protocol
from fsrl.experiments.training_strategy.summaries import liu_endpoints

ABSOLUTE = 1e-5
RELATIVE = 1e-5
RECONSTRUCTION = 1e-10
EDGE_SLACK = 1e-12


def margin_budget(reference: np.ndarray) -> np.ndarray:
    return ABSOLUTE + RELATIVE * np.abs(np.asarray(reference, dtype=np.float64))


def _metadata(left: np.ndarray, right: np.ndarray, name: str) -> np.ndarray:
    observed, reference = np.asarray(left), np.asarray(right)
    if observed.shape != reference.shape:
        raise RuntimeError(f"{name} shape differs")
    finite = np.isfinite(reference)
    if not np.array_equal(np.isfinite(observed), finite):
        raise RuntimeError(f"{name} finite mask differs")
    return finite


def bounded_error(
    observed: np.ndarray, reference: np.ndarray, budget: np.ndarray, name: str
) -> float:
    observed_array = np.asarray(observed)
    reference_array = np.asarray(reference)
    finite = _metadata(observed_array, reference_array, name)
    allowed = np.broadcast_to(
        np.asarray(budget, dtype=np.float64), reference_array.shape
    )
    error = np.abs(observed_array - reference_array)
    if np.any(error[finite] > allowed[finite]):
        excess = np.full(error.shape, -np.inf, dtype=np.float64)
        excess[finite] = error[finite] - allowed[finite]
        index = np.unravel_index(np.argmax(excess), error.shape)
        raise RuntimeError(
            f"{name} exceeds propagated budget at {index}: "
            f"error={error[index]}, budget={allowed[index]}"
        )
    return float(np.max(error[finite], initial=0.0))


def reconstructed_error(
    observed: np.ndarray, reconstructed: np.ndarray, name: str
) -> float:
    return bounded_error(observed, reconstructed, RECONSTRUCTION, name)


def probability_budget(reference_margin: np.ndarray, temperature: float) -> np.ndarray:
    return margin_budget(reference_margin) / (4.0 * temperature) + EDGE_SLACK


def cross_entropy_budget(
    reference_margin: np.ndarray, temperature: float
) -> np.ndarray:
    return margin_budget(reference_margin) / temperature + EDGE_SLACK


def masked_mean_budget(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    budget = np.asarray(values, dtype=np.float64)
    selected = np.asarray(mask, dtype=bool)
    count = selected.sum(axis=1)
    return (
        np.divide(
            np.sum(budget * selected, axis=1),
            count,
            out=np.full(len(budget), np.nan, dtype=np.float64),
            where=count > 0,
        )
        + EDGE_SLACK
    )


def generic_probability_budgets(raw: dict) -> dict[str, np.ndarray]:
    budget = probability_budget(raw["margins"], 1.0)
    learned = np.asarray(raw["learned"], dtype=bool)
    return {
        "generic_learned": masked_mean_budget(budget, learned),
        "generic_nonlearned": masked_mean_budget(budget, ~learned),
    }


def liu_probability_budgets(
    margins: np.ndarray, cpu, recipe: dict
) -> dict[str, np.ndarray]:
    protocol = size_protocol(recipe, 8)
    geometry = build_complete_graph_geometry(protocol)
    subjects = margins.shape[0]
    retention = np.asarray(cpu.arrays["retention"], dtype=bool)
    learned_28 = np.asarray([pair in protocol.learned_pairs for pair in geometry.pairs])
    retained_28 = np.zeros((subjects, len(geometry.pairs)), dtype=bool)
    for index, relation in enumerate(protocol.support_pairs_higher_lower):
        retained_28[:, geometry.pairs.index(tuple(sorted(relation)))] = retention[
            :, index
        ]
    learned = np.repeat(learned_28, 2)[None].repeat(subjects, axis=0)
    omitted = np.repeat(learned_28[None] & ~retained_28, 2, axis=1)
    budget = probability_budget(margins, recipe["evaluation"]["liu"]["temperature"])
    return {
        "liu_learned": masked_mean_budget(budget, learned),
        "liu_nonlearned": masked_mean_budget(budget, ~learned),
        "liu_omitted": masked_mean_budget(budget, omitted),
    }


def liu_reconstruction(raw: dict, cpu, recipe: dict) -> dict[str, np.ndarray]:
    endpoints = liu_endpoints(
        {"intact": {"logits": raw["bundles__intact__logits"]}},
        cpu.arrays["retention"],
        size_protocol(recipe, 8),
        recipe["evaluation"]["liu"]["temperature"],
    )
    return {
        f"liu_{name}": value
        for name, value in endpoints["intact"]["probability"].items()
        if name in {"learned", "nonlearned", "omitted"}
    }


def _margin_arrays(raw: dict) -> dict[str, np.ndarray]:
    result = {
        "generic/margins": raw["generic"]["margins"],
        "generic/global_margins": raw["generic"]["global_margins"],
    }
    for name, value in raw["liu"].items():
        if (
            name.startswith("bundles__")
            and name.endswith("__logits")
            or name in {"removed", "removed_global"}
        ):
            result[f"liu/{name}"] = value
    return result


@dataclass(frozen=True)
class BudgetCaps:
    generic_probability: float
    liu_probability: float
    generic_ce: float
    liu_ce: float
    coherence: float

    @classmethod
    def from_parent_raw(cls, raw: dict) -> BudgetCaps:
        generic = margin_budget(raw["generic"]["margins"])
        liu = np.concatenate(
            [
                value.reshape(-1)
                for key, value in _margin_arrays(raw).items()
                if key.startswith("liu/")
            ]
        )
        liu_margin = margin_budget(liu)
        coherence = max(
            coherence_budget(raw["liu"][f"bundles__{name}__logits"])
            for name in ("intact", "local_off")
        )
        return cls(
            generic_probability=float(generic.max() / 4.0 + 2 * EDGE_SLACK),
            liu_probability=float(liu_margin.max() + 2 * EDGE_SLACK),
            generic_ce=float(generic.max() + 2 * EDGE_SLACK),
            liu_ce=float(4.0 * liu_margin.max() + 2 * EDGE_SLACK),
            coherence=coherence,
        )

    def merged(self, other: BudgetCaps) -> BudgetCaps:
        return BudgetCaps(
            **{
                name: max(getattr(self, name), getattr(other, name))
                for name in self.__dataclass_fields__
            }
        )


def audit_unit(
    direct: dict, parent: dict, liu_cpu, recipe: dict
) -> tuple[dict, BudgetCaps]:
    discrete_arrays = assert_discrete_equal(direct, parent)
    direct_margins, parent_margins = _margin_arrays(direct), _margin_arrays(parent)
    if direct_margins.keys() != parent_margins.keys():
        raise RuntimeError("margin inventory differs")
    errors = {
        name: bounded_error(
            direct_margins[name],
            parent_margins[name],
            margin_budget(parent_margins[name]),
            name,
        )
        for name in direct_margins
    }
    for label, raw in (("parent", parent), ("direct", direct)):
        generic = raw["generic"]
        ce = np.logaddexp(0.0, -generic["margins"] * generic["signs"]).mean(1)
        reconstructed_error(generic["ce"], ce, f"{label} generic CE reconstruction")
        rebuilt = liu_reconstruction(raw["liu"], liu_cpu, recipe)
        for name, value in rebuilt.items():
            reconstructed_error(
                raw["liu"][
                    f"endpoints__intact__probability__{name.removeprefix('liu_')}"
                ],
                value,
                f"{label} {name} reconstruction",
            )
    generic_budgets = generic_probability_budgets(parent["generic"])
    liu_budgets = liu_probability_budgets(
        parent["liu"]["bundles__intact__logits"], liu_cpu, recipe
    )
    endpoint_errors = {}
    direct_liu = liu_reconstruction(direct["liu"], liu_cpu, recipe)
    parent_liu = liu_reconstruction(parent["liu"], liu_cpu, recipe)
    for name, budget in {**generic_budgets, **liu_budgets}.items():
        if name.startswith("generic_"):
            learned = parent["generic"]["learned"]
            mask = (
                learned
                if name.endswith("learned") and not name.endswith("nonlearned")
                else ~learned
            )
            signs = parent["generic"]["signs"]
            direct_probability = 1.0 / (
                1.0 + np.exp(-direct["generic"]["margins"] * signs)
            )
            parent_probability = 1.0 / (
                1.0 + np.exp(-parent["generic"]["margins"] * signs)
            )
            direct_value = np.sum(direct_probability * mask, axis=1) / mask.sum(axis=1)
            parent_value = np.sum(parent_probability * mask, axis=1) / mask.sum(axis=1)
        else:
            direct_value, parent_value = direct_liu[name], parent_liu[name]
        endpoint_errors[name] = bounded_error(
            direct_value, parent_value, budget, f"endpoint {name}"
        )
    return {
        "maximum_margin_error": max(errors.values()),
        "maximum_endpoint_error": max(endpoint_errors.values()),
        "margin_arrays": len(errors),
        "probability_endpoints": len(endpoint_errors),
        "discrete_arrays": discrete_arrays,
    }, BudgetCaps.from_parent_raw(parent)


def merge_caps(values: list[BudgetCaps]) -> BudgetCaps:
    if not values:
        raise ValueError("cannot merge an empty budget inventory")
    result = values[0]
    for value in values[1:]:
        result = result.merged(value)
    return result


def coherence_budget(reference_margin: np.ndarray) -> float:
    margins = np.asarray(reference_margin, dtype=np.float64)
    budget = margin_budget(margins)
    field = (margins[:, ::2] - margins[:, 1::2]) / 2.0
    field_budget = (budget[:, ::2] + budget[:, 1::2]) / 2.0
    delta = np.linalg.vector_norm(field_budget, axis=1)
    norm = np.linalg.vector_norm(field, axis=1)
    if np.any(norm <= delta):
        raise RuntimeError("coherence parent field is too small for perturbation bound")
    return float(np.minimum(1.0, 4.0 * delta / (norm - delta)).max() + 3 * EDGE_SLACK)


def assert_discrete_equal(direct: dict, parent: dict) -> int:
    names = (
        "generic/signs",
        "generic/learned",
        "generic/episode_indices",
        "liu/evidence_route",
        "liu/routes__full__sampled_orders",
        "liu/routes__full__sampled_mask",
        "liu/routes__full__internal__orders",
        "liu/routes__global__sampled_orders",
        "liu/routes__global__sampled_mask",
        "liu/routes__global__internal__orders",
    )
    for name in names:
        phase, key = name.split("/", 1)
        if key not in direct[phase] or key not in parent[phase]:
            raise RuntimeError(f"discrete external array is missing: {name}")
        if not np.array_equal(direct[phase][key], parent[phase][key]):
            raise RuntimeError(f"discrete external array differs: {name}")
    return len(names)


def summary_budget(path: str, caps: BudgetCaps) -> float:
    segments = set(path.split("/"))
    factor = (
        4.0
        if "/interaction/" in path
        else 2.0
        if any(
            token in path
            for token in ("/error_A/", "/error_C/", "/training_under_error/")
        )
        else 1.0
    )
    if "/noninferiority_equal_panel_mean/" in path:
        base = caps.liu_probability if "/liu_" in path else caps.generic_probability
        return 2.0 * base + 6.0 * EDGE_SLACK
    if "coherence" in segments and "behavior" in segments:
        return caps.coherence + 6.0 * EDGE_SLACK
    if (
        "CE" in segments
        or "cross_entropy" in segments
        or segments.intersection(
            {"generic_full", "generic_global", "liu_full", "liu_global"}
        )
    ):
        base = (
            caps.liu_ce
            if segments.intersection({"liu_full", "liu_global"})
            else caps.generic_ce
        )
        return factor * base + 6.0 * EDGE_SLACK
    if "/effects/" in path or "/summaries/" in path:
        return 2.0 * caps.liu_probability + 6.0 * EDGE_SLACK
    return RECONSTRUCTION


__all__ = [
    "ABSOLUTE",
    "EDGE_SLACK",
    "RECONSTRUCTION",
    "RELATIVE",
    "BudgetCaps",
    "assert_discrete_equal",
    "audit_unit",
    "bounded_error",
    "coherence_budget",
    "cross_entropy_budget",
    "margin_budget",
    "merge_caps",
    "probability_budget",
    "reconstructed_error",
    "summary_budget",
]
