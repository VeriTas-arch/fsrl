"""Analytic propagation of the frozen cross-path margin error budget."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fsrl.experiments.single_p_time_role.analysis import liu_endpoints

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
    finite = _metadata(observed, reference, name)
    allowed = np.broadcast_to(np.asarray(budget, dtype=np.float64), reference.shape)
    error = np.abs(np.asarray(observed) - np.asarray(reference))
    if np.any(error[finite] > allowed[finite]):
        index = np.unravel_index(np.argmax(error - allowed), error.shape)
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


def liu_probability_budgets(margins: np.ndarray, cpu) -> dict[str, np.ndarray]:
    subjects = margins.shape[0]
    query_pairs = np.asarray(cpu.arrays["query_pairs"])
    support_pairs = np.asarray(cpu.arrays["support_pairs"])
    retention = np.asarray(cpu.arrays["retention"], dtype=bool)
    support = [tuple(sorted(map(int, pair))) for pair in support_pairs[:8, 0]]
    relation_index = {pair: index for index, pair in enumerate(support)}
    indices = np.asarray(
        [relation_index.get(tuple(sorted(map(int, pair))), -1) for pair in query_pairs]
    )
    learned = np.broadcast_to(indices >= 0, (subjects, len(indices)))
    omitted = np.zeros_like(learned)
    for query, relation in enumerate(indices):
        if relation >= 0:
            omitted[:, query] = ~retention[:, relation]
    budget = probability_budget(margins, 0.25)
    return {
        "liu_learned": masked_mean_budget(budget, learned),
        "liu_nonlearned": masked_mean_budget(budget, ~learned),
        "liu_omitted": masked_mean_budget(budget, omitted),
    }


def liu_reconstruction(raw: dict, cpu) -> dict[str, np.ndarray]:
    return liu_endpoints(
        raw["bundles__intact__logits"],
        cpu.arrays["targets"],
        cpu.arrays["query_pairs"],
        cpu.arrays["support_pairs"],
        cpu.arrays["retention"],
    )


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
        return cls(
            generic_probability=float(generic.max() / 4.0 + 2 * EDGE_SLACK),
            liu_probability=float(liu_margin.max() + 2 * EDGE_SLACK),
            generic_ce=float(generic.max() + 2 * EDGE_SLACK),
            liu_ce=float(4.0 * liu_margin.max() + 2 * EDGE_SLACK),
        )

    def merged(self, other: BudgetCaps) -> BudgetCaps:
        return BudgetCaps(
            **{
                name: max(getattr(self, name), getattr(other, name))
                for name in self.__dataclass_fields__
            }
        )


def audit_unit(direct: dict, parent: dict, liu_cpu) -> tuple[dict, BudgetCaps]:
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
        rebuilt = liu_reconstruction(raw["liu"], liu_cpu)
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
        parent["liu"]["bundles__intact__logits"], liu_cpu
    )
    endpoint_errors = {}
    direct_liu = liu_reconstruction(direct["liu"], liu_cpu)
    parent_liu = liu_reconstruction(parent["liu"], liu_cpu)
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
    }, BudgetCaps.from_parent_raw(parent)


def merge_caps(values: list[BudgetCaps]) -> BudgetCaps:
    if not values:
        raise ValueError("cannot merge an empty budget inventory")
    result = values[0]
    for value in values[1:]:
        result = result.merged(value)
    return result


def summary_budget(path: str, caps: BudgetCaps) -> float:
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
    if any(
        token in path
        for token in (
            "/CE/",
            "/cross_entropy/",
            "/generic_full/",
            "/generic_global/",
            "/liu_full/",
            "/liu_global/",
        )
    ):
        base = caps.liu_ce if "/liu_" in path else caps.generic_ce
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
    "audit_unit",
    "bounded_error",
    "cross_entropy_budget",
    "margin_budget",
    "merge_caps",
    "probability_budget",
    "reconstructed_error",
    "summary_budget",
]
