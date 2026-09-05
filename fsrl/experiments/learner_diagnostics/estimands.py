"""Fixed task scoring, participant contrasts, and readout accounting."""

from __future__ import annotations

import numpy as np

from fsrl.analysis.hodge import build_complete_graph_geometry
from fsrl.analysis.statistics import stable_sigmoid
from fsrl.experiments.training_strategy.estimands import estimate, subject_means


def scoring(protocol, retention: np.ndarray) -> dict:
    geometry = build_complete_graph_geometry(protocol)
    pairs = np.asarray(geometry.pairs)
    learned = np.asarray([pair in protocol.learned_pairs for pair in geometry.pairs])
    retained = np.zeros((len(retention), len(pairs)), dtype=bool)
    for r, pair in enumerate(protocol.support_pairs_higher_lower):
        retained[:, geometry.pairs.index(tuple(sorted(pair)))] = retention[:, r]
    positions = np.empty(protocol.n_items, dtype=int)
    positions[list(protocol.true_order_high_to_low)] = np.arange(protocol.n_items)
    distance = np.abs(positions[pairs[:, 0]] - positions[pairs[:, 1]])
    centered = distance[~learned] - distance[~learned].mean()
    slope_weight = np.zeros(len(pairs))
    slope_weight[~learned] = centered / np.square(centered).sum()
    position_weight = np.zeros((protocol.n_items, len(pairs)))
    for position, item in enumerate(protocol.true_order_high_to_low):
        selected = np.any(pairs == item, axis=1) & ~learned
        position_weight[position, selected] = 1 / selected.sum()
    serial_weight = position_weight[[0, -1]].mean(0) - position_weight[1:-1].mean(0)
    return {
        "signs": (geometry.true_sign[:, None] * [1, -1]).reshape(-1),
        "groups": {
            "overall": np.ones(len(pairs), dtype=bool),
            "learned": learned,
            "nonlearned": ~learned,
            "retained": retained,
            "omitted": learned & ~retained,
        },
        "distance": distance,
        "slope_weight": slope_weight,
        "serial_weight": serial_weight,
    }


def pair_average(oriented: np.ndarray) -> np.ndarray:
    return oriented.reshape(len(oriented), -1, 2).mean(axis=-1)


def grouped(oriented: np.ndarray, context: dict) -> dict:
    pair = pair_average(oriented)
    return {name: subject_means(pair, mask) for name, mask in context["groups"].items()}


def endpoints(margin, gain, context, temperature, tie_tolerance) -> tuple[dict, dict]:
    correct = margin * context["signs"]
    probability = pair_average(stable_sigmoid(correct / temperature))
    exact = pair_average((np.sign(correct) + 1) / 2)
    values = {}
    for measure, pair in (("probability", probability), ("exact", exact)):
        values.update(
            {
                f"{measure}/{name}": subject_means(pair, mask)
                for name, mask in context["groups"].items()
            }
        )
        values[f"{measure}/distance_slope"] = pair @ context["slope_weight"]
        values[f"{measure}/serial_contrast"] = pair @ context["serial_weight"]
    raw = correct[:, ::2] / gain
    adjacent = context["distance"] == 1
    values["latent/strict_correct_order"] = np.all(
        raw[:, adjacent] > tie_tolerance, 1
    ).astype(float)
    values["latent/has_tied_pair"] = np.any(np.abs(raw) <= tie_tolerance, 1).astype(
        float
    )
    values["latent/pair_discordance"] = (
        (raw < -tie_tolerance) + 0.5 * (np.abs(raw) <= tie_tolerance)
    ).mean(1)
    return values, {"probability": probability, "exact": exact}


def global_contrasts(cells: dict) -> dict:
    differences = {
        "admission_at_finite": ("AF", "RF"),
        "integration_at_retained": ("RL", "RF"),
        "admission_at_least_squares": ("AL", "RL"),
        "integration_at_all": ("AL", "AF"),
        "total": ("AL", "RF"),
    }
    result = {
        name: {key: cells[a][key] - cells[b][key] for key in cells[a]}
        for name, (a, b) in differences.items()
    }
    result["interaction"] = {
        key: result["admission_at_least_squares"][key]
        - result["admission_at_finite"][key]
        for key in cells["RF"]
    }
    return result


def readout_accounting(margin: np.ndarray, context: dict, temperature: float) -> dict:
    correct = margin * context["signs"]
    p = stable_sigmoid(correct / temperature)
    contributions = {
        "correct_shortfall": np.where(correct > 0, p - 1, 0),
        "wrong_rescue": np.where(correct < 0, p, 0),
        "ties": np.where(correct == 0, p - 0.5, 0),
        "total": p - (np.sign(correct) + 1) / 2,
        "tie_fraction": (correct == 0).astype(float),
    }
    return {name: grouped(values, context) for name, values in contributions.items()}


def summarize(values: dict, seed: int, statistics: dict) -> dict:
    return {
        name: estimate(np.asarray(row, dtype=float), seed=seed, statistics=statistics)
        for name, row in values.items()
    }


def direction(rows: list[dict]) -> str:
    lows = [row["bootstrap"]["lower"] for row in rows]
    highs = [row["bootstrap"]["upper"] for row in rows]
    if all(value is not None and value > 0 for value in lows):
        return "consistently_positive"
    if all(value is not None and value < 0 for value in highs):
        return "consistently_negative"
    return "mixed_or_uncertain"
