"""Conditional predeclared K4/K8 transport at fixed parameter donors."""

import numpy as np

from fsrl.experiments.encoding_contribution.protocol import parameters, parent_arrays
from fsrl.experiments.minimal_learner.locks import validate_complete
from fsrl.experiments.minimal_learner.training import runtime
from fsrl.experiments.structural_identification.evaluation import (
    load_runner,
    manipulation_record,
    mean_interval,
    predict,
)
from fsrl.experiments.structural_identification.inputs import save_arrays
from fsrl.experiments.structural_identification.model import encode
from fsrl.infra.provenance import load_json, write_json_exclusive

from .encoding import gaussian
from .execution import unit
from .protocol import RUN_ROOT, SEEDS, STRUCTURES, cell, input_batch, validate_source
from .statistics import equivalence


def replay(seed, donor):
    directory = RUN_ROOT / "K_check" / f"{seed}-theta_{donor}"
    if directory.exists():
        validate_complete(directory)
        with np.load(directory / "outputs.npz", allow_pickle=False) as saved:
            return {key: saved[key] for key in saved.files}
    runner = load_runner(parameters(seed, donor))
    arrays = {}
    with unit(directory, {"seed": seed, "donor": donor}):
        for condition in ("balanced_K4", "balanced_K8"):
            base, uniforms = input_batch(f"manipulation-{condition}")
            for structure in ("M10", "M11", "G"):
                prefix = f"{structure}__{condition}__"
                if structure == donor:
                    previous = parent_arrays(
                        f"manipulations/{seed}-{structure}-decay/outputs.npz"
                    )
                    values = {
                        key.removeprefix(condition + "__"): value
                        for key, value in previous.items()
                        if key.startswith(condition + "__")
                    }
                else:
                    encoded = (
                        gaussian(base, uniforms)[0]
                        if structure == "G"
                        else encode(base, structure, uniforms)
                    )
                    margins, state = predict(runner, encoded)
                    values = {
                        **manipulation_record(margins, base, 1350011),
                        "margins": margins,
                        "w": state,
                    }
                arrays.update({prefix + key: value for key, value in values.items()})
        save_arrays(directory / "outputs.npz", **arrays)
    return arrays


def summarize_change(arrays, seed):
    changes, result = {}, {}
    for domain in ("cross", "bridge"):
        for structure in ("M10", "M11", "G"):
            levels = []
            for condition in ("balanced_K4", "balanced_K8"):
                prefix = f"{structure}__{condition}__"
                mask = arrays[prefix + domain]
                np.testing.assert_array_equal(
                    mask, arrays[f"M10__balanced_K4__{domain}"]
                )
                np.testing.assert_array_equal(
                    mask.sum(axis=1), 16 if domain == "cross" else 1
                )
                levels.append(
                    (arrays[prefix + "sampled_correct"] * mask).sum(axis=1)
                    / mask.sum(axis=1)
                )
            changes[structure] = levels[1] - levels[0]
        interval = mean_interval(changes["G"] - changes["M11"], 8510000 + seed)
        result[domain] = {
            "K8_minus_K4": {
                key: mean_interval(value, 8510000 + seed)
                for key, value in changes.items()
            },
            "G_minus_Q_change": interval,
            "status": equivalence(interval, 0.005)
            if domain == "cross"
            else "supporting",
        }
    return result


def run():
    validate_source()
    summary = load_json(RUN_ROOT / "summary.json")
    if not summary["K_check_trigger"]:
        result = {
            "status": "not_triggered",
            "reason": "Requires all six primary-equivalent qualified G cells and no new sustained continuous mismatch.",
            "new_rollouts": 0,
        }
    else:
        runtime()
        rows = {
            cell(seed, "G", donor): summarize_change(replay(seed, donor), seed)
            for seed in SEEDS
            for donor in STRUCTURES
        }
        result = {
            "status": "completed",
            "cells": rows,
            "persistent_cross_equivalence": all(
                row["cross"]["status"] == "equivalent" for row in rows.values()
            ),
        }
    write_json_exclusive(RUN_ROOT / "K_check.json", result)
    return result
