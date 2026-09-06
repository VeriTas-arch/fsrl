"""Paired contrasts and fixed-reference whole-participant diagnostics."""

import numpy as np

from fsrl.experiments.cohort_diagnostic.statistics import reference_intervals, wilson
from fsrl.experiments.minimal_learner.locks import validate_complete
from fsrl.experiments.structural_identification.evaluation import mean_interval
from fsrl.experiments.structural_identification.inputs import save_arrays
from fsrl.experiments.structural_identification.measurement import human_choices
from fsrl.experiments.structural_identification.observation import record
from fsrl.experiments.structural_identification.reporting import flag_pass
from fsrl.infra.provenance import load_json, write_json_exclusive
from fsrl.tasks.protocol_catalog import load_registered_protocol

from .execution import unit
from .protocol import RUN_ROOT, SEEDS, parent_arrays, parent_result, validate_source


def signed_counts(values, interval):
    counts = dict.fromkeys(("below", "inside", "above", "undefined"), 0)
    for value in values:
        if value is None or not np.isfinite(value):
            counts["undefined"] += 1
        elif value < interval["lower"]:
            counts["below"] += 1
        elif value > interval["upper"]:
            counts["above"] += 1
        else:
            counts["inside"] += 1
    return counts


def failure_profile(rows, refs):
    names = list(rows[0]["flags"])
    failures = np.asarray(
        [
            [
                not (
                    row["flags"][name]["qualitative"]
                    and row["flags"][name]["calibration"]
                )
                for name in names
            ]
            for row in rows
        ],
        dtype=np.int64,
    )
    return {
        "joint_rate": wilson(~failures.any(axis=1)),
        "qualitative_rate": wilson([flag_pass(row, "qualitative") for row in rows]),
        "failure_rates": {name: wilson(failures[:, i]) for i, name in enumerate(names)},
        "cofailure_names": names,
        "cofailure_counts": (failures.T @ failures).tolist(),
        "signed_counts": {
            name: signed_counts([row["values"][name] for row in rows], interval)
            for name, interval in reference_intervals(refs).items()
        },
    }


def finite_cohort():
    validate_source()
    choices = human_choices()
    indices = np.random.default_rng(8410001).integers(0, 77, size=(1000, 77))
    protocol = load_registered_protocol("liu_v2")
    refs = parent_result()["measurement"]["references"]
    for start in range(0, 1000, 100):
        directory = RUN_ROOT / "finite_cohort" / f"chunk-{start:03}"
        if directory.exists():
            validate_complete(directory)
            continue
        with unit(directory, "finite_cohort", {"start": start, "seed": 8410001}):
            selected = indices[start : start + 100]
            save_arrays(directory / "indices.npz", indices=selected)
            rows = [record(choices[index], protocol, refs) for index in selected]
            write_json_exclusive(directory / "result.json", rows)
        print("human bootstrap", start + 100, "complete", flush=True)
    return {
        "cohorts": 1000,
        "participants_per_cohort": 77,
        "reference_reestimated": False,
    }


def paired_change(full_new, full_base, simple_new, simple_base, seed):
    values = np.asarray(full_new) - full_base - (np.asarray(simple_new) - simple_base)
    return values, mean_interval(values, seed)


def manipulation_seed(seed):
    models = {
        st: parent_arrays(f"manipulations/{seed}-{st}-decay/outputs.npz")
        for st in ("M10", "M11")
    }
    arrays, estimates = {}, {}
    for condition in ("balanced_K8", "clustered_K4", "reverse_clustered_K4"):
        estimates[condition] = {}
        for domain in ("cross", "bridge"):
            observations = {}
            for st, model in models.items():
                for level in (condition, "balanced_K4"):
                    mask = model[f"{level}__{domain}"]
                    np.testing.assert_array_equal(
                        mask, models["M10"][f"balanced_K4__{domain}"]
                    )
                    np.testing.assert_array_equal(
                        mask.sum(axis=1), 16 if domain == "cross" else 1
                    )
                    correct = model[f"{level}__sampled_correct"]
                    observations[(st, level)] = (correct * mask).sum(axis=1) / mask.sum(
                        axis=1
                    )
            values, estimate = paired_change(
                observations[("M11", condition)],
                observations[("M11", "balanced_K4")],
                observations[("M10", condition)],
                observations[("M10", "balanced_K4")],
                8420000 + seed,
            )
            arrays[f"{seed}__{condition}__{domain}"] = values
            estimates[condition][domain] = estimate
    return estimates, arrays


def manipulations():
    validate_source()
    directory = RUN_ROOT / "manipulations"
    if directory.exists():
        validate_complete(directory)
        return load_json(directory / "result.json")
    with unit(directory, "manipulations", {"seeds": list(SEEDS)}):
        results, arrays = {}, {}
        for seed in SEEDS:
            results[str(seed)], values = manipulation_seed(seed)
            arrays.update(values)
        save_arrays(directory / "outputs.npz", **arrays)
        write_json_exclusive(directory / "result.json", results)
    return results
