"""Predeclared equivalence and separate directional failure comparisons."""

import numpy as np

from fsrl.experiments.cohort_diagnostic.statistics import reference_intervals
from fsrl.experiments.structural_identification.evaluation import mean_interval


def equivalence(interval, bound):
    if interval["lower"] is None:
        return "unresolved"
    if interval["lower"] > -bound and interval["upper"] < bound:
        return "equivalent"
    if interval["lower"] > bound or interval["upper"] < -bound:
        return "meaningfully_different"
    return "unresolved"


def continuous_contrast(a, b, seed):
    return {
        key: mean_interval(
            np.asarray([row[key] for row in a], float)
            - np.asarray([row[key] for row in b], float),
            seed,
        )
        for key in a[0]
    }


def flag_failure(row, name):
    value = row["flags"][name]
    return not (value["qualitative"] and value["calibration"])


def outside(value, interval, side):
    if value is None or not np.isfinite(value):
        return None
    return value < interval["lower"] if side == "below" else value > interval["upper"]


def failure_contrast(a, b, refs, seed):
    flags = {
        key: mean_interval(
            np.asarray([flag_failure(row, key) for row in a], float)
            - np.asarray([flag_failure(row, key) for row in b], float),
            seed,
        )
        for key in a[0]["flags"]
    }
    directions = {}
    for key, interval in reference_intervals(refs).items():
        directions[key] = {
            side: mean_interval(
                np.asarray(
                    [outside(row["values"][key], interval, side) for row in a], float
                )
                - np.asarray(
                    [outside(row["values"][key], interval, side) for row in b], float
                ),
                seed,
            )
            for side in ("below", "above")
        }
    return {"row_joint_failure": flags, "directional_failure": directions}


def sustained_outside(value, interval):
    return value["lower"] is not None and (
        value["lower"] > interval["upper"] or value["upper"] < interval["lower"]
    )


def new_mismatches(gaussian, quantized, refs):
    return [
        key
        for key, interval in reference_intervals(refs).items()
        if sustained_outside(gaussian[key], interval)
        and not sustained_outside(quantized[key], interval)
    ]


def improvement(common_profile, contrast, failures, new_outside, refs):
    accuracy = common_profile["estimates"]["nonlearned_accuracy"]
    target = reference_intervals(refs)["nonlearned_accuracy"]
    changes = failures["directional_failure"]
    return bool(
        accuracy["lower"] is not None
        and accuracy["lower"] >= target["lower"]
        and accuracy["upper"] <= target["upper"]
        and equivalence(contrast["nonlearned_accuracy"], 0.005) == "equivalent"
        and changes["correct_ranker"]["below"]["upper"] is not None
        and changes["correct_ranker"]["below"]["upper"] < 0
        and changes["self_consistent_incorrect"]["above"]["upper"] is not None
        and changes["self_consistent_incorrect"]["above"]["upper"] < 0
        and not new_outside
    )
