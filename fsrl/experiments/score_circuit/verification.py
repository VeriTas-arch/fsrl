"""Independent raw-endpoint/bootstrap reconstruction and decision audit."""

import numpy as np

from fsrl.experiments.training_strategy.behavior import (
    behavior_metrics,
    classify_rows,
    human_references,
)
from fsrl.experiments.training_strategy.locks import verify_reference
from fsrl.infra.file_contracts import validate_run_manifest
from fsrl.infra.provenance import load_json
from fsrl.tasks.protocol_catalog import load_registered_protocol

from .analysis import maximum_difference
from .decisions import decide_fit
from .evidence import load_arrays
from .execution import original_spec


def read_npz(ref: dict) -> dict:
    with np.load(verify_reference(ref), allow_pickle=False) as saved:
        result = {name: saved[name] for name in saved.files}
    if any(value.dtype.kind not in "biuf" for value in result.values()):
        raise RuntimeError("non-numeric scientific array")
    for name, value in result.items():
        invalid = (
            np.isinf(value) if name.startswith("endpoints__") else ~np.isfinite(value)
        )
        if np.any(invalid):
            raise RuntimeError(f"non-finite circuit array: {name}")
    return result


def independent_groups(inputs: dict) -> tuple:
    protocol = load_registered_protocol("liu_v2")
    rank = {item: r for r, item in enumerate(protocol.true_order_high_to_low)}
    pairs = inputs["liu__inputs__query_pairs"]
    signs = np.asarray(
        [
            [1 if rank[left] < rank[right] else -1 for left, right in row]
            for row in pairs
        ]
    )
    retained, learned = np.zeros(pairs.shape[:2], bool), np.zeros(pairs.shape[:2], bool)
    for s, row in enumerate(pairs):
        for q, (left, right) in enumerate(row):
            pair = tuple(sorted((left, right)))
            learned[s, q] = pair in protocol.learned_pairs
            matches = np.all(
                np.sort(inputs["liu__inputs__support_pairs"][s], axis=1) == pair, axis=1
            )
            retained[s, q] = np.any(inputs["liu__inputs__retention"][matches, s] > 0)
    return signs, {
        "overall": np.ones_like(learned),
        "learned": learned,
        "nonlearned": ~learned,
        "retained": retained,
        "omitted": learned & ~retained,
    }


def manual_endpoints(raw: dict, inputs: dict) -> dict:
    generic = np.empty(inputs["generic__signs"].shape, dtype=np.float64)
    for key, value in raw.items():
        if key.startswith("generic_") and key.endswith("__margin"):
            length = key.split("__")[0].split("_")[1]
            generic[inputs[f"generic__groups__{length}__episode_indices"]] = value
    signs, groups = independent_groups(inputs)
    domains = (
        ("liu", raw["liu__margin"], signs, groups, 0.25),
        (
            "generic",
            generic,
            inputs["generic__signs"],
            {
                "learned": inputs["generic__learned"],
                "nonlearned": ~inputs["generic__learned"],
            },
            1.0,
        ),
    )
    result = {}
    for domain, margin, sign, masks, temperature in domains:
        correct = margin * sign
        values = {
            "probability": np.exp(-np.logaddexp(0, -correct / temperature)),
            "exact_decision": (correct > 0).astype(float) + 0.5 * (correct == 0),
        }
        for measure, value in values.items():
            for group, mask in masks.items():
                result[f"{domain}/{measure}/{group}"] = np.asarray(
                    [
                        row[selected].mean() if selected.any() else np.nan
                        for row, selected in zip(value, mask, strict=True)
                    ]
                )
    return result


def check_estimate(values, record, seed) -> float:
    valid = np.isfinite(values)
    sample = values[valid]
    if record["subjects"] != len(sample) or record["total_subjects"] != len(values):
        raise RuntimeError("endpoint sample size differs")
    if record["excluded_subject_indices"] != np.flatnonzero(~valid).tolist():
        raise RuntimeError("undefined participants differ")
    counts = np.random.default_rng(98000 + seed).multinomial(
        len(sample), np.full(len(sample), 1 / len(sample)), size=10000
    )
    draws = (counts @ sample) / len(sample)
    expected = [sample.mean(), np.quantile(draws, 0.025), np.quantile(draws, 0.975)]
    observed = [
        record["mean"],
        record["bootstrap"]["lower"],
        record["bootstrap"]["upper"],
    ]
    error = maximum_difference(expected, observed)
    if error > 1e-10:
        raise RuntimeError(f"independent bootstrap differs: {error}")
    return error


def verify_physical_and_query(
    raw: dict, case: dict, inputs: dict, scale: float, gain: float
) -> None:
    for group, physical in case["physical"].items():
        state = raw[f"{group}__trajectory"]
        diag = raw[f"{group}__diagnostics"]
        width = (state.shape[-1] - 6) // 2
        rebuilt = [
            diag[:, 0].min(),
            -diag[:, 1].min(),
            diag[:, 2].min(),
            -diag[:, 3].min(),
            diag[:, 4].sum(),
        ]
        saved = [
            physical[name]
            for name in (
                "minimum_efficacy",
                "maximum_efficacy",
                "minimum_activity_rate",
                "maximum_pair_sum_error",
                "bound_engagements",
            )
        ]
        np.testing.assert_allclose(rebuilt, saved, atol=1e-12, rtol=0)
        if (
            state[..., : 2 * width].min() < rebuilt[0] - 1e-12
            or state[..., : 2 * width].max() > rebuilt[1] + 1e-12
        ):
            raise RuntimeError("recorded extrema do not enclose saved states")
        prefix = (
            "liu__inputs__"
            if group == "liu"
            else f"generic__groups__{group.split('_')[1]}__inputs__"
        )
        cue = inputs[prefix + "query_cues"]
        x = cue[..., :width] - cue[..., width:]
        final = state[:, -1]
        potential = np.einsum(
            "sf,sqf->sq", final[:, :width] - final[:, width : 2 * width], x
        )
        previous = final[:, 2 * width] - final[:, 2 * width + 1]
        margin = np.empty_like(potential)
        decay = np.exp(-0.1 / (0.002 * scale))
        for q in range(potential.shape[1]):
            previous = (1 - decay) * potential[:, q] + decay * previous
            margin[:, q] = gain * previous
        np.testing.assert_allclose(margin, raw[f"{group}__margin"], atol=1e-10, rtol=0)


def verify_checks(fit: dict) -> None:
    primary = read_npz(fit["cases"]["primary/4096"]["arrays"])
    checks = read_npz(fit["check_arrays"])
    observed = maximum_difference(
        checks["affine_trajectory"], primary["liu__trajectory"]
    )
    np.testing.assert_allclose(
        observed, fit["reference_checks"]["affine_max_error"], atol=1e-12, rtol=0
    )
    for name, expected in fit["reference_checks"]["query_errors"].items():
        observed = maximum_difference(
            checks[f"query_margins__{name}"], primary["liu__margin"]
        )
        np.testing.assert_allclose(observed, expected, atol=1e-12, rtol=0)
    for scale in ("fast", "primary", "slow"):
        coarse = read_npz(fit["cases"][f"{scale}/4096"]["arrays"])
        fine = read_npz(fit["cases"][f"{scale}/8192"]["arrays"])
        for group, expected in fit["refinement"][scale].items():
            y, other = coarse[f"{group}__trajectory"], fine[f"{group}__trajectory"]
            width = (y.shape[-1] - 6) // 2
            observed = [
                maximum_difference(
                    y[..., :width] - y[..., width : 2 * width],
                    other[..., :width] - other[..., width : 2 * width],
                ),
                maximum_difference(
                    coarse[f"{group}__margin"], fine[f"{group}__margin"]
                ),
            ]
            np.testing.assert_allclose(
                observed, [expected["state"], expected["margin"]], atol=1e-12, rtol=0
            )
    for name in ("teacher_off", "mismatch_clamp"):
        raw = read_npz(fit["cases"][name]["arrays"])
        unchanged = all(
            np.all(value[..., :30] == 1)
            for key, value in raw.items()
            if key.endswith("__trajectory")
        )
        if unchanged != fit["control_no_write"][name]:
            raise RuntimeError("no-write control record differs")


def verify_behavior(seed: int, fit: dict) -> None:
    spec = original_spec()
    behavior = load_json(verify_reference(fit["sampled_behavior"]))
    record = behavior_metrics(behavior, 85000 + seed, spec["statistics"])
    if record["metrics"] != fit["behavior"]["metrics"]:
        raise RuntimeError("saved behavioral metrics do not reconstruct")
    flags = classify_rows(record["metrics"], human_references(spec))
    if flags != fit["behavior"]["flags"]:
        raise RuntimeError("nine frozen behavioral classifiers differ")


def verify_fit(seed: int, fit: dict) -> dict:
    inputs = load_arrays(seed)
    reference = read_npz(fit["reference"]["arrays"])
    original = manual_endpoints(reference, inputs)
    error, count = 0.0, 0
    for name, value in original.items():
        error = max(
            error, check_estimate(value, fit["reference"]["endpoints"][name], seed)
        )
        count += 1
    expected_cases = {
        f"{scale}/{steps}"
        for scale in ("fast", "primary", "slow")
        for steps in (4096, 8192)
    }
    expected_cases.update(("teacher_off", "mismatch_clamp", "teaching_shuffle"))
    if set(fit["cases"]) != expected_cases:
        raise RuntimeError("mandatory circuit matrix incomplete")
    for case_name, case in fit["cases"].items():
        raw = read_npz(case["arrays"])
        scale = {"fast": 0.5, "slow": 2.0}.get(case_name.split("/")[0], 1.0)
        verify_physical_and_query(
            raw, case, inputs, scale, fit["parameters"]["gamma_G"]
        )
        endpoints = manual_endpoints(raw, inputs)
        for name, values in endpoints.items():
            np.testing.assert_allclose(
                values, raw[f"endpoints__{name}"], atol=1e-12, rtol=0, equal_nan=True
            )
            error = max(
                error,
                check_estimate(values, case["endpoints"][name], seed),
                check_estimate(
                    values - original[name], case["paired_differences"][name], seed
                ),
            )
            count += 2
        for group in case["trajectory_errors"]:
            y = raw[f"{group}__trajectory"]
            width = (y.shape[-1] - 6) // 2
            observed = maximum_difference(
                y[..., :width] - y[..., width : 2 * width],
                reference[f"{group}__trajectory"],
            )
            if abs(observed - case["trajectory_errors"][group]) > 1e-12:
                raise RuntimeError("trajectory correspondence differs")
            margin_error = maximum_difference(
                raw[f"{group}__margin"], reference[f"{group}__margin"]
            )
            if abs(margin_error - case["margin_errors"][group]) > 1e-12:
                raise RuntimeError("margin correspondence differs")
    rebuilt = decide_fit(fit, original_spec()["decision_contract"]["competence"])
    if rebuilt != fit["decision"]:
        raise RuntimeError("registered decision differs")
    verify_checks(fit)
    verify_behavior(seed, fit)
    return {"estimates": count, "maximum_estimate_error": error, "passed": True}


def verify_run(directory) -> dict:
    audit = validate_run_manifest(directory / "run.json")
    if (
        not audit["passed"]
        or load_json(directory / "run.json")["lifecycle_state"] != "complete"
    ):
        raise RuntimeError(f"execution manifest invalid or incomplete: {audit}")
    result = load_json(directory / "result.json")
    return verify_result(result)


def verify_result(result: dict) -> dict:
    if set(result["fits"]) != {"2111", "2112", "2113"}:
        raise RuntimeError("mandatory fit inventory differs")
    fits = {seed: verify_fit(int(seed), fit) for seed, fit in result["fits"].items()}
    outcomes = {fit["decision"]["outcome"] for fit in result["fits"].values()}
    expected = "conditional_circuit_sufficiency"
    if outcomes != {expected}:
        expected = (
            "noninterpretable_execution"
            if "noninterpretable_execution" in outcomes
            else "qualified_circuit_mismatch"
        )
    if result["outcome"] != expected:
        raise RuntimeError("cross-fit registered outcome differs")
    return {
        "experiment_id": "score_circuit_v1",
        "passed": True,
        "fits": fits,
        "estimates": sum(row["estimates"] for row in fits.values()),
        "maximum_estimate_error": max(
            row["maximum_estimate_error"] for row in fits.values()
        ),
    }
