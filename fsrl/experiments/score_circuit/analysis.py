"""Fixed circuit cases, parent bridges and paired subject-level endpoints."""

import time

import numpy as np
import torch

from fsrl.experiments.training_strategy.estimands import estimate, query_endpoints
from fsrl.experiments.training_strategy.summaries import liu_endpoints
from fsrl.tasks.protocol_catalog import load_registered_protocol

from .circuit import integrate_support, query_read
from .evidence import batches
from .reference import affine_support, discrete

STATISTICS = {"samples": 10000, "interval": 0.95}


def maximum_difference(a, b) -> float:
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def collect_margins(raw: dict, arrays: dict) -> dict:
    margins = {
        "generic": np.empty_like(arrays["generic__margins__intact"]),
        "liu": raw["liu"]["margin"],
    }
    for name, group in raw.items():
        if name.startswith("generic_"):
            prefix = f"generic__groups__{name.split('_')[1]}__"
            margins["generic"][arrays[prefix + "episode_indices"]] = group["margin"]
    return margins


def endpoint_vectors(margins: dict, arrays: dict) -> dict:
    protocol = load_registered_protocol("liu_v2")
    generic = query_endpoints(
        margins["generic"][..., None],
        arrays["generic__signs"][..., None],
        {
            "learned": arrays["generic__learned"],
            "nonlearned": ~arrays["generic__learned"],
        },
        temperature=1.0,
    )
    liu = liu_endpoints(
        {"intact": {"logits": margins["liu"]}}, arrays["liu__retention"], protocol, 0.25
    )["intact"]
    return {
        f"{domain}/{measure}/{group}": value
        for domain, endpoints in (("generic", generic), ("liu", liu))
        for measure, groups in endpoints.items()
        for group, value in groups.items()
    }


def summarize(values: dict, seed: int) -> dict:
    return {
        name: estimate(value, seed=98000 + seed, statistics=STATISTICS)
        for name, value in values.items()
    }


def reference_case(arrays: dict, parameters: dict) -> tuple[dict, dict]:
    raw = {
        name: discrete(inputs, parameters["eta"], parameters["gamma_G"])
        for name, inputs in batches(arrays).items()
    }
    margins = collect_margins(raw, arrays)
    errors = {
        "generic_margin": maximum_difference(
            margins["generic"], arrays["generic__margins__intact"]
        ),
        "liu_margin": maximum_difference(
            margins["liu"], arrays["liu__bundles__intact__logits"]
        ),
        "liu_state": maximum_difference(
            raw["liu"]["trajectory"][:, -1], arrays["liu__w"]
        ),
    }
    if max(errors.values()) > 1e-5:
        raise RuntimeError(f"parent score reconstruction failed: {errors}")
    raw["endpoints"] = endpoint_vectors(margins, arrays)
    return raw, errors


def circuit_case(arrays, parameters, scale, steps, runner, *, control="intact") -> dict:
    raw = {}
    for name, inputs in batches(arrays).items():
        start = time.perf_counter()
        output = integrate_support(
            inputs, parameters["eta"], scale, steps, runner, control=control
        )
        state = output["trajectory"][:, -1]
        output["margin"] = query_read(
            state, inputs["query_cues"], parameters["gamma_G"], 0.002 * scale
        )
        torch.cuda.synchronize()
        output["seconds"] = time.perf_counter() - start
        raw[name] = output
        print(
            f"  {name}: scale={scale} steps={steps} control={control} {output['seconds']:.2f}s",
            flush=True,
        )
    raw["endpoints"] = endpoint_vectors(collect_margins(raw, arrays), arrays)
    return raw


def compare_case(raw: dict, reference: dict, seed: int) -> dict:
    trajectory_errors, margin_errors, physical = {}, {}, {}
    for name, group in raw.items():
        if name == "endpoints":
            continue
        state = group["trajectory"]
        width = (state.shape[-1] - 6) // 2
        effective = state[..., :width] - state[..., width : 2 * width]
        trajectory_errors[name] = maximum_difference(
            effective, reference[name]["trajectory"]
        )
        margin_errors[name] = maximum_difference(
            group["margin"], reference[name]["margin"]
        )
        diag = group["diagnostics"]
        physical[name] = {
            "minimum_efficacy": float(diag[:, 0].min()),
            "maximum_efficacy": float(-diag[:, 1].min()),
            "minimum_activity_rate": float(diag[:, 2].min()),
            "maximum_pair_sum_error": float(-diag[:, 3].min()),
            "bound_engagements": int(diag[:, 4].sum()),
            "minimum_input_rate": group["minimum_input_rate"],
        }
    differences = {
        name: value - reference["endpoints"][name]
        for name, value in raw["endpoints"].items()
    }
    return {
        "trajectory_errors": trajectory_errors,
        "margin_errors": margin_errors,
        "physical": physical,
        "endpoints": summarize(raw["endpoints"], seed),
        "paired_differences": summarize(differences, seed),
        "seconds": {
            name: group["seconds"] for name, group in raw.items() if name != "endpoints"
        },
    }


def reference_and_query_checks(
    raw: dict, arrays: dict, parameters: dict
) -> tuple[dict, dict]:
    inputs = batches(arrays)["liu"]
    affine = affine_support(inputs, parameters["eta"], 1.0)
    state = raw["liu"]["trajectory"][:, -1]
    before = state.copy()
    margins, errors = {}, {}
    for duration in (0.05, 0.1, 0.2):
        for reverse in (False, True):
            name = f"{duration}/{reverse}"
            margins[name] = query_read(
                state,
                inputs["query_cues"],
                parameters["gamma_G"],
                0.002,
                duration,
                reverse=reverse,
            )
            errors[name] = maximum_difference(margins[name], raw["liu"]["margin"])
    np.testing.assert_array_equal(state, before)
    return (
        {
            "affine_max_error": maximum_difference(affine, raw["liu"]["trajectory"]),
            "query_no_write": True,
            "query_errors": errors,
        },
        {"affine_trajectory": affine, "query_margins": margins},
    )


def refinement(coarse: dict, fine: dict) -> dict:
    errors = {}
    for name, group in coarse.items():
        if name == "endpoints":
            continue
        y, other = group["trajectory"], fine[name]["trajectory"]
        width = (y.shape[-1] - 6) // 2
        errors[name] = {
            "state": maximum_difference(
                y[..., :width] - y[..., width : 2 * width],
                other[..., :width] - other[..., width : 2 * width],
            ),
            "margin": maximum_difference(group["margin"], fine[name]["margin"]),
        }
    return errors
