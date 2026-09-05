"""Complete locked circuit matrix with write-once per-case artifacts."""

import numpy as np

from fsrl.experiments.training_strategy.behavior import evaluate_behavior
from fsrl.experiments.training_strategy.evaluation import (
    flatten_arrays,
    json_ready,
    write_arrays,
)
from fsrl.experiments.training_strategy.locks import reference
from fsrl.infra.provenance import load_json, write_json_exclusive
from fsrl.infra.run_manifest import ProspectiveRun
from fsrl.paths import REPO_ROOT, STUDIES_ROOT
from fsrl.tasks.protocol_catalog import load_registered_protocol

from .analysis import (
    circuit_case,
    compare_case,
    reference_and_query_checks,
    reference_case,
    refinement,
    summarize,
)
from .decisions import decide_fit
from .evidence import (
    PROTOCOL_HASH,
    load_arrays,
    parent_result,
    require_clean_pushed,
    specification,
    validate_lock,
)
from .qualification import compiled_runner, runtime


def original_spec() -> dict:
    return load_json(
        STUDIES_ROOT
        / "minimal_relational_learner/records/benchmarks/minimal_relational_learner_v1.json"
    )


def save_case(directory, name, raw) -> dict:
    path = directory / f"{name}.npz"
    write_arrays(path, flatten_arrays(raw))
    return reference(path)


def run_scales(seed, arrays, parameters, reference_raw, runner, directory) -> tuple:
    cases, refinements, primary = {}, {}, None
    for name, scale in specification()["circuit"]["scales"].items():
        coarse = None
        for steps in specification()["numerics"]["steps_per_support"]:
            raw = circuit_case(arrays, parameters, scale, steps, runner)
            record = compare_case(raw, reference_raw, seed)
            record["arrays"] = save_case(directory, f"seed-{seed}-{name}-{steps}", raw)
            cases[f"{name}/{steps}"] = record
            if steps == 4096:
                coarse = raw
                if name == "primary":
                    primary = raw
            else:
                assert coarse is not None
                refinements[name] = refinement(coarse, raw)
            print(f"Completed {seed}/{name}/{steps}", flush=True)
    return cases, refinements, primary


def run_controls(seed, arrays, parameters, reference_raw, runner, directory) -> tuple:
    cases, unchanged = {}, {}
    for name in ("teacher_off", "mismatch_clamp", "teaching_shuffle"):
        raw = circuit_case(arrays, parameters, 1, 4096, runner, control=name)
        record = compare_case(raw, reference_raw, seed)
        record["arrays"] = save_case(directory, f"seed-{seed}-{name}", raw)
        cases[name] = record
        if name != "teaching_shuffle":
            unchanged[name] = all(
                np.all(group["trajectory"][..., :30] == 1)
                for batch, group in raw.items()
                if batch != "endpoints"
            )
    return cases, unchanged


def run_fit(seed, parameters, runner, directory) -> dict:
    arrays = load_arrays(seed)
    old = original_spec()
    reference_raw, bridge = reference_case(arrays, parameters)
    ref_file = save_case(directory, f"seed-{seed}-reference", reference_raw)
    cases, refinements, primary = run_scales(
        seed, arrays, parameters, reference_raw, runner, directory
    )
    controls, unchanged = run_controls(
        seed, arrays, parameters, reference_raw, runner, directory
    )
    cases.update(controls)
    checks, check_arrays = reference_and_query_checks(primary, arrays, parameters)
    behavior = evaluate_behavior(
        {"logits": primary["liu"]["margin"]},
        load_registered_protocol("liu_v2"),
        seed,
        old,
    )
    behavior_path = directory / f"seed-{seed}-sampled-behavior.json"
    write_json_exclusive(behavior_path, json_ready(behavior["sampled_behavior"]))
    fit = {
        "parameters": parameters,
        "parent_bridge": bridge,
        "reference": {
            "arrays": ref_file,
            "endpoints": summarize(reference_raw["endpoints"], seed),
        },
        "cases": cases,
        "refinement": refinements,
        "reference_checks": checks,
        "check_arrays": save_case(
            directory, f"seed-{seed}-reference-checks", check_arrays
        ),
        "behavior": behavior["record"],
        "sampled_behavior": reference(behavior_path),
        "parent_behavior": parent_result()["conditions"][f"{seed}/score_only"][
            "behavior"
        ],
        "control_no_write": unchanged,
    }
    fit["decision"] = decide_fit(fit, old["decision_contract"]["competence"])
    return fit


def execute() -> dict:
    lock = validate_lock(pushed=True)
    commit = require_clean_pushed()
    spec = specification()
    snapshot = runtime()
    runner = compiled_runner()
    directory = REPO_ROOT / spec["numerics"]["run_directory"]
    with ProspectiveRun.start(
        directory,
        workflow_id=spec["experiment_id"],
        execution_id="evaluation-v1",
        producer={
            "module": __name__,
            "source_commit": lock["source_commit"],
            "execution_commit": commit,
        },
        resolved_config={
            "protocol_sha256": PROTOCOL_HASH,
            "runtime": snapshot,
            "specification": spec,
        },
    ) as run:
        fits = {}
        for seed in spec["seeds"]:
            fit = run_fit(seed, lock["parameters"][str(seed)], runner, run.output_dir)
            fits[str(seed)] = fit
            write_json_exclusive(
                run.output_dir / f"seed-{seed}-result.json", json_ready(fit)
            )
            print(f"Stream {seed}: {fit['decision']['outcome']}", flush=True)
        outcomes = [fit["decision"]["outcome"] for fit in fits.values()]
        outcome = "conditional_circuit_sufficiency"
        if any(value != outcome for value in outcomes):
            outcome = (
                "noninterpretable_execution"
                if "noninterpretable_execution" in outcomes
                else "qualified_circuit_mismatch"
            )
        result = {
            "experiment_id": spec["experiment_id"],
            "protocol_sha256": PROTOCOL_HASH,
            "protocol_commit": lock["protocol_commit"],
            "source_commit": lock["source_commit"],
            "execution_commit": commit,
            "runtime": snapshot,
            "fits": fits,
            "outcome": outcome,
            "boundaries": spec["boundaries"],
            "stop_rule": spec["stop_rule"],
        }
        validate_lock(pushed=True)
        write_json_exclusive(run.output_dir / "result.json", json_ready(result))
    return {"directory": str(directory), "outcome": outcome}
