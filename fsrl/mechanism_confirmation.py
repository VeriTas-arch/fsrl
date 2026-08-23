"""Prospective formal confirmation of the frozen FSRL mechanism chain."""

from __future__ import annotations

import argparse
import copy
import json
import tempfile
from pathlib import Path

import numpy as np

from .assembly_diagnostics import file_sha256, load_json, resolve_path
from .assembly_trajectory import (
    bootstrap_counts,
    run_assembly_trajectory,
    summarize_subjects,
)
from .confirmation import (
    DEFAULT_OUTPUT_ROOT,
    _validate_checkpoint,
    write_json,
)
from .history_state_factorial import run_history_state_factorial
from .support_factor_swap import run_support_factor_swap

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPECIFICATION_PATH = ROOT / "benchmarks" / "mechanism_confirmation_v1.json"
DEFAULT_RESULT_PATH = ROOT / "results" / "mechanism_confirmation_v1.json"
DEVELOPMENT_TRAINING_PATH = ROOT / "benchmarks" / "pilot_v1.json"
COMPONENT_SOURCES = {
    "assembly": ROOT / "benchmarks" / "assembly_trajectory_v1.json",
    "factor_swap": ROOT / "benchmarks" / "support_factor_swap_v1.json",
    "history_state": ROOT / "benchmarks" / "history_state_factorial_v1.json",
}
COMPONENT_RESULTS = {
    "assembly": ROOT / "results" / "assembly_trajectory_v1.json",
    "factor_swap": ROOT / "results" / "support_factor_swap_v1.json",
    "history_state": ROOT / "results" / "history_state_factorial_v1.json",
}
COMPONENT_RUNNERS = {
    "assembly": run_assembly_trajectory,
    "factor_swap": run_support_factor_swap,
    "history_state": run_history_state_factorial,
}


def _registered_file(path: Path) -> dict:
    resolved = path.resolve()
    try:
        registered_path = str(resolved.relative_to(ROOT))
    except ValueError:
        registered_path = str(resolved)
    return {"path": registered_path, "sha256": file_sha256(resolved)}


def _validate_file(registration: dict) -> dict:
    path = resolve_path(registration["path"])
    observed = file_sha256(path)
    return {
        "path": registration["path"],
        "expected_sha256": registration["sha256"],
        "observed_sha256": observed,
        "passed": observed == registration["sha256"],
    }


def validate_mechanism_contract(
    specification_path: Path = DEFAULT_SPECIFICATION_PATH,
) -> dict:
    specification = load_json(specification_path)
    source_checks = {
        name: _validate_file(registration)
        for name, registration in specification["registered_sources"].items()
    }
    formal_registration = specification["registered_sources"][
        "formal_training_contract"
    ]
    formal = load_json(resolve_path(formal_registration["path"]))
    population_checks = {
        "seeds": specification["formal_artifact_contract"]["seeds"]
        == formal["training"]["seeds"],
        "checkpoint_selection": specification["formal_artifact_contract"][
            "checkpoint_selection"
        ]
        == formal["training"]["checkpoint_selection"],
        "subject_encoding_mode": specification["formal_artifact_contract"][
            "subject_encoding_mode"
        ]
        == formal["training"]["subject_encoding_mode"],
        "evaluation_subjects": specification["formal_artifact_contract"][
            "evaluation_subjects"
        ]
        == formal["evaluation"]["batch_size"],
    }
    return {
        "confirmation_id": specification["confirmation_id"],
        "passed": all(row["passed"] for row in source_checks.values())
        and all(population_checks.values()),
        "source_checks": source_checks,
        "population_checks": population_checks,
    }


def _seed_input_registration(
    seed: int,
    *,
    training_specification_path: Path,
    output_root: Path,
) -> tuple[dict, dict]:
    training_specification = load_json(training_specification_path)
    if seed not in training_specification["training"]["seeds"]:
        raise ValueError(f"seed {seed} is not registered by the training contract")
    seed_dir = output_root / f"seed-{seed}"
    checkpoint = seed_dir / "net.dat"
    config = seed_dir / "config.json"
    behavior = seed_dir / "behavior-stable_omission.json"
    confirmation = seed_dir / "confirmation.json"
    qualification = seed_dir / "qualification.json"
    for path in (checkpoint, config, behavior, confirmation, qualification):
        if not path.is_file():
            raise RuntimeError(f"registered seed artifact is missing: {path}")
    metadata = _validate_checkpoint(checkpoint, training_specification, seed)
    behavior_result = load_json(behavior)
    confirmation_result = load_json(confirmation)
    qualification_result = load_json(qualification)
    if behavior_result["checkpoint"]["sha256"] != metadata["checkpoint"]["sha256"]:
        raise RuntimeError("behavior artifact does not match the registered checkpoint")
    if confirmation_result["seed"] != seed:
        raise RuntimeError("confirmation artifact has the wrong seed")
    if confirmation_result["checkpoint"]["sha256"] != metadata["checkpoint"]["sha256"]:
        raise RuntimeError("confirmation artifact does not match the checkpoint")
    if (
        confirmation_result["passes"]["causal_qualification"]
        != qualification_result["passed"]
    ):
        raise RuntimeError("confirmation and qualification artifacts disagree")
    adapter_registration = {
        "seed": seed,
        "checkpoint_path": _registered_file(checkpoint)["path"],
        "checkpoint_sha256": file_sha256(checkpoint),
        "config_path": _registered_file(config)["path"],
        "config_sha256": file_sha256(config),
        "behavior_path": _registered_file(behavior)["path"],
        "behavior_sha256": file_sha256(behavior),
    }
    all_inputs = {
        "checkpoint": _registered_file(checkpoint),
        "config": _registered_file(config),
        "behavior": _registered_file(behavior),
        "confirmation": _registered_file(confirmation),
        "qualification": _registered_file(qualification),
    }
    return adapter_registration, {
        "files": all_inputs,
        "confirmation": confirmation_result,
        "qualification": qualification_result,
    }


def _adapter_specification(
    component: str,
    *,
    training_specification_path: Path,
    artifact_registrations: list[dict],
) -> dict:
    specification = copy.deepcopy(load_json(COMPONENT_SOURCES[component]))
    specification["registration_status"] = (
        "formal_artifact_adapter_from_source_locked_development_specification"
    )
    specification["registered_sources"]["pilot_specification"] = _registered_file(
        training_specification_path
    )
    specification["registered_sources"]["pilot_artifacts"] = artifact_registrations
    execution = specification["execution_contract"]
    if "seeds" in execution:
        execution["seeds"] = [int(row["seed"]) for row in artifact_registrations]
    if "formal_seed_access" in execution:
        execution["formal_seed_access"] = (
            "authorized only by frozen mechanism_confirmation_v1"
        )
    return specification


def _write_adapters(
    directory: Path,
    *,
    training_specification_path: Path,
    artifact_registrations: list[dict],
) -> dict[str, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = {}
    for component in COMPONENT_SOURCES:
        path = directory / f"{component}.json"
        write_json(
            path,
            _adapter_specification(
                component,
                training_specification_path=training_specification_path,
                artifact_registrations=artifact_registrations,
            ),
        )
        paths[component] = path
    return paths


def _run_components(adapter_paths: dict[str, Path]) -> dict[str, dict]:
    return {
        component: COMPONENT_RUNNERS[component](path)
        for component, path in adapter_paths.items()
    }


def reproduce_development_components() -> dict:
    artifact_registrations = load_json(COMPONENT_SOURCES["assembly"])[
        "registered_sources"
    ]["pilot_artifacts"]
    with tempfile.TemporaryDirectory(prefix="fsrl-mechanism-dryrun-") as temp_dir:
        adapter_paths = _write_adapters(
            Path(temp_dir),
            training_specification_path=DEVELOPMENT_TRAINING_PATH,
            artifact_registrations=artifact_registrations,
        )
        observed = _run_components(adapter_paths)
    checks = {}
    for component, result in observed.items():
        expected = load_json(COMPONENT_RESULTS[component])
        checks[component] = {
            "passed": result["pilot_seeds"] == expected["pilot_seeds"],
            "seeds": sorted(result["pilot_seeds"]),
            "expected_result": _registered_file(COMPONENT_RESULTS[component]),
        }
    return {"passed": all(row["passed"] for row in checks.values()), "checks": checks}


def _mean(summary: dict) -> float:
    value = summary["mean"]
    if value is None or not np.isfinite(value):
        raise RuntimeError("registered seed-level estimand is nonfinite")
    return float(value)


def _seed_estimands(components: dict[str, dict]) -> tuple[dict, dict]:
    assembly = components["assembly"]
    factor = components["factor_swap"]
    history = components["history_state"]
    primary = {
        "immediate_remote_absolute": _mean(
            assembly["matched_zero_evidence_branches"]["aggregate"]["retained"][
                "mean_absolute"
            ]["remote"]
        ),
        "loo_remote_absolute": _mean(
            assembly["leave_one_relation_out"]["aggregate"]["retained"][
                "mean_absolute"
            ]["remote"]
        ),
        "loo_third_party_relational_fraction": _mean(
            assembly["leave_one_relation_out"]["aggregate"]["retained"][
                "gauge_invariant_R_third_rel"
            ]
        ),
        "eligibility_donor_identity_advantage": _mean(
            factor["eligibility_identity_transfer"]["donor_identity_advantage"]
        ),
        "da_write_norm_difference": _mean(
            factor["da_magnitude_transfer"]["write_norm_difference"]
        ),
        "da_policy_norm_difference": _mean(
            factor["da_magnitude_transfer"]["policy_norm_difference"]
        ),
        "da_direction_cosine": _mean(
            factor["da_magnitude_transfer"]["direction_cosine"]
        ),
        "alpha_actual_minus_null_mean_gain": _mean(
            factor["alpha_systematic_gain"]["actual_minus_null_mean_gain"]
        ),
        "history_baseline_expression": _mean(
            history["potential_norm"]["baseline_expression_effect"]
        ),
        "history_interaction": _mean(history["potential_norm"]["interaction"]),
        "terminal_distributional_minus_map_cosine": _mean(
            assembly["prefix_trajectory"]["final_distributional_minus_map_alignment"][
                "cosine"
            ]
        ),
    }
    diagnostics = {
        "history_factor_generation": _mean(
            history["potential_norm"]["factor_generation_effect"]
        ),
        "history_alignment_factor_generation": _mean(
            history["first_exposure_relation_alignment"]["factor_generation_effect"]
        ),
        "immediate_remote_correctness_aligned": _mean(
            assembly["matched_zero_evidence_branches"]["aggregate"]["retained"][
                "mean_correctness_aligned"
            ]["remote"]
        ),
        "loo_remote_correctness_aligned": _mean(
            assembly["leave_one_relation_out"]["aggregate"]["retained"][
                "mean_correctness_aligned"
            ]["remote"]
        ),
    }
    return primary, diagnostics


def _validation_error_max(validation: dict) -> float:
    values = [
        float(value)
        for name, value in validation.items()
        if name.endswith(("max_abs_error", "max_abs"))
    ]
    return max(values, default=0.0)


def _reproduction_gate(components: dict[str, dict], tolerance: float) -> dict:
    assembly = components["assembly"]
    factor = components["factor_swap"]
    history = components["history_state"]
    checks = {
        "assembly_incremental_endpoint": assembly["matched_zero_evidence_branches"][
            "incremental_endpoint_max_abs_error"
        ],
        "assembly_stable_omitted_update": assembly["matched_zero_evidence_branches"][
            "stable_omitted_max_abs_effect"
        ],
        "assembly_stable_omitted_loo": assembly["leave_one_relation_out"][
            "stable_omitted_max_abs_pair_influence"
        ],
        "factor_validation": _validation_error_max(factor["validation"]),
        "history_validation": _validation_error_max(history["validation"]),
    }
    return {
        "tolerance": tolerance,
        "checks": {name: float(value) for name, value in checks.items()},
        "passed": all(float(value) <= tolerance for value in checks.values()),
    }


def run_mechanism_seed(
    seed: int,
    *,
    specification_path: Path = DEFAULT_SPECIFICATION_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict:
    contract_validation = validate_mechanism_contract(specification_path)
    if not contract_validation["passed"]:
        raise RuntimeError("mechanism confirmation contract validation failed")
    specification = load_json(specification_path)
    if seed not in specification["formal_artifact_contract"]["seeds"]:
        raise ValueError(f"seed {seed} is not registered for mechanism confirmation")
    training_path = resolve_path(
        specification["registered_sources"]["formal_training_contract"]["path"]
    )
    seed_dir = output_root / f"seed-{seed}"
    artifact_registration, inputs = _seed_input_registration(
        seed,
        training_specification_path=training_path,
        output_root=output_root,
    )
    adapter_paths = _write_adapters(
        seed_dir / "mechanism-adapters",
        training_specification_path=training_path,
        artifact_registrations=[artifact_registration],
    )
    component_wrappers = _run_components(adapter_paths)
    components = {
        name: wrapper["pilot_seeds"][str(seed)]
        for name, wrapper in component_wrappers.items()
    }
    component_paths = {}
    for name, wrapper in component_wrappers.items():
        path = seed_dir / f"mechanism-{name}.json"
        write_json(path, wrapper)
        component_paths[name] = path
    primary, diagnostics = _seed_estimands(components)
    tolerance = float(
        specification["execution_contract"]["floating_reproduction_tolerance"]
    )
    reproduction = _reproduction_gate(components, tolerance)
    result = {
        "schema_version": 1,
        "confirmation_id": specification["confirmation_id"],
        "seed": seed,
        "checkpoint": inputs["confirmation"]["checkpoint"],
        "formal_seed_interpretation": "deferred_until_all_ten_seeds_complete",
        "competence": {
            "fast_weight_content_necessity": bool(inputs["qualification"]["passed"]),
            "source_reproduction": reproduction,
        },
        "primary_seed_means": primary,
        "registered_nonprimary_seed_means": diagnostics,
        "orchestration_source": _registered_file(Path(__file__)),
        "input_artifacts": inputs["files"],
        "adapter_artifacts": {
            name: _registered_file(path) for name, path in adapter_paths.items()
        },
        "component_artifacts": {
            name: _registered_file(path) for name, path in component_paths.items()
        },
    }
    write_json(seed_dir / "mechanism-confirmation.json", result)
    return result


def _validate_seed_result(seed: int, path: Path, confirmation_id: str) -> dict:
    result = load_json(path)
    if result.get("seed") != seed or result.get("confirmation_id") != confirmation_id:
        raise RuntimeError(f"mechanism seed result identity mismatch: {path}")
    registrations = {
        "orchestration_source": result["orchestration_source"],
        **result["input_artifacts"],
        **{f"adapter_{name}": row for name, row in result["adapter_artifacts"].items()},
        **{
            f"component_{name}": row
            for name, row in result["component_artifacts"].items()
        },
    }
    failures = [
        name for name, row in registrations.items() if not _validate_file(row)["passed"]
    ]
    if failures:
        raise RuntimeError(f"mechanism seed artifact hash mismatch: {failures}")
    return result


def _threshold_status(summary: dict, threshold: float = 0.0) -> str:
    lower = summary["bootstrap"]["lower"]
    upper = summary["bootstrap"]["upper"]
    if lower > threshold:
        return "confirmed"
    if upper < threshold:
        return "directionally_contrary"
    return "unresolved"


def aggregate_mechanism_confirmation(
    *,
    specification_path: Path = DEFAULT_SPECIFICATION_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict:
    validation = validate_mechanism_contract(specification_path)
    if not validation["passed"]:
        raise RuntimeError("mechanism confirmation contract validation failed")
    specification = load_json(specification_path)
    seeds = specification["formal_artifact_contract"]["seeds"]
    seed_results = []
    missing = []
    for seed in seeds:
        path = output_root / f"seed-{seed}" / "mechanism-confirmation.json"
        if not path.is_file():
            missing.append(seed)
        else:
            seed_results.append(
                _validate_seed_result(seed, path, specification["confirmation_id"])
            )
    if missing:
        raise RuntimeError(
            f"all registered formal seeds are mandatory; missing results for {missing}"
        )
    inference = specification["formal_inference"]
    rng = np.random.default_rng(int(inference["network_bootstrap_seed"]))
    counts = bootstrap_counts(
        rng, int(inference["network_bootstrap_samples"]), len(seeds)
    )
    estimand_names = tuple(seed_results[0]["primary_seed_means"])
    summaries = {
        name: summarize_subjects(
            np.asarray([row["primary_seed_means"][name] for row in seed_results]),
            counts,
            interval=float(inference["interval"]),
        )
        for name in estimand_names
    }
    diagnostic_names = tuple(seed_results[0]["registered_nonprimary_seed_means"])
    diagnostic_summaries = {
        name: summarize_subjects(
            np.asarray(
                [row["registered_nonprimary_seed_means"][name] for row in seed_results]
            ),
            counts,
            interval=float(inference["interval"]),
        )
        for name in diagnostic_names
    }
    competence = all(
        row["competence"]["fast_weight_content_necessity"] for row in seed_results
    )
    reproduction = all(
        row["competence"]["source_reproduction"]["passed"] for row in seed_results
    )
    link_estimands = {
        "immediate_and_episode_global_reassembly": (
            ("immediate_remote_absolute", 0.0),
            ("loo_remote_absolute", 0.0),
            ("loo_third_party_relational_fraction", 0.0),
        ),
        "eligibility_relation_direction": (
            ("eligibility_donor_identity_advantage", 0.0),
        ),
        "da_gain_with_direction_preservation": (
            ("da_write_norm_difference", 0.0),
            ("da_policy_norm_difference", 0.0),
            ("da_direction_cosine", 0.99),
        ),
        "alpha_high_gain_placement": (("alpha_actual_minus_null_mean_gain", 0.0),),
        "history_dependent_expression": (("history_baseline_expression", 0.0),),
        "history_matched_nonlinearity": (("history_interaction", 0.0),),
        "terminal_distributional_projection": (
            ("terminal_distributional_minus_map_cosine", 0.0),
        ),
    }
    links = {}
    for link, criteria in link_estimands.items():
        criteria_status = {
            name: _threshold_status(summaries[name], threshold)
            for name, threshold in criteria
        }
        if not competence or not reproduction:
            status = "non_interpretable"
        elif all(value == "confirmed" for value in criteria_status.values()):
            status = "confirmed"
        elif any(
            value == "directionally_contrary" for value in criteria_status.values()
        ):
            status = "directionally_contrary"
        else:
            status = "unresolved"
        links[link] = {"status": status, "criteria": criteria_status}
    complete = (
        competence
        and reproduction
        and all(row["status"] == "confirmed" for row in links.values())
    )
    return {
        "schema_version": 1,
        "confirmation_id": specification["confirmation_id"],
        "status": "complete_all_declared_formal_seeds_reported",
        "declared_seeds": seeds,
        "reported_seeds": [row["seed"] for row in seed_results],
        "seed_filtering": False,
        "contract_validation": validation,
        "competence_and_integrity": {
            "all_seeds_fast_weight_content_necessary": competence,
            "all_seed_source_reproduction_passed": reproduction,
        },
        "primary_network_seed_summaries": summaries,
        "registered_nonprimary_network_seed_summaries": diagnostic_summaries,
        "linkwise_confirmation": links,
        "complete_chain_confirmed": complete,
        "seed_results": seed_results,
    }


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Run the frozen formal mechanism confirmation workflow."
    )
    parser.add_argument(
        "action",
        choices=["validate", "validate-development", "run-seed", "aggregate"],
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--specification", type=Path, default=DEFAULT_SPECIFICATION_PATH
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(args)


def main(args=None):
    parsed = parse_args(args)
    if parsed.action == "validate":
        result = validate_mechanism_contract(parsed.specification)
    elif parsed.action == "validate-development":
        result = reproduce_development_components()
    elif parsed.action == "run-seed":
        if parsed.seed is None:
            raise ValueError("--seed is required for run-seed")
        result = run_mechanism_seed(
            parsed.seed,
            specification_path=parsed.specification,
            output_root=parsed.output_root,
        )
    else:
        result = aggregate_mechanism_confirmation(
            specification_path=parsed.specification,
            output_root=parsed.output_root,
        )
    output = parsed.output
    if parsed.action == "aggregate" and output is None:
        output = DEFAULT_RESULT_PATH
    if output is None:
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    else:
        write_json(output, result)
    return 0 if result.get("passed", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
