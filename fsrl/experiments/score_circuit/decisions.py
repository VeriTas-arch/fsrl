"""Only the prospectively frozen circuit sufficiency conjunction."""


def physical_constraints(case: dict) -> bool:
    return all(
        row["minimum_efficacy"] >= 0
        and row["maximum_efficacy"] <= 2
        and row["minimum_activity_rate"] >= 0
        and row["minimum_input_rate"] >= 0
        and row["bound_engagements"] == 0
        and row["maximum_pair_sum_error"] <= 1e-9
        for row in case["physical"].values()
    )


def behavior_preservation(case: dict, competence: dict) -> bool:
    for name, row in case["paired_differences"].items():
        if "/probability/" in name:
            if row["bootstrap"]["lower"] < -0.01 or row["bootstrap"]["upper"] > 0.01:
                return False
        elif abs(row["mean"]) > 0.01:
            return False
    for group in ("learned", "nonlearned"):
        if (
            case["endpoints"][f"generic/exact_decision/{group}"]["mean"]
            < competence[f"generic_{group}"]
        ):
            return False
    return True


def decide_fit(fit: dict, competence: dict) -> dict:
    cases = [fit["cases"][f"{scale}/4096"] for scale in ("fast", "primary", "slow")]
    all_cases = [
        value
        for name, value in fit["cases"].items()
        if name.split("/")[0] in {"fast", "primary", "slow"}
    ]
    refinements = [
        value
        for group in fit["refinement"].values()
        for values in group.values()
        for value in values.values()
    ]
    checks = {
        "integrity": max(fit["parent_bridge"].values()) <= 1e-5
        and fit["reference_checks"]["affine_max_error"] <= 1e-5
        and max(refinements) <= 1e-5,
        "physical_constraints": all(physical_constraints(case) for case in all_cases),
        "correspondence": all(
            max(case["trajectory_errors"].values()) <= 0.01
            and max(case["margin_errors"].values()) <= 0.1
            for case in all_cases
        ),
        "behavior_preservation": all(
            behavior_preservation(case, competence) for case in cases
        )
        and len(fit["behavior"]["flags"]) == 9
        and set(fit["behavior"]["flags"]) == set(fit["parent_behavior"]["flags"])
        and all(row["qualitative"] for row in fit["behavior"]["flags"].values()),
        "robustness": max(refinements) <= 1e-5
        and fit["reference_checks"]["query_no_write"]
        and max(fit["reference_checks"]["query_errors"].values()) <= 0.005,
        "no_write_controls": all(
            fit["control_no_write"][name] for name in ("teacher_off", "mismatch_clamp")
        ),
    }
    outcome = (
        "conditional_circuit_sufficiency"
        if all(checks.values())
        else "qualified_circuit_mismatch"
    )
    if not checks["integrity"]:
        outcome = "noninterpretable_execution"
    return {"checks": checks, "outcome": outcome}
