"""Paired route decisions, conditional transport and immutable study artifacts."""

import shutil

import numpy as np

from fsrl.experiments.memory_structure.measurement import above
from fsrl.experiments.training_strategy.estimands import paired_estimate
from fsrl.experiments.training_strategy.evaluation import json_ready
from fsrl.experiments.training_strategy.locks import reference
from fsrl.infra.file_contracts import validate_run_manifest
from fsrl.infra.provenance import load_json, write_json_exclusive
from fsrl.paths import REPO_ROOT

from .locks import ARTIFACT_LOCK, SOURCE_LOCK, validate_artifacts
from .measurement import paired_tau, selection
from .protocol import (
    PROTOCOL_SHA256,
    RECORD_ROOT,
    RUN_ROOT,
    run_directory,
    specification,
)


def read_cell(seed: int, condition: str, *, transport=False) -> tuple:
    directory = (
        RUN_ROOT / ("transport" if transport else "evaluation") / str(seed) / condition
    )
    run = load_json(directory / "run.json")
    if (
        run["lifecycle_state"] != "complete"
        or not validate_run_manifest(directory / "run.json")["passed"]
    ):
        raise RuntimeError("evaluation incomplete or modified")
    result = load_json(directory / "result.json")
    if result["protocol_sha256"] != PROTOCOL_SHA256 or result[
        "artifact_lock"
    ] != reference(ARTIFACT_LOCK):
        raise RuntimeError("evaluation identity differs")
    with np.load(directory / "raw.npz", allow_pickle=False) as data:
        raw = {key: data[key] for key in data.files}
    return result, raw, directory


def negative(row: dict) -> bool:
    upper = row["interval"]["upper"]
    return upper is not None and upper < 0


def compare_pair(seed: int, spec: dict) -> dict:
    shared, a, a_dir = read_cell(seed, "shared")
    isolated, b, b_dir = read_cell(seed, "isolated")
    a_orders, a_mask = selection(load_json(a_dir / "behavior-N8.json"))
    b_orders, b_mask = selection(load_json(b_dir / "behavior-N8.json"))
    bootstrap_seed = spec["statistics"]["seed_offset"] + seed
    tau = paired_tau(
        a_orders, b_orders, a_mask, b_mask, bootstrap_seed, spec["statistics"]
    )
    full = np.ones(len(a_orders), dtype=bool)
    field_tau = paired_tau(
        a["N8__internal__orders"],
        b["N8__internal__orders"],
        full,
        full,
        bootstrap_seed,
        spec["statistics"],
    )
    estimates = {}
    for group in ("learned", "nonlearned", "omitted"):
        key = "N8__endpoints__intact__probability__" + group
        estimates[group] = paired_estimate(
            b[key], a[key], seed=bootstrap_seed, statistics=spec["statistics"]
        )
    for name in ("stable_error_prevalence", "stable_error_density"):
        key = "N8__internal__" + name
        estimates["internal_" + name] = paired_estimate(
            b[key], a[key], seed=bootstrap_seed, statistics=spec["statistics"]
        )
    cell = isolated["cells"]["8"]
    competent = all(
        row["generic"]["competence"] and row["cells"]["8"]["competence"]
        for row in (shared, isolated)
    )
    stable = above(cell["internal"]["stable_error_prevalence"], 0)
    functional = all(
        above(cell["effects"][f"intact_minus_{name}_nonlearned"], 0)
        for name in ("P_off", "evidence_shuffle")
    )
    supported = (
        competent
        and cell["core_passed"]
        and negative(tau)
        and negative(field_tau)
        and stable
        and functional
    )
    if not competent:
        outcome = "competence_limited"
    elif supported:
        outcome = "supported_recovery"
    elif negative(tau) and negative(field_tau):
        outcome = "contribution_without_recovery"
    elif negative(tau):
        outcome = "effect_with_unresolved_structure"
    else:
        outcome = "unresolved_or_opposite"
    return {
        "shared": shared,
        "isolated": isolated,
        "paired": {
            "conditional_tau": tau,
            "full_panel_internal_tau": field_tau,
            "estimates": estimates,
            "shared_only_analysis_ids": np.flatnonzero(a_mask & ~b_mask).tolist(),
            "isolated_only_analysis_ids": np.flatnonzero(b_mask & ~a_mask).tolist(),
        },
        "checks": {
            "both_competent": competent,
            "isolated_core": cell["core_passed"],
            "negative_conditional_tau": negative(tau),
            "negative_internal_tau": negative(field_tau),
            "internal_stable_errors": stable,
            "functional_controls": functional,
        },
        "outcome": outcome,
    }


def primary_result(lock: dict, spec: dict) -> dict:
    pairs = {str(seed): compare_pair(seed, spec) for seed in spec["seeds"]["mandatory"]}
    outcomes = {seed: pair["outcome"] for seed, pair in pairs.items()}
    count = sum(value == "supported_recovery" for value in outcomes.values())
    return {
        "schema_version": 1,
        "protocol_sha256": PROTOCOL_SHA256,
        "source_commit": lock["source_commit"],
        "artifact_lock": reference(ARTIFACT_LOCK),
        "seeds": pairs,
        "pair_outcomes": outcomes,
        "supported_recovery_pairs": count,
        "transport_triggered": count > 0,
        "claim_boundary": spec["claim_boundary"],
    }


def copy_record(path, target) -> dict:
    target.parent.mkdir(parents=True, exist_ok=True)
    with path.open("rb") as source, target.open("xb") as handle:
        shutil.copyfileobj(source, handle)
    if reference(path)["sha256"] != reference(target)["sha256"]:
        raise RuntimeError("archive copy differs")
    return reference(target)


def interval(row: dict, key="interval") -> str:
    point = row.get("point", row.get("mean"))
    ci = row[key]
    if point is None or ci["lower"] is None or ci["upper"] is None:
        return f"{point} [undefined]"
    return f"{point:.5f} [{ci['lower']:.5f}, {ci['upper']:.5f}]"


def render(result: dict) -> str:
    lines = [
        "# Weak-evidence routing: paired development result",
        "",
        f"Supported recovery: {result['supported_recovery_pairs']}/3 paired initializations. Conditional transport: {result['transport_status']}.",
        "",
        "The task data are project-exposed. Intervals describe participants within each network; networks are not pooled. The two arms retain the same local evidence and jointly train all slow parameters.",
        "",
        "| Seed | Route | Learned probability | Nonlearned probability | Behavioral tau [95% CI] | Analysis N | Core |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for seed, pair in result["seeds"].items():
        for arm in ("shared", "isolated"):
            cell = pair[arm]["cells"]["8"]
            p = cell["summaries"]["intact"]["probability"]
            lines.append(
                f"| {seed} | {arm} | {p['learned']['mean']:.5f} | {p['nonlearned']['mean']:.5f} | {interval(cell['behavior']['diversity'])} | {cell['behavior']['analysis_subjects']} | {cell['core_passed']} |"
            )
    lines += [
        "",
        "All paired differences are isolated minus shared.",
        "",
        "| Seed | Conditional tau difference | All-subject internal-order tau difference | Outcome |",
        "| --- | --- | --- | --- |",
    ]
    for seed, pair in result["seeds"].items():
        p = pair["paired"]
        lines.append(
            f"| {seed} | {interval(p['conditional_tau'])} | {interval(p['full_panel_internal_tau'])} | {pair['outcome']} |"
        )
    lines += ["", "## Accuracy and internal-error differences", ""]
    for seed, pair in result["seeds"].items():
        for name, row in pair["paired"]["estimates"].items():
            lines.append(f"- {seed} {name}: {interval(row, 'bootstrap')}.")
    lines += ["", "## Conditional transport", ""]
    for seed, pair in result["transport"].items():
        for arm, row in pair.items():
            for size, cell in row["cells"].items():
                p = cell["summaries"]["intact"]["probability"]["nonlearned"]
                lines.append(
                    f"- {seed} {arm} N={size}: nonlearned {interval(p, 'bootstrap')}; competence={cell['competence']}; core flags={cell['core_flags']}."
                )
    lines += [
        "",
        "## Claim boundaries",
        "",
        *[f"- {line}" for line in result["claim_boundary"]],
        "",
        "The canonical JSON also reports inclusion changes, undefined bootstrap draws, internal confidence-defined errors, local-off/fast-weight/evidence interventions, remote source-removal effects and all legacy qualitative rows. N6/N10 change support and query counts together with item count.",
        "",
    ]
    return "\n".join(lines)


def write_report():
    lock = validate_artifacts()
    spec = specification()
    result = primary_result(lock, spec)
    if json_ready(result) != load_json(RUN_ROOT / "primary.json"):
        raise RuntimeError("primary decision differs from the transport trigger")
    result["transport"] = {}
    result["artifacts"] = {}
    for seed in spec["seeds"]["mandatory"]:
        for condition in spec["seeds"]["conditions"]:
            directories = [run_directory(seed, condition)]
            _, _, directory = read_cell(seed, condition)
            directories.append(directory)
            if result["transport_triggered"]:
                row, _, directory = read_cell(seed, condition, transport=True)
                result["transport"].setdefault(str(seed), {})[condition] = row
                directories.append(directory)
            for directory in directories:
                phase = directory.parent.parent.name
                for path in sorted(directory.iterdir()):
                    if (
                        path.suffix == ".pth"
                        or path.name == "raw.npz"
                        or path.name.startswith("behavior-")
                    ):
                        name = f"seed-{seed}.{condition}.{phase}.{path.name}"
                        result["artifacts"][name] = copy_record(
                            path, RECORD_ROOT / "results" / name
                        )
    for name, row in load_json(SOURCE_LOCK)["inputs"].items():
        result["artifacts"][name] = copy_record(
            REPO_ROOT / row["file"]["path"], RECORD_ROOT / "inputs" / f"{name}.npz"
        )
    result["transport_status"] = (
        "completed_all_six_models" if result["transport_triggered"] else "not_triggered"
    )
    write_json_exclusive(RECORD_ROOT / "results" / "result.json", json_ready(result))
    target = RECORD_ROOT / "reports" / "report.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x") as handle:
        handle.write(render(result))
    return {
        "pair_outcomes": result["pair_outcomes"],
        "transport_status": result["transport_status"],
        "result": reference(RECORD_ROOT / "results" / "result.json"),
    }
