"""Archive the prospective comparison without changing its outcome rules."""

import shutil

import numpy as np

from fsrl.experiments.training_strategy.estimands import estimate
from fsrl.experiments.training_strategy.evaluation import json_ready
from fsrl.experiments.training_strategy.locks import reference
from fsrl.infra.file_contracts import validate_run_manifest
from fsrl.infra.provenance import load_json, write_json_exclusive

from .locks import ARTIFACT_LOCK, validate_artifacts
from .measurement import above
from .protocol import (
    PROTOCOL_SHA256,
    RECORD_ROOT,
    RUN_ROOT,
    run_directory,
    specification,
)


def paired_comparison(dual: dict, single: dict, seed: int, spec: dict) -> dict:
    prefix = "N8__endpoints__intact__probability__"
    differences = {
        group: single[prefix + group] - dual[prefix + group]
        for group in ("learned", "nonlearned", "omitted")
    }
    differences["dual_omitted_benefit"] = -differences["omitted"]
    differences["dual_omitted_specificity"] = (
        differences["nonlearned"] - differences["omitted"]
    )
    summaries = {
        name: estimate(
            values,
            seed=spec["statistics"]["seed_offset"] + seed,
            statistics=spec["statistics"],
        )
        for name, values in differences.items()
    }
    margin = spec["decision"]["noninferiority_margin"]
    return {
        "estimates": summaries,
        "single_noninferior": all(
            summaries[group]["bootstrap"]["lower"] is not None
            and summaries[group]["bootstrap"]["lower"] >= -margin
            for group in ("learned", "nonlearned", "omitted")
        ),
        "local_specific_benefit": above(summaries["dual_omitted_benefit"], 0)
        and above(summaries["dual_omitted_specificity"], 0),
        "local_material_benefit": above(summaries["dual_omitted_benefit"], margin),
    }


def outcome(rows: dict) -> str:
    pairs = list(rows.values())
    competent = all(
        pair[arm]["generic"]["competence"] and pair[arm]["cells"]["8"]["competence"]
        for pair in pairs
        for arm in ("dual", "single")
    )
    if not competent:
        return "competence_limited"
    single_core = all(pair["single"]["cells"]["8"]["core_passed"] for pair in pairs)
    dual_core = all(pair["dual"]["cells"]["8"]["core_passed"] for pair in pairs)
    if single_core and all(pair["paired"]["single_noninferior"] for pair in pairs):
        return "single_preferred"
    specific = all(
        pair["paired"]["local_specific_benefit"]
        and above(
            pair["dual"]["cells"]["8"]["effects"]["intact_minus_local_off_omitted"], 0
        )
        for pair in pairs
    )
    material = sum(pair["paired"]["local_material_benefit"] for pair in pairs)
    if dual_core and specific and material >= 2:
        return "local_value_supported"
    if not any(
        pair[arm]["cells"]["8"]["core_passed"]
        for pair in pairs
        for arm in ("dual", "single")
    ):
        return "both_fail_core"
    return "heterogeneous_or_unresolved"


def archive_condition(seed: int, condition: str) -> tuple:
    directory = RUN_ROOT / "evaluation" / str(seed) / condition
    manifest = load_json(directory / "run.json")
    if (
        manifest["lifecycle_state"] != "complete"
        or not validate_run_manifest(directory / "run.json")["passed"]
    ):
        raise RuntimeError("cannot archive an incomplete or modified evaluation")
    result = load_json(directory / "result.json")
    if result["protocol_sha256"] != PROTOCOL_SHA256 or result[
        "artifact_lock"
    ] != reference(ARTIFACT_LOCK):
        raise RuntimeError("evaluation identity differs from frozen study")
    artifacts = {}
    paths = list(directory.glob("*.npz")) + list(directory.glob("behavior-*.json"))
    paths += list(run_directory(seed, condition).glob("*.pth"))
    destination = RECORD_ROOT / "results"
    destination.mkdir(parents=True, exist_ok=True)
    for path in paths:
        target = destination / f"seed-{seed}.{condition}.{path.name}"
        with path.open("rb") as source, target.open("xb") as handle:
            shutil.copyfileobj(source, handle)
        artifacts[path.name] = reference(target)
    with np.load(directory / "raw.npz", allow_pickle=False) as data:
        raw = {key: data[key] for key in data.files}
    result["artifacts"] = artifacts
    return result, raw


def render(result: dict) -> str:
    lines = [
        "# Information-matched RNN memory structure",
        "",
        f"Registered development outcome: **{result['outcome']}**.",
        "",
        "Three paired initializations; every candidate jointly trained for 48,000 generic episodes with the same two effective-evidence inputs. Human interval inclusion is descriptive. No seeds or model outcomes were pooled.",
        "",
        "| Seed | Condition | N | Learned probability | Nonlearned probability | Omitted probability | Core |",
        "|---|---|---|---|---|---|---|",
    ]
    for seed, pair in result["seeds"].items():
        for arm in ("dual", "single"):
            for size, cell in pair[arm]["cells"].items():
                p = cell["summaries"]["intact"]["probability"]
                numbers = [
                    f"{p[group]['mean']:.4f}"
                    for group in ("learned", "nonlearned", "omitted")
                ]
                lines.append(
                    f"| {seed} | {arm} | {size} | {' | '.join(numbers)} | {cell['core_passed']} |"
                )
    lines += [
        "",
        "## Paired N=8 differences",
        "",
        "Single minus dual, except explicitly named dual-benefit contrasts; 95% participant bootstrap intervals within each network.",
        "",
    ]
    for seed, pair in result["seeds"].items():
        for name, row in pair["paired"]["estimates"].items():
            ci = row["bootstrap"]
            lines.append(
                f"- {seed} {name}: {row['mean']:.6f} [{ci['lower']:.6f}, {ci['upper']:.6f}]."
            )
    lines += [
        "",
        "## Boundaries",
        "",
        *[f"- {line}" for line in result["claim_boundary"]],
        "",
        "N=6/10 are frozen-parameter transport under covarying support/query counts. Their descriptive core flags do not establish size-invariant binary stable-error prevalence. The canonical JSON retains all gates, effects, nine-row classifications, costs and artifact identities.",
        "",
    ]
    return "\n".join(lines)


def write_report():
    lock = validate_artifacts()
    spec = specification()
    rows = {}
    for seed in spec["seeds"]["mandatory"]:
        dual, dual_raw = archive_condition(seed, "dual")
        single, single_raw = archive_condition(seed, "single")
        rows[str(seed)] = {
            "dual": dual,
            "single": single,
            "paired": paired_comparison(dual_raw, single_raw, seed, spec),
        }
    result = {
        "schema_version": 1,
        "protocol_sha256": PROTOCOL_SHA256,
        "source_commit": lock["source_commit"],
        "artifact_lock": reference(ARTIFACT_LOCK),
        "outcome": outcome(rows),
        "seeds": rows,
        "claim_boundary": spec["claim_boundary"],
    }
    write_json_exclusive(RECORD_ROOT / "results" / "result.json", json_ready(result))
    report = RECORD_ROOT / "reports" / "report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("x") as handle:
        handle.write(render(result))
    return {
        "outcome": result["outcome"],
        "result": reference(RECORD_ROOT / "results" / "result.json"),
    }
