"""Portable complete-matrix evidence; a diagnostic never promotes its parent."""

from fsrl.experiments.quantized_learner.protocol import resolved_specification
from fsrl.experiments.training_strategy.behavior import human_references
from fsrl.experiments.training_strategy.evaluation import json_ready, write_arrays
from fsrl.experiments.training_strategy.locks import reference, verify_reference
from fsrl.infra.provenance import load_json, write_json_exclusive

from .execution import evaluated_shard, verify_shard
from .inputs import read_arrays
from .locks import LOCK, validate_lock
from .protocol import PROTOCOL_HASH, RECORDS, load_parameters, specification
from .statistics import summarize_fit

RESULT = RECORDS / "results/resampled_cohort_diagnostic_v1.json"
REPORT = RECORDS / "reports/resampled_cohort_diagnostic_v1.md"


def summarize(shards: list[dict]) -> dict:
    spec = specification()
    rows = [row for shard in shards for row in shard["points"]]
    references = human_references(resolved_specification())
    fits = {
        str(seed): summarize_fit(
            [{"cohort": row["cohort"], **row["fits"][str(seed)]} for row in rows],
            seed,
            spec,
            references,
        )
        for seed in spec["fits"]
    }
    outcomes = {row["outcome"] for row in fits.values()}
    return {
        "fits": fits,
        "outcome": next(iter(outcomes))
        if len(outcomes) == 1
        else "training_fit_heterogeneity",
    }


def render_report(result: dict) -> str:
    lines = [
        "# Fixed Resampled cohort diagnostic",
        "",
        f"Diagnostic outcome: `{result['outcome']}`. All 400 independent 77-person cohorts per frozen fit are included; no parameter adaptation, participant pooling or parent-outcome revision.",
        "",
        "| Fit | Mean distance slope | Whole-cohort 95% CI | Classification | All-nine joint cohort pass rate |",
        "| --- | --- | --- | --- | --- |",
    ]
    for key, row in result["fits"].items():
        slope = row["continuous"]["symbolic_distance_effect"]
        lines.append(
            f"| {key} | {slope['mean']} | {slope['interval']} | {row['outcome']} | {row['all_nine']['joint']} |"
        )
        lines.extend(
            [
                "",
                f"## {key}: all original continuous endpoints",
                "",
                "| Endpoint | Mean | 95% CI | Original reference | Classification | Undefined cohorts |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        )
        lines.extend(
            f"| {name} | {value['mean']} | {value['interval']} | {value['reference']} | {value['classification']} | {len(value['undefined_cohorts'])} |"
            for name, value in row["continuous"].items()
        )
        lines.extend(
            [
                "",
                "| Original behavior row | Qualitative pass rate | Quantitative pass rate |",
                "| --- | --- | --- |",
            ]
        )
        lines.extend(
            f"| {name} | {value['qualitative']} | {value['calibration']} |"
            for name, value in row["pass_rates"].items()
        )
    lines.extend(
        [
            "",
            "Pointwise simulation uncertainty, not new human equivalence intervals. The input evidence, temperature and human reference intervals retain their historical exposure. All six morphology counts per cohort and the joint bimodal/unimodal/low-accuracy distribution are retained in linked records.",
            "",
            "This is a diagnostic of the failed pilot, not a new admission test. The original partial_behavioral_reproduction outcome remains unchanged; no main model is promoted.",
            "",
            result["stop_rule"],
            "",
        ]
    )
    return "\n".join(lines)


def publish() -> dict:
    lock = validate_lock()
    configs = load_parameters()
    size = specification()["cohorts"]["shard_size"]
    shards, verification = [], []
    for offset, input_ref in enumerate(lock["cohort_shards"]):
        start = offset * size
        shard = evaluated_shard(input_ref, start)
        verification.append(verify_shard(shard, input_ref, start, configs))
        shards.append(shard)
        print(f"Reconstructed cohorts {start}..{start + size - 1}", flush=True)
    result = {
        "experiment_id": specification()["experiment_id"],
        "protocol_sha256": PROTOCOL_HASH,
        "execution_lock": reference(LOCK),
        **summarize(shards),
        "verification": verification,
        "stop_rule": specification()["classification"]["stop"],
    }
    destination = RECORDS / "results"
    destination.mkdir(parents=True, exist_ok=False)
    refs = []
    for offset, shard in enumerate(shards):
        start = offset * size
        path = destination / f"outputs-{start:03d}.npz"
        write_arrays(path, read_arrays(shard["arrays"]))
        published = {**shard, "arrays": reference(path)}
        path = destination / f"cohorts-{start:03d}.json"
        write_json_exclusive(path, published)
        refs.append(reference(path))
    result["shards"] = refs
    write_json_exclusive(RESULT, json_ready(result))
    REPORT.parent.mkdir(parents=True, exist_ok=False)
    REPORT.open("x").write(render_report(result))
    return {
        "outcome": result["outcome"],
        "result": reference(RESULT),
        "report": reference(REPORT),
    }


def verify_record() -> dict:
    lock = validate_lock()
    result = load_json(RESULT)
    if (
        result["execution_lock"] != reference(LOCK)
        or result["protocol_sha256"] != PROTOCOL_HASH
    ):
        raise RuntimeError("result source/protocol identity differs")
    if len(result["shards"]) != len(lock["cohort_shards"]):
        raise RuntimeError("result omits a mandatory shard")
    configs = load_parameters()
    shards = [load_json(verify_reference(ref)) for ref in result["shards"]]
    size = specification()["cohorts"]["shard_size"]
    checks = [
        verify_shard(shard, input_ref, index * size, configs)
        for index, (shard, input_ref) in enumerate(
            zip(shards, lock["cohort_shards"], strict=True)
        )
    ]
    if checks != result["verification"]:
        raise RuntimeError("saved independent verification differs")
    for key, value in summarize(shards).items():
        if json_ready(value) != result[key]:
            raise RuntimeError("cohort diagnostic statistics do not reconstruct")
    if REPORT.read_text() != render_report(result):
        raise RuntimeError("cohort diagnostic report differs")
    return {
        "passed": True,
        "fits": len(configs),
        "cohorts_per_fit": lock["cohort_count"],
        "outcome": result["outcome"],
        "max_recurrence_error": max(row["max_recurrence_error"] for row in checks),
    }
