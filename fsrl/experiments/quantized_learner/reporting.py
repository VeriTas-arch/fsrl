"""Publish all nine fits and their controls; never select a partial winner."""

import shutil

from fsrl.experiments.training_strategy.evaluation import json_ready
from fsrl.experiments.training_strategy.locks import reference, verify_reference
from fsrl.infra.provenance import load_json, write_json_exclusive

from .decisions import recipe_decision
from .evaluation import validate_evaluation
from .evidence import ARTIFACT_LOCK, RECOVERY_RESULT, validate_artifacts
from .protocol import PROTOCOL_HASH, RECORDS, specification
from .verification import verify_fit

RESULT = RECORDS / "results/quantized_relational_learner_v1.json"
REPORT = RECORDS / "reports/quantized_relational_learner_v1.md"


def publish() -> dict:
    lock = validate_artifacts()
    spec = specification()
    fits = {}
    for seed in spec["seeds"]["mandatory"]:
        for condition in spec["seeds"]["conditions"]:
            result = validate_evaluation(seed, condition, lock)
            result["verification"] = verify_fit(result)
            fits[f"{seed}/{condition}"] = result
    # No public partial model result: validate every mandatory fit before copying.
    for identity, result in fits.items():
        destination = RECORDS / "results" / identity.replace("/", "-")
        destination.mkdir(parents=True, exist_ok=False)
        for name, ref in result["files"].items():
            source = verify_reference(ref)
            target = destination / source.name
            shutil.copyfile(source, target)
            result["files"][name] = reference(target)
        target = destination / "behavior.json"
        shutil.copyfile(verify_reference(result["sampled_behavior"]), target)
        result["sampled_behavior"] = reference(target)
    recovery = load_json(RECOVERY_RESULT)
    recipes = {
        condition: recipe_decision(
            {
                str(seed): fits[f"{seed}/{condition}"]
                for seed in spec["seeds"]["mandatory"]
            },
            spec["seeds"]["mandatory"],
            recovery["summary"]["outcome"],
        )
        for condition in spec["seeds"]["conditions"]
    }
    result = {
        "experiment_id": spec["experiment_id"],
        "protocol_sha256": PROTOCOL_HASH,
        "artifact_lock": reference(ARTIFACT_LOCK),
        "source_commit": lock["source_commit"],
        "recovery": reference(RECOVERY_RESULT),
        "fits": fits,
        "recipes": recipes,
        "stop_rule": spec["decision"]["stop"],
        "promotion_boundary": spec["decision"]["promotion"],
    }
    write_json_exclusive(RESULT, json_ready(result))
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.open("x").write(render_report(result))
    return {
        "recipes": recipes,
        "result": reference(RESULT),
        "report": reference(REPORT),
    }


def render_report(result: dict) -> str:
    lines = [
        "# Fixed four-valued relational teaching-code pilot",
        "",
        "All three paired training streams and all nine final fits are reported. No participant pooling, human refitting, post-evaluation codebook repair or main-model promotion.",
        "",
        "| Fit | Generic competence | Qualitative | Quantitative | Binding | Strict correct internal orders (Liu) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for identity, fit in result["fits"].items():
        flags = fit["behavior"]["flags"]
        decision = fit["decision"]
        lines.append(
            f"| {identity} | {all(decision['competence'].values())} | {sum(row['qualitative'] for row in flags.values())}/9 | {sum(row['calibration'] for row in flags.values())}/9 | {decision['binding_passed']} | {fit['internal']['liu']['strict_correct_order_count']}/77 |"
        )
    for identity, fit in result["fits"].items():
        lines.extend(
            [
                "",
                f"## {identity}",
                "",
                f"Parameters: {fit['parameters']}. Fixed-parameter codec control uses Exact parameters: {fit['fixed_parameters']}.",
                "",
                "| Original behavioral row | Qualitative | Quantitative |",
                "| --- | --- | --- |",
            ]
        )
        lines.extend(
            f"| {name} | {row['qualitative']} | {row['calibration']} |"
            for name, row in fit["behavior"]["flags"].items()
        )
    lines.extend(["", "## Registered decisions", ""])
    lines.extend(
        f"- {condition}: `{row['outcome']}`; eligible for unchanged replication: {row['eligible_for_unchanged_replication']}."
        for condition, row in result["recipes"].items()
    )
    lines.extend(
        [
            "",
            "Full original metrics, denominators, uncertainty, per-fit controls, parameter archives and verification are in the companion result JSON and linked arrays. Conditional code enumeration is not a new participant cohort.",
            "",
            "Two-bit code content excludes cue-address storage, admission state and continuous score weights. Existing score-circuit results do not automatically cover these fits or the encoder. Human mechanism identification is not claimed.",
            "",
            result["stop_rule"],
            "",
            result["promotion_boundary"],
            "",
        ]
    )
    return "\n".join(lines)


def verify_record() -> dict:
    lock = validate_artifacts()
    result = load_json(RESULT)
    if set(result["fits"]) != set(lock["archives"]):
        raise RuntimeError("published results omit a mandatory fit")
    for identity, fit in result["fits"].items():
        config = lock["archives"][identity]["config"]
        if (
            fit["raw_parameters"] != config["raw_parameters"]
            or fit["parameters"] != config["physical_parameters"]
        ):
            raise RuntimeError("published fit differs from its locked parameters")
    checks = {identity: verify_fit(row) for identity, row in result["fits"].items()}
    spec = specification()
    recovery = load_json(verify_reference(result["recovery"]))
    expected = {
        condition: recipe_decision(
            {
                str(seed): result["fits"][f"{seed}/{condition}"]
                for seed in spec["seeds"]["mandatory"]
            },
            spec["seeds"]["mandatory"],
            recovery["summary"]["outcome"],
        )
        for condition in spec["seeds"]["conditions"]
    }
    if result["recipes"] != expected or REPORT.read_text() != render_report(result):
        raise RuntimeError("published classification/report does not reconstruct")
    return {"passed": True, "fits": checks}
