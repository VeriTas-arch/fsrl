"""Complete diagnostic tables; interpretation never promotes a new model."""

from __future__ import annotations

import shutil

from fsrl.experiments.training_strategy.locks import reference
from fsrl.infra.provenance import load_json, write_json_exclusive

from .evidence import LOCK, RECORD_ROOT, specification, validate_lock
from .verification import verify_run


def number(value) -> str:
    return "undefined" if value is None else f"{value:.6f}"


def table(title: str, rows: dict) -> list[str]:
    lines = [
        "",
        title,
        "",
        "| Estimand | Mean | 95% lower | 95% upper | N |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, row in rows.items():
        interval = row["bootstrap"]
        lines.append(
            f"| {name} | {number(row['mean'])} | {number(interval['lower'])} | {number(interval['upper'])} | {row['subjects']} |"
        )
    return lines


def report_text(result: dict, validation: dict) -> str:
    lines = [
        "# Frozen minimal learner: quantitative-source diagnosis",
        "",
        "Three mandatory exposed training streams; no training, parameter selection, human fitting, or additional seeds. Registered outcome is diagnostic localization, not a main-model promotion or a quantitative-equivalence test.",
        "",
        "RF/AF = retained/all-observed admission with the original finite update. RL/AL = retained/all-observed minimum-norm least-squares references. All cells use each score-only fit's unchanged gain and T=0.25. L cells use observed support constraints, never query labels; they are offline references, not candidate models or human posteriors.",
        "",
        "Every estimate is a separate within-fit participant bootstrap (10,000 draws, 95% percentile). Missing group subjects are removed before resampling; JSON contains their exact indices. The direction labels require each of the three intervals to lie on the same side of zero; they are descriptive and not familywise confirmation. All registered contrasts are reported below.",
        "",
        f"Protocol witness: `{result['protocol_commit']}`. Implementation witness: `{result['source_commit']}`. Execution-lock witness: `{result['execution_commit']}`.",
        f"Independent validation passed: `{validation['passed']}`.",
        "",
        "## Cross-stream descriptive directions",
        "",
        "| Domain | Contrast | Endpoint | Direction |",
        "| --- | --- | --- | --- |",
    ]
    for domain, contrasts in result["directions"].items():
        for contrast, endpoints in contrasts.items():
            for name, label in endpoints.items():
                lines.append(f"| {domain} | {contrast} | {name} | {label} |")
    for seed, row in result["fits"].items():
        lines += [
            "",
            f"## Training stream {seed}",
            "",
            f"Frozen parameters: `{row['parameters']}`.",
        ]
        global_result = row["global"]
        for heading, domain in (
            ("Global reference", "cells"),
            ("Global contrast", "contrasts"),
            ("Readout accounting", "readout"),
        ):
            for name, endpoints in global_result[domain].items():
                lines += table(f"### {heading}: {name}", endpoints)
        lines += table(
            "### Retained graph coverage (fixed connected/disconnected strata)",
            global_result["coverage"],
        )
        lines += table(
            "### Frozen probability minus exact decision",
            global_result["readout_difference"],
        )
        for domain in ("cells", "effects", "between_recipe"):
            for name, endpoints in row["local"][domain].items():
                lines += table(f"### Local {domain}: {name}", endpoints)
        lines += [
            "",
            "### Original behavior anchors (unchanged)",
            "",
            "| Recipe | Qualitative rows | Frozen quantitative rows |",
            "| --- | --- | --- |",
        ]
        for condition, behavior in row["parent_behavior"].items():
            flags = behavior["flags"].values()
            lines.append(
                f"| {condition} | {sum(r['qualitative'] for r in flags)}/9 | {sum(r['calibration'] for r in flags)}/9 |"
            )
    lines += [
        "",
        "## Interpretation boundaries",
        "",
        "Strict latent correct order uses all 77 subjects and a fixed raw-score tie tolerance. It is not the old sampled-choice/Hodge classification or its eligible-subject cohort. Serial contrast uses the two endpoints versus six interior positions; probability profiles are expected-choice diagnostics, not fresh sampled behavioral classification.",
        "Positive gain cannot change latent ordering, but can change sampling. No optimal readout or encoding-precision parameter is identified here. Reference cells need not be closer to humans when their task accuracy increases.",
        "Local self/cross terms use relation identity only for offline attribution. They do not authorize a learned-query flag or self-only model. The two-order sigmoid allocation is an exact response attribution, not an independent neural circuit. Between-recipe effects separately retain changes in fitted global parameters.",
        "No new noise family, training, calibration or additional evaluation is admitted after these fixed analyses. The frozen parent evidence and closed family outcomes remain unchanged.",
        "",
    ]
    return "\n".join(lines)


def publish(directory) -> dict:
    validate_lock(pushed=True)
    spec = specification()
    validation = verify_run(directory, spec)
    result = load_json(directory / "result.json")
    target = RECORD_ROOT / "results/minimal_learner_diagnostics_v1.npz"
    target.parent.mkdir(parents=True, exist_ok=True)
    with (directory / "arrays.npz").open("rb") as reader, target.open("xb") as writer:
        shutil.copyfileobj(reader, writer)
    result["registered_arrays"] = reference(target)
    if result["registered_arrays"]["sha256"] != result["arrays"]["sha256"]:
        raise RuntimeError("registered arrays differ from executed arrays")
    result["execution_lock"] = reference(LOCK)
    paths = {
        "result": RECORD_ROOT / "results/minimal_learner_diagnostics_v1.json",
        "validation": RECORD_ROOT
        / "results/minimal_learner_diagnostics_v1.validation.json",
        "report": RECORD_ROOT / "reports/minimal_learner_diagnostics_v1.md",
    }
    write_json_exclusive(paths["result"], result)
    write_json_exclusive(paths["validation"], validation)
    paths["report"].parent.mkdir(parents=True, exist_ok=True)
    with paths["report"].open("x") as handle:
        handle.write(report_text(result, validation))
    return {name: reference(path) for name, path in {**paths, "arrays": target}.items()}
