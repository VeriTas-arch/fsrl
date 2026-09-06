"""Complete eighteen-cell comparison and append-only portable archival."""

import shutil

from fsrl.experiments.encoding_contribution.protocol import (
    parent_result as structural_result,
)
from fsrl.experiments.encoding_contribution.reporting import verify_archive
from fsrl.experiments.encoding_contribution.statistics import failure_profile
from fsrl.experiments.minimal_learner.locks import validate_complete
from fsrl.experiments.structural_identification.evaluation import mean_interval
from fsrl.experiments.structural_identification.reporting import archive_group, profile
from fsrl.experiments.training_strategy.locks import reference
from fsrl.infra.provenance import load_json, write_json_exclusive

from .protocol import (
    RECORDS,
    RUN_ROOT,
    SEEDS,
    STRUCTURES,
    cell,
    parent_cohorts,
    parent_result,
    register,
    specification,
    validate_source,
)
from .statistics import (
    continuous_contrast,
    equivalence,
    failure_contrast,
    improvement,
    new_mismatches,
)


def cohorts():
    paths = sorted((RUN_ROOT / "liu").glob("*/result.json"))
    expected = {cell(seed, "G", donor) for seed in SEEDS for donor in STRUCTURES}
    if len(paths) != 100:
        raise RuntimeError("requires all 100 Gaussian cohorts")
    rows = []
    for path in paths:
        validate_complete(path.parent)
        row = load_json(path)
        if set(row) != expected:
            raise RuntimeError("Gaussian cohort cells incomplete")
        rows.append(row)
    return rows


def gaussian_profile(rows, seed, name):
    refs = structural_result()["measurement"]
    directory = RUN_ROOT / "generic" / name
    validate_complete(directory)
    generic = load_json(directory / "result.json")
    return {
        "generic": generic,
        "mechanism_interpretable": generic["passed"],
        **{
            observer: profile(
                [row[observer] for row in rows], refs[key], 7900001 + seed
            )
            for observer, key in (
                ("common", "references"),
                ("legacy", "legacy_reference"),
            )
        },
        "failure_profile": failure_profile(
            [row["common"] for row in rows], refs["references"]
        ),
        "internal": {
            key: mean_interval([row["internal"][key] for row in rows], 8500000 + seed)
            for key in rows[0]["internal"]
        },
        "tail_counts": {
            key: sum(row["tail"][key] for row in rows) for key in rows[0]["tail"]
        },
        "scalar_closure_max": max(row["scalar_closure_max"] for row in rows),
    }


def comparison(rows, parent, seed, donor, gaussian_profile):
    refs = structural_result()["measurement"]["references"]
    results = {}
    for structure, label in (("M11", "G_minus_Q"), ("M10", "G_minus_E")):
        previous = [row[cell(seed, structure, donor)] for row in parent]
        results[label] = {
            observer: continuous_contrast(
                [row[observer]["values"] for row in rows],
                [row[observer]["values"] for row in previous],
                8500000 + seed,
            )
            for observer in ("common", "legacy")
        }
        results[label]["internal"] = continuous_contrast(
            [row["internal"] for row in rows],
            [row["internal"] for row in previous],
            8500000 + seed,
        )
    quantized = [row[cell(seed, "M11", donor)]["common"] for row in parent]
    failures = failure_contrast(
        [row["common"] for row in rows], quantized, refs, 8500000 + seed
    )
    bounds = specification()["analysis"]["equivalence_bounds"]
    statuses = {
        key: equivalence(results["G_minus_Q"]["common"][key], bound)
        for key, bound in bounds.items()
    }
    mismatch = new_mismatches(
        gaussian_profile["common"]["estimates"],
        parent_result()["cells"][cell(seed, "M11", donor)]["common"]["estimates"],
        refs,
    )
    passed = gaussian_profile["mechanism_interpretable"]
    return {
        **results,
        "failure_changes": failures,
        "primary_status": statuses,
        "primary_equivalent": passed
        and all(status == "equivalent" for status in statuses.values()),
        "new_sustained_mismatches": mismatch,
        "tradeoff_improved": passed
        and improvement(
            gaussian_profile["common"],
            results["G_minus_Q"]["common"],
            failures,
            mismatch,
            refs,
        ),
        "mechanism_interpretable": passed,
    }


def summarize():
    lock = validate_source()
    observed, previous = cohorts(), parent_cohorts()
    cells = dict(parent_result()["cells"])
    comparisons = {}
    for seed in SEEDS:
        for donor in STRUCTURES:
            name = cell(seed, "G", donor)
            rows = [row[name] for row in observed]
            cells[name] = gaussian_profile(rows, seed, name)
            cells[name]["parameters"] = cells[cell(seed, donor, donor)]["parameters"]
            comparisons[name] = comparison(rows, previous, seed, donor, cells[name])
    qualified = all(row["mechanism_interpretable"] for row in comparisons.values())
    equivalent = all(row["primary_equivalent"] for row in comparisons.values())
    result = {
        "experiment_id": "encoding_distribution_v1",
        "source_commit": lock["source_commit"],
        "condition_aliases": {"M10": "E", "M11": "Q", "G": "G"},
        "cells": cells,
        "comparisons": comparisons,
        "all_competence_passed": qualified,
        "persistent_primary_equivalence": equivalent,
        "persistent_tradeoff_improvement": all(
            row["tradeoff_improved"] for row in comparisons.values()
        ),
        "K_check_trigger": equivalent
        and all(not row["new_sustained_mismatches"] for row in comparisons.values()),
        "inherited_Q_minus_E": {
            seed: {donor: operations[f"encoding_at_{donor}"] for donor in STRUCTURES}
            for seed, operations in parent_result()["paired_contrasts"].items()
        },
        "human_context": parent_result()["human_finite_cohort"],
        "parent_outcome_changed": False,
        "promotion": False,
        "boundary": "Fixed-parameter exposed-data diagnostic. Equivalence is confined to the four predeclared primary endpoints and tolerances, not arbitrary distributions or full behavior. G retains the codebook variance function and relaxes support range.",
    }
    write_json_exclusive(RUN_ROOT / "summary.json", result)
    return {
        key: result[key]
        for key in (
            "all_competence_passed",
            "persistent_primary_equivalence",
            "persistent_tradeoff_improvement",
            "K_check_trigger",
        )
    }


def publish():
    validate_source()
    result = load_json(RUN_ROOT / "summary.json")
    result["conditional_K_check"] = load_json(RUN_ROOT / "K_check.json")
    write_json_exclusive(RECORDS / "results/encoding_distribution_v1.json", result)
    groups = {
        "liu": sorted((RUN_ROOT / "liu").glob("*/outputs.npz")),
        "generic": sorted((RUN_ROOT / "generic").glob("*/outputs.npz")),
        "K_check": sorted((RUN_ROOT / "K_check").glob("*/outputs.npz")),
    }
    mapping = {}
    for family, paths in groups.items():
        for start in range(0, len(paths), 5):
            subset = paths[start : start + 5]
            destination = RECORDS / "results" / f"{family}-{start:03}.npz"
            archive_group(subset, destination)
            refs = [reference(path) for path in subset]
            verify_archive(destination, refs)
            mapping[destination.name] = refs
    rows = cohorts()
    for start in range(0, 100, 10):
        write_json_exclusive(
            RECORDS / "results" / f"liu-rows-{start:03}.json", rows[start : start + 10]
        )
    for name in ("source_lock.json", "qualification.json"):
        with (
            (RECORDS / "benchmarks" / name).open("xb") as out,
            (RUN_ROOT / name).open("rb") as src,
        ):
            shutil.copyfileobj(src, out)
    write_json_exclusive(RECORDS / "results/archive_map.json", mapping)
    if any(
        path.stat().st_size > 5000000 for path in RECORDS.rglob("*") if path.is_file()
    ):
        raise RuntimeError("record exceeds inline review threshold")
    register(
        "supporting" if result["all_competence_passed"] else "unresolved",
        "Completed fixed-parameter moment-matched distribution comparison with all primary equivalence and failure diagnostics; no main-model promotion.",
    )
    return {
        "archives": len(mapping),
        "byte_reconstruction_verified": True,
        "promotion": False,
    }
