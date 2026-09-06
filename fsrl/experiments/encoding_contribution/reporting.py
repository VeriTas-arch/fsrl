"""Complete per-network summaries and portable append-only archival."""

import hashlib
import io
import shutil

import numpy as np

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
    parameters,
    parent_result,
    register,
    validate_source,
)
from .statistics import failure_profile


def read_units(family, count):
    paths = sorted((RUN_ROOT / family).glob("*/result.json"))
    if len(paths) != count:
        raise RuntimeError(f"incomplete {family}: {len(paths)} of {count}")
    for path in paths:
        validate_complete(path.parent)
    return [load_json(path) for path in paths]


def contrasts(cohorts, seed):
    result = {}
    operations = {
        **{
            f"encoding_at_{donor}": [(1, "M11", donor), (-1, "M10", donor)]
            for donor in STRUCTURES
        },
        **{
            f"parameters_at_{st}": [(1, st, "M11"), (-1, st, "M10")]
            for st in STRUCTURES
        },
        "interaction": [
            (1, "M11", "M11"),
            (-1, "M10", "M11"),
            (-1, "M11", "M10"),
            (1, "M10", "M10"),
        ],
    }
    for operation, terms in operations.items():
        result[operation] = {}
        for observer in ("common", "legacy", "internal"):
            sources = [
                (sign, [row[cell(seed, st, donor)][observer] for row in cohorts])
                for sign, st, donor in terms
            ]
            if observer != "internal":
                sources = [
                    (sign, [row["values"] for row in rows]) for sign, rows in sources
                ]
            result[operation][observer] = {
                endpoint: mean_interval(
                    sum(
                        sign * np.asarray([row[endpoint] for row in rows], dtype=float)
                        for sign, rows in sources
                    ),
                    8400000 + seed,
                )
                for endpoint in sources[0][1][0]
            }
    return result


def cell_profile(cohorts, seed, st, donor):
    rows = [row[cell(seed, st, donor)] for row in cohorts]
    parent = parent_result()
    refs = parent["measurement"]
    profiles = {
        observer: profile([row[observer] for row in rows], refs[key], 7900001 + seed)
        for observer, key in (("common", "references"), ("legacy", "legacy_reference"))
    }
    if st == donor:
        previous = parent["fits"][f"{seed}-{st}-decay"]
        if any(profiles[key] != previous[key] for key in profiles):
            raise RuntimeError(
                "diagonal observation profile differs from frozen parent"
            )
        generic = previous["generic"]
        passed = generic["competence"] and generic["binding_pass"]
    else:
        directory = RUN_ROOT / "generic" / cell(seed, st, donor)
        validate_complete(directory)
        generic = load_json(directory / "result.json")
        passed = generic["passed"]
    return {
        "parameters": parameters(seed, donor),
        "generic": generic,
        "mechanism_interpretable": bool(passed),
        **profiles,
        "failure_profile": failure_profile(
            [row["common"] for row in rows], refs["references"]
        ),
        "internal": {
            key: mean_interval([row["internal"][key] for row in rows], 8400000 + seed)
            for key in rows[0]["internal"]
        },
        "moments": {
            key: (
                max(row["moments"][key] for row in rows)
                if key == "mean_parity_max"
                else mean_interval(
                    [row["moments"][key] for row in rows], 8400000 + seed
                )
            )
            for key in rows[0].get("moments", {})
        },
    }


def persistent_directions(results):
    names = (
        "nonlearned_accuracy",
        "correct_ranker",
        "self_consistent_incorrect",
        "self_inconsistent",
    )
    verdicts = {}
    for name in names:
        values = [
            results[str(seed)][f"encoding_at_{donor}"]["common"][name]
            for seed in SEEDS
            for donor in STRUCTURES
        ]
        positive = all(v["lower"] is not None and v["lower"] > 0 for v in values)
        negative = all(v["upper"] is not None and v["upper"] < 0 for v in values)
        verdicts[name] = (
            "positive"
            if positive
            else "negative"
            if negative
            else "heterogeneous_or_unresolved"
        )
    return verdicts


def summarize():
    lock = validate_source()
    cohorts = read_units("cross", 100)
    human_rows = [row for chunk in read_units("finite_cohort", 10) for row in chunk]
    if len(human_rows) != 1000:
        raise RuntimeError("human bootstrap incomplete")
    validate_complete(RUN_ROOT / "manipulations")
    comparisons = {str(seed): contrasts(cohorts, seed) for seed in SEEDS}
    cells = {
        cell(seed, st, donor): cell_profile(cohorts, seed, st, donor)
        for seed in SEEDS
        for st in STRUCTURES
        for donor in STRUCTURES
    }
    result = {
        "experiment_id": "encoding_contribution_v1",
        "source_commit": lock["source_commit"],
        "all_competence_passed": all(
            row["mechanism_interpretable"] for row in cells.values()
        ),
        "diagonal_parent_profiles_exact": True,
        "cells": cells,
        "paired_contrasts": comparisons,
        "persistent_primary_directions": persistent_directions(comparisons),
        "human_finite_cohort": failure_profile(
            human_rows, parent_result()["measurement"]["references"]
        ),
        "manipulations": load_json(RUN_ROOT / "manipulations/result.json"),
        "parent_outcome_changed": False,
        "promotion": False,
        "boundary": "Fixed-parameter development diagnostics on exposed inputs; no new training, humans, confirmation or population-level inference. Wilson intervals quantify Monte Carlo rates only. Profile intervals retain the parent's seed for exact diagonal verification; paired contrasts use the new registered seed.",
    }
    write_json_exclusive(RECORDS / "results/encoding_contribution_v1.json", result)
    return {
        key: result[key]
        for key in (
            "all_competence_passed",
            "diagonal_parent_profiles_exact",
            "persistent_primary_directions",
            "promotion",
        )
    }


def verify_archive(path, references):
    with np.load(path, allow_pickle=False) as saved:
        for index, ref in enumerate(references):
            prefix = f"member{index:03}__"
            buffer = io.BytesIO()
            np.savez_compressed(
                buffer,
                **{
                    key.removeprefix(prefix): saved[key]
                    for key in saved.files
                    if key.startswith(prefix)
                },
            )
            payload = buffer.getvalue()
            if (
                len(payload) != ref["bytes"]
                or hashlib.sha256(payload).hexdigest() != ref["sha256"]
            ):
                raise RuntimeError("archive byte reconstruction differs")


def publish():
    validate_source()
    result = load_json(RECORDS / "results/encoding_contribution_v1.json")
    mapping = {}
    families = {
        "cross": sorted((RUN_ROOT / "cross").glob("*/outputs.npz")),
        "generic": sorted((RUN_ROOT / "generic").glob("*/outputs.npz")),
        "finite_indices": sorted((RUN_ROOT / "finite_cohort").glob("*/indices.npz")),
        "manipulations": [RUN_ROOT / "manipulations/outputs.npz"],
    }
    for family, paths in families.items():
        for start in range(0, len(paths), 5):
            subset = paths[start : start + 5]
            destination = RECORDS / "results" / f"{family}-{start:03}.npz"
            archive_group(subset, destination)
            if destination.stat().st_size > 5000000:
                raise RuntimeError("inline archive too large")
            refs = [reference(path) for path in subset]
            verify_archive(destination, refs)
            mapping[destination.name] = refs
    for family, count in (("cross", 100), ("finite_cohort", 10)):
        rows = read_units(family, count)
        for start in range(0, len(rows), 10):
            write_json_exclusive(
                RECORDS / "results" / f"{family}-rows-{start:03}.json",
                rows[start : start + 10],
            )
    for name in ("source_lock.json", "qualification.json"):
        with (
            (RECORDS / "benchmarks" / name).open("xb") as out,
            (RUN_ROOT / name).open("rb") as src,
        ):
            shutil.copyfileobj(src, out)
    write_json_exclusive(RECORDS / "results/archive_map.json", mapping)
    register(
        "supporting" if result["all_competence_passed"] else "unresolved",
        "Completed fixed-parameter encoding cross, exact conditional moments, 1000 empirical cohorts and paired support contrasts; no main-model promotion.",
    )
    return {
        "archives": len(mapping),
        "exact_byte_reconstruction": True,
        "promotion": False,
    }
