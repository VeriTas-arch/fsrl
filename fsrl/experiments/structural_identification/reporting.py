"""Complete per-fit decisions, compact archival, and the component-retention table."""

import json
import shutil

import numpy as np

from fsrl.experiments.cohort_diagnostic.statistics import reference_intervals, wilson
from fsrl.experiments.minimal_learner.locks import validate_complete
from fsrl.experiments.training_strategy.locks import reference
from fsrl.infra.provenance import load_json, write_json_exclusive
from fsrl.paths import REPO_ROOT

from .evaluation import identity, mean_interval
from .execution import fitted_parameters
from .inputs import load_inputs, save_arrays
from .measurement import REFERENCE
from .observation import replicated_cycles
from .protocol import RECORDS, RUN_ROOT, SCHEDULES, SEEDS, STRUCTURES, specification


def flag_pass(record, kind):
    return all(row[kind] for row in record["flags"].values())


def profile(rows, refs, seed):
    intervals = reference_intervals(refs)
    estimates = {
        name: mean_interval([row["values"][name] for row in rows], seed)
        for name in intervals
    }
    stable = all(
        value["lower"] is not None
        and value["lower"] >= intervals[name]["lower"]
        and value["upper"] <= intervals[name]["upper"]
        for name, value in estimates.items()
    )
    witnesses = all(
        flag_pass(row, "qualitative") and flag_pass(row, "calibration")
        for row in rows[:3]
    )
    return {
        "estimates": estimates,
        "continuous_stability": stable,
        "three_witnesses": witnesses,
        "qualitative_rate": wilson([flag_pass(row, "qualitative") for row in rows]),
        "joint_rate": wilson(
            [
                flag_pass(row, "qualitative") and flag_pass(row, "calibration")
                for row in rows
            ]
        ),
        "qualified": stable and witnesses,
    }


def scalar_power():
    rng = np.random.default_rng(1360001)
    results = {}
    for probability in (0.7, 0.9):
        choices = np.ones((10000, 10, 28), dtype=bool)
        # Canonical pair indices: (0,1)=0, (0,2)=1, (1,2)=7.
        choices[:, :, [0, 1, 7]] = rng.random((10000, 10, 3)) < np.asarray(
            [probability, 1 - probability, probability]
        )
        results[str(probability)] = wilson(replicated_cycles(choices) > 0)
    return results


def manipulation_summary(fits):
    results = {}
    for row in fits:
        name = identity(row)
        directory = RUN_ROOT / "manipulations" / name
        validate_complete(directory)
        with np.load(directory / "outputs.npz", allow_pickle=False) as arrays:
            per_condition = {}
            individual = {}
            for condition in specification()["manipulations"]["conditions"]:
                correct = arrays[f"{condition}__sampled_correct"]
                masks = {
                    key: arrays[f"{condition}__{key}"] for key in ("bridge", "cross")
                }
                individual[condition] = {
                    key: (correct * mask).sum(axis=1) / mask.sum(axis=1)
                    for key, mask in masks.items()
                }
                classes = arrays[f"{condition}__classes"]
                per_condition[condition] = {
                    **{
                        key: mean_interval(values, 7900001 + row["seed"])
                        for key, values in individual[condition].items()
                    },
                    "classes": {
                        key: float(np.mean(classes == key))
                        for key in (
                            "correct",
                            "self_consistent_incorrect",
                            "self_inconsistent",
                        )
                    },
                }
            contrasts = {
                condition: {
                    key: mean_interval(
                        value - individual["balanced_K4"][key], 7900001 + row["seed"]
                    )
                    for key, value in individual[condition].items()
                }
                for condition in individual
                if condition != "balanced_K4"
            }
            results[name] = {
                "conditions": per_condition,
                "paired_against_K4": contrasts,
            }
    return results


def component_decisions(fits, profiles, rank_costs):
    qualified = []
    for schedule in SCHEDULES:
        for structure in STRUCTURES:
            names = [f"{seed}-{structure}-{schedule}" for seed in SEEDS]
            if all(profiles[name]["development_qualified"] for name in names):
                qualified.append((structure, schedule))
    comparisons = []
    for full in qualified:
        for simple in qualified:
            full_bits = (int(full[0][1]), int(full[0][2]), int(full[1] == "decay"))
            simple_bits = (
                int(simple[0][1]),
                int(simple[0][2]),
                int(simple[1] == "decay"),
            )
            if full == simple or not all(
                s <= f for s, f in zip(simple_bits, full_bits, strict=True)
            ):
                continue
            estimates = {
                str(seed): mean_interval(
                    rank_costs[f"{seed}-{simple[0]}-{simple[1]}"]
                    - rank_costs[f"{seed}-{full[0]}-{full[1]}"],
                    7900001 + seed,
                )
                for seed in SEEDS
            }
            comparisons.append(
                {
                    "fuller": list(full),
                    "simpler": list(simple),
                    "rank_TV_cost": estimates,
                    "removal_supported": all(
                        value["upper"] is not None and value["upper"] <= 0.01
                        for value in estimates.values()
                    ),
                }
            )
    dominated = {
        tuple(row["fuller"]) for row in comparisons if row["removal_supported"]
    }
    return {
        "qualified_recipes": [list(row) for row in qualified],
        "deletions": comparisons,
        "minimal_qualified_set": [
            list(row) for row in qualified if row not in dominated
        ],
        "scope": "Within the eight frozen recipes and their generic training procedure; no universal structural necessity.",
    }


def summarize() -> dict:
    fits = fitted_parameters()
    inputs = load_inputs()
    human = load_json(REFERENCE)
    cohort_rows = []
    for index in range(len(inputs["liu"])):
        directory = RUN_ROOT / "liu" / f"cohort-{index:03}"
        validate_complete(directory)
        cohort_rows.append(load_json(directory / "result.json"))
    profiles, rank_costs = {}, {}
    keys = ("correct_ranker", "self_consistent_incorrect", "self_inconsistent")
    target = np.asarray([human["human_record"]["values"][key] for key in keys])
    for row in fits:
        name = identity(row)
        generic_directory = RUN_ROOT / "generic" / name
        validate_complete(generic_directory)
        generic = load_json(generic_directory / "result.json")
        values = [cohort["fits"][name] for cohort in cohort_rows]
        common = profile(
            [value["common"] for value in values],
            human["references"],
            7900001 + row["seed"],
        )
        legacy = profile(
            [value["legacy"] for value in values],
            human["legacy_reference"],
            7900001 + row["seed"],
        )
        rank_costs[name] = np.asarray(
            [
                0.5
                * sum(
                    abs(value["common"]["values"][key] - target[i])
                    for i, key in enumerate(keys)
                )
                for value in values
            ]
        )
        cycles = np.asarray([value["replicated_cycle_prevalence"] for value in values])
        profiles[name] = {
            "parameters": row,
            "generic": generic,
            "common": common,
            "legacy": legacy,
            "rank_TV": mean_interval(rank_costs[name], 7900001 + row["seed"]),
            "replicated_cycle_predictive_interval": np.quantile(
                cycles, [0.025, 0.975]
            ).tolist(),
            "replicated_cycle_mean": float(cycles.mean()),
            "development_qualified": bool(
                generic["competence"]
                and generic["binding_pass"]
                and common["qualified"]
            ),
        }
    recovery = {}
    for name in inputs["recovery"]:
        directory = RUN_ROOT / "recovery" / name
        validate_complete(directory)
        recovery[name] = load_json(directory / "result.json")
    decisions = component_decisions(fits, profiles, rank_costs)
    outcome = (
        "development_qualified_measurement_unresolved"
        if decisions["qualified_recipes"]
        else "no_fully_sufficient_recipe"
    )
    result = {
        "experiment_id": "structural_identification_v1",
        "outcome": outcome,
        "measurement": human,
        "fits": profiles,
        "component_decisions": decisions,
        "recovery": recovery,
        "manipulations": manipulation_summary(fits),
        "scalar_diagnostic": {
            "human_replicated_cycle_prevalence": float(
                np.mean(np.asarray(human["replicated_cycle_counts"]) > 0)
            ),
            "power": scalar_power(),
        },
        "promotion": False,
        "boundary": "Development comparison only. Legacy/source correspondence unresolved; no tuning, extra cohorts or promotion after this result.",
    }
    write_json_exclusive(RECORDS / "results/structural_identification_v1.json", result)
    return result


def report_text(result):
    lines = [
        "# Minimal sufficient structure identification",
        "",
        f"Registered outcome: `{result['outcome']}`.",
        "",
        "Three paired training streams, all eight recipes, and 100 separate 77-subject cohorts per fit. No network pooling or Liu fitting.",
        "",
        "## Complete comparison",
        "",
        "| Fit | eta0 | gain | Generic | Common stability | Common witnesses | Legacy stability | Qualified |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, row in result["fits"].items():
        p = row["parameters"]
        lines.append(
            f"| {name} | {p['eta0']:.6f} | {p['gamma_G']:.6f} | {row['generic']['competence']} | {row['common']['continuous_stability']} | {row['common']['three_witnesses']} | {row['legacy']['continuous_stability']} | {row['development_qualified']} |"
        )
    lines += [
        "",
        "## Measurement and claim boundary",
        "",
        "The common observer uses strict observed majorities, an explicit incorrect-direction convention for ties, and uniform Beta endpoint clipping. Original published Figure 3b correspondence remains unresolved; old results and criteria remain unchanged.",
        "",
        "The canonical JSON reports every continuous endpoint, uncertainty interval, witness/whole-cohort outcome, legacy bridge, recovery matrix, and paired manipulation contrast.",
        "",
        "## Component retention",
        "",
        "```json",
        json.dumps(result["component_decisions"], indent=2),
        "```",
        "",
        "No main model is promoted. The outcome constrains these exact recipes, not every possible score, attention, precision or learning-schedule theory.",
        "",
    ]
    return "\n".join(lines)


def archive_group(paths, destination):
    arrays = {}
    for index, path in enumerate(paths):
        with np.load(path, allow_pickle=False) as saved:
            arrays.update(
                {f"member{index:03}__{key}": saved[key] for key in saved.files}
            )
    return save_arrays(destination, **arrays)


def publish() -> dict:
    result_path = RECORDS / "results/structural_identification_v1.json"
    result = load_json(result_path) if result_path.exists() else summarize()
    report = RECORDS / "reports/structural_identification_v1.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("x") as handle:
        handle.write(report_text(result))
    snapshots = {
        "source_lock.json": RUN_ROOT / "source_lock.json",
        "input_lock.json": RUN_ROOT / "input_lock.json",
        "fit_lock.json": RUN_ROOT / "fit_lock.json",
        "qualification.json": RUN_ROOT / "qualification.json",
        "osf_inventory.json": RUN_ROOT / "source_audit/osf_inventory.json",
    }
    for name, source in snapshots.items():
        destination = RECORDS / "benchmarks" / name
        with destination.open("xb") as out, source.open("rb") as src:
            shutil.copyfileobj(src, out)
    # Shard portable outputs and inputs; retain exact count/margin arrays, never pickle.
    groups = {
        "generic": sorted((RUN_ROOT / "generic").glob("*/outputs.npz")),
        "recovery": sorted((RUN_ROOT / "recovery").glob("*/outputs.npz")),
        "manipulations": sorted((RUN_ROOT / "manipulations").glob("*/outputs.npz")),
        "liu": sorted((RUN_ROOT / "liu").glob("*/outputs.npz")),
        "inputs": sorted((RUN_ROOT / "inputs").glob("*.npz")),
    }
    archive_map = {}
    for family, paths in groups.items():
        size = 1 if family == "recovery" else 5
        for start in range(0, len(paths), size):
            subset = paths[start : start + size]
            destination = RECORDS / "results" / f"{family}-{start:03}.npz"
            archive_group(subset, destination)
            if destination.stat().st_size > 5000000:
                raise RuntimeError("archive exceeds inline review threshold")
            archive_map[destination.name] = [reference(path) for path in subset]
    write_json_exclusive(RECORDS / "results/archive_map.json", archive_map)
    register_records(result)
    return {
        "outcome": result["outcome"],
        "promotion": False,
        "result": str(result_path.relative_to(REPO_ROOT)),
    }


def register_records(result):
    manifest = RECORDS.parent / "study.toml"
    header = manifest.read_text().split("[[records]]", 1)[0]
    header = header.replace('status = "frozen_contract"', 'status = "unresolved"')
    header = header.replace(
        'finding = "Prospective measurement bridge and matched structural-deletion study; no candidate outcome is available."',
        f'finding = "Completed all 24 paired fits, common and legacy observation profiles, observable recovery and support manipulations; outcome {result["outcome"]}."',
    )
    blocks = []
    for path in sorted(RECORDS.rglob("*")):
        if not path.is_file():
            continue
        meta = reference(path)
        relative = path.relative_to(RECORDS.parent).as_posix()
        role = "supporting_artifact"
        if path.suffix == ".md":
            role = "report"
        elif path.name == "structural_identification_v1.json":
            role = (
                "registered_contract"
                if path.parent.name == "benchmarks"
                else "frozen_result"
            )
        blocks.append(
            f'[[records]]\npath = "{relative}"\nlegacy_path = "{meta["path"]}"\norigin = "native"\nrole = "{role}"\nsha256 = "{meta["sha256"]}"\nbytes = {meta["bytes"]}\nsource_ref = "sha256:{meta["sha256"]}"\n'
        )
    manifest.write_text(header + "\n".join(blocks))
