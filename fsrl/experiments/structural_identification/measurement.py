"""Source-locked human measurement bridge; never fits a candidate to humans."""

import csv
from itertools import combinations

import numpy as np

from fsrl.analysis.behavioral import kendall_tau_positions
from fsrl.experiments.confirmation.reproduction_map import (
    endpoint_statistics,
    position_profile,
)
from fsrl.experiments.human.benchmark import LIU_DATASET_FILES, LIU_DATASET_ROOT
from fsrl.experiments.training_strategy.behavior import human_references
from fsrl.infra.provenance import file_sha256, load_json, write_json_exclusive
from fsrl.tasks.protocol_catalog import load_registered_protocol

from .observation import CLASSES, observe, record, replicated_cycles
from .protocol import RECORDS, RUN_ROOT, model_specification

REFERENCE = RECORDS / "results/common_human_reference.json"


def human_choices() -> np.ndarray:
    pairs = {pair: i for i, pair in enumerate(combinations(range(8), 2))}
    cohorts = []
    for name in ("preregistered", "replication"):
        entry = LIU_DATASET_FILES[name]
        path = LIU_DATASET_ROOT / entry["path"]
        if file_sha256(path) != entry["sha256"]:
            raise RuntimeError("immutable human source changed")
        with path.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        ids = sorted({int(row["id"]) for row in rows})
        index = {value: i for i, value in enumerate(ids)}
        choices = np.zeros((len(ids), 10, 28), dtype=bool)
        seen = np.zeros_like(choices, dtype=int)
        for row in rows:
            first, second = int(row["film_index_1"]) - 1, int(row["film_index_2"]) - 1
            pair = min(first, second), max(first, second)
            key = index[int(row["id"])], int(row["block"]) - 1, pairs[pair]
            choices[key] = int(row["film_choose_index"]) - 1 == pair[0]
            seen[key] += 1
        if not np.all(seen == 1) or len(ids) != entry["participants"]:
            raise RuntimeError("human subject/block/pair coverage differs")
        cohorts.append(choices)
    return np.concatenate(cohorts)


def bootstrap_reference(behavior: dict) -> dict:
    """Whole-participant bootstrap of continuous statistics, preserving exclusions."""
    rows = behavior["subjects"]
    n = len(rows)
    counts = np.random.default_rng(1300001).multinomial(
        n, np.full(n, 1 / n), size=10000
    )
    eligible = np.asarray([row["overall_accuracy"] >= 0.5 for row in rows])
    analysis = eligible & np.asarray(
        [row["ranking_class"] != "correct" for row in rows]
    )
    eligible_counts = counts * eligible
    analysis_counts = counts * analysis
    denominators = eligible_counts.sum(axis=1), analysis_counts.sum(axis=1)
    if any(np.any(d == 0) for d in denominators):
        raise RuntimeError("undefined common-reference bootstrap cohort")
    values = {
        name: counts @ np.asarray([row[name] for row in rows]) / n
        for name in (
            "learned_accuracy",
            "nonlearned_accuracy",
            "symbolic_distance_slope",
        )
    }
    for name in CLASSES:
        values[name + "_proportion"] = (
            eligible_counts
            @ np.asarray([row["ranking_class"] == name for row in rows])
            / denominators[0]
        )
    values["stable_error_80_analysis_proportion"] = (
        analysis_counts
        @ np.asarray([row["stable_error_pair_counts"]["80"] > 0 for row in rows])
        / denominators[1]
    )
    serial = []
    ranks = []
    for row in rows:
        point_rows = [
            {**pair, "value": value}
            for pair, value in zip(behavior["pairs"], row["pair_accuracy"], strict=True)
        ]
        serial.append(
            endpoint_statistics(position_profile(point_rows, "value"))[
                "mean_endpoint_contrast"
            ]
        )
        rank = np.empty(8, dtype=int)
        rank[row["subjective_order_high_to_low"]] = np.arange(8)
        ranks.append(rank)
    values["serial"] = counts @ np.asarray(serial) / n
    tau = np.asarray([[kendall_tau_positions(a, b) for b in ranks] for a in ranks])
    k = denominators[1]
    if np.any(k < 2):
        raise RuntimeError("undefined ranking-diversity reference")
    values["tau"] = (
        np.einsum("bi,ij,bj->b", analysis_counts, tau, analysis_counts) - k
    ) / (k * (k - 1))
    intervals = {
        name: dict(
            zip(
                ("lower", "upper"),
                np.quantile(value, [0.025, 0.975]).tolist(),
                strict=True,
            )
        )
        for name, value in values.items()
    }
    intervals["correct_ranker_proportion"] = intervals.pop("correct_proportion")
    return {
        "serial": intervals.pop("serial"),
        "tau": intervals.pop("tau"),
        "intervals": intervals,
    }


def run_measurement() -> dict:
    from .execution import validate_source

    validate_source()
    if REFERENCE.exists():
        return load_json(REFERENCE)
    protocol = load_registered_protocol("liu_v2")
    choices = human_choices()
    behavior = observe(choices, protocol)
    refs = bootstrap_reference(behavior)
    published_path = LIU_DATASET_ROOT / LIU_DATASET_FILES["figure3b"]["path"]
    with published_path.open(newline="", encoding="utf-8-sig") as handle:
        published = [row["Group"] for row in csv.DictReader(handle)]
    names = {
        "Correct rank": "correct",
        "Self_consistent": "self_consistent_incorrect",
        "Self_inconsistent": "self_inconsistent",
    }
    exceptions = [
        {
            "combined_id": i + 1,
            "common": row["ranking_class"],
            "published": names[released],
            "ties": row["majority_ties"],
        }
        for i, (row, released) in enumerate(
            zip(behavior["subjects"], published, strict=True)
        )
        if row["ranking_class"] != names[released]
    ]
    inventory = load_json(RUN_ROOT / "source_audit/osf_inventory.json")
    executable = [
        row
        for row in inventory["entries"]
        if str(row["name"]).endswith((".py", ".m", ".R", ".ipynb", ".zip"))
    ]
    legacy = human_references(model_specification())
    result = {
        "observer": "strict_observed_v1",
        "references": refs,
        "human_record": record(choices, protocol, refs),
        "published_class_exceptions": exceptions,
        "legacy_reference": legacy,
        "replicated_cycle_counts": replicated_cycles(choices).tolist(),
        "published_correspondence": "unresolved",
        "source_inventory_entries": len(inventory["entries"]),
        "executable_source_candidates": executable,
        "scope": "Common observed-choice estimand; published classes remain a distinct frozen target. No model parameters fitted.",
    }
    write_json_exclusive(REFERENCE, result)
    with (RUN_ROOT / "human_choices.npz").open("xb") as handle:
        np.savez_compressed(handle, choices=choices)
    return result
