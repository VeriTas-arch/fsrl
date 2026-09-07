"""Pair completed frozen-network cells and archive sufficient native evidence."""

import shutil

from fsrl.experiments.training_strategy.locks import reference
from fsrl.infra.provenance import write_json_exclusive

from . import locks, measurement
from .execution import arrays_at, completed
from .protocol import LOCK, PROTOCOL_SHA256, RECORDS, RUNS, parent, specification


def unpack(raw):
    values = {
        key.removeprefix("endpoints__"): value
        for key, value in raw.items()
        if key.startswith("endpoints__")
    }
    structures = {
        condition: {
            key.removeprefix(f"structure__{condition}__"): value
            for key, value in raw.items()
            if key.startswith(f"structure__{condition}__")
        }
        for condition in ("intact", "joint")
    }
    return values, structures


def direction(row):
    bounds = row["bootstrap"]
    if bounds["lower"] is not None and bounds["lower"] > 0:
        return "positive"
    if bounds["upper"] is not None and bounds["upper"] < 0:
        return "negative"
    return "unresolved"


def report():
    lock = locks.validate()
    spec = specification()
    cells, pairs = {}, {}
    for seed in spec["design"]["seeds"]:
        for arm in spec["design"]["arms"]:
            row = completed(RUNS / str(seed) / arm)
            if (
                row["seed"],
                row["arm"],
                row["source_commit"],
                row["protocol_sha256"],
            ) != (seed, arm, lock["source_commit"], PROTOCOL_SHA256):
                raise RuntimeError("cell identity mismatch")
            cells[f"{seed}/{arm}"] = row
    for seed in spec["design"]["seeds"]:
        a, sa = unpack(arrays_at(RUNS / str(seed) / "shared/raw.npz"))
        b, sb = unpack(arrays_at(RUNS / str(seed) / "cost/raw.npz"))
        pairs[str(seed)] = measurement.summarize_pair(a, b, sa, sb, seed)
    decisions = {}
    for seed in spec["design"]["seeds"]:
        cost = cells[f"{seed}/cost"]["endpoints"]
        pair = pairs[str(seed)]["cost_minus_shared"]
        decisions[str(seed)] = {
            name: {
                "cost_use": direction(cost[f"{name}_nonlearned_ce_benefit"]),
                "cost_minus_shared": direction(pair[f"{name}_nonlearned_ce_benefit"]),
            }
            for name in ("single", "joint")
        }
    destination = RECORDS / "artifacts"
    shutil.copytree(RUNS, destination)
    files = []
    for original in sorted(RUNS.rglob("*")):
        if original.is_file():
            copied = destination / original.relative_to(RUNS)
            if reference(original)["sha256"] != reference(copied)["sha256"]:
                raise RuntimeError("archive copy mismatch")
            files.append(reference(copied))
    result = {
        "study_id": spec["study_id"],
        "tier": spec["tier"],
        "protocol_sha256": PROTOCOL_SHA256,
        "source_lock": reference(LOCK),
        "parent_result": reference(parent() / "results/result.json"),
        "cells": cells,
        "pairs": pairs,
        "decisions": decisions,
        "artifacts": files,
        "scope": spec["decision"]["claim_boundary"],
        "not_run": [
            "training",
            "new_cohort",
            "bridge_panel",
            "N6/10",
            "equivalence_test",
            "main_model_admission",
        ],
    }
    write_json_exclusive(RECORDS / "results/result.json", result)
    return {
        "cells": len(cells),
        "pairs": len(pairs),
        "artifacts": len(files),
        "decisions": decisions,
    }
