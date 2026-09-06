"""Write-once generic model recovery and rho selection."""

from __future__ import annotations

import numpy as np

from fsrl.experiments.cohort_diagnostic.statistics import wilson
from fsrl.experiments.training_strategy.locks import reference
from fsrl.infra.provenance import write_json_exclusive
from fsrl.infra.run_manifest import ProspectiveRun

from .encoding import draw_streams
from .inputs import load_group
from .protocol import (
    CODEBOOK,
    CONDITIONS,
    RECORDS,
    RECOVERY_DATASETS,
    RECOVERY_SEED_BASE,
    RHO_GRID,
    RUN_ROOT,
)
from .recovery import choices, common_log_likelihood, likelihoods

RESULT = RECORDS / "results/generic_recovery.json"
LOCK = RECORDS / "benchmarks/rho_lock.json"


def recover_dataset(groups: list, generator: str, rho: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    architecture_scores: dict[str, float] = dict.fromkeys(CONDITIONS, 0.0)
    rho_scores: dict[float, float] = dict.fromkeys(RHO_GRID, 0.0)
    for record in groups:
        batch, _ = load_group(record)
        observed, probability, keys = choices(
            batch, generator, rho, draw_streams(batch, rng), CODEBOOK
        )
        for condition, score in likelihoods(observed, probability, keys, rho).items():
            architecture_scores[condition] += score
        if generator != "independent":
            for candidate in RHO_GRID:
                rho_scores[candidate] += common_log_likelihood(
                    observed, probability, keys, generator, candidate
                )
    return {
        "architecture": max(architecture_scores, key=architecture_scores.__getitem__),
        "rho": (
            None
            if generator == "independent"
            else max(rho_scores, key=rho_scores.__getitem__)
        ),
    }


def summarize(rows: list[dict]) -> dict:
    candidates = {}
    selected = None
    for rho in RHO_GRID:
        generators = {}
        for generator in CONDITIONS:
            subset = [
                row
                for row in rows
                if row["generator"] == generator and row["generating_rho"] == rho
            ]
            architecture_successes = sum(
                row["architecture"] == generator for row in subset
            )
            architecture_interval = wilson(
                [row["architecture"] == generator for row in subset]
            )
            architecture_passed = (
                architecture_successes / RECOVERY_DATASETS >= 0.9
                and architecture_interval["lower"] > 0.84
            )
            rho_successes = (
                None
                if generator == "independent"
                else sum(row["rho"] == rho for row in subset)
            )
            rho_passed = (
                True
                if rho_successes is None
                else rho_successes / RECOVERY_DATASETS >= 0.8
            )
            generators[generator] = {
                "architecture_successes": architecture_successes,
                "architecture_rate": architecture_successes / RECOVERY_DATASETS,
                "architecture_interval": architecture_interval,
                "architecture_passed": architecture_passed,
                "rho_successes": rho_successes,
                "rho_rate": (
                    None if rho_successes is None else rho_successes / RECOVERY_DATASETS
                ),
                "rho_passed": rho_passed,
            }
        passed = all(
            row["architecture_passed"] and row["rho_passed"]
            for row in generators.values()
        )
        candidates[str(rho)] = {"generators": generators, "passed": passed}
        if selected is None and passed:
            selected = rho
    return {
        "candidates": candidates,
        "selected_rho": selected,
        "passed": selected is not None,
        "liu_evaluated": False,
        "parameters_trained": False,
    }


def evaluate_recovery() -> dict:
    from .evidence import validate_source

    source = validate_source()
    groups = list(source["generic_groups"].values())
    directory = RUN_ROOT / "generic-recovery"
    with ProspectiveRun.start(
        directory,
        workflow_id="common_encoding_state_v1",
        execution_id="generic-model-recovery",
        producer={"module": __name__, "source_commit": source["source_commit"]},
        resolved_config={"rho_grid": RHO_GRID, "datasets": RECOVERY_DATASETS},
    ):
        rows = []
        for rho_index, rho in enumerate(RHO_GRID):
            for generator_index, generator in enumerate(CONDITIONS):
                for dataset in range(RECOVERY_DATASETS):
                    seed = (
                        RECOVERY_SEED_BASE
                        + 100_000 * rho_index
                        + 10_000 * generator_index
                        + dataset
                    )
                    rows.append(
                        {
                            "generator": generator,
                            "generating_rho": rho,
                            "dataset": dataset,
                            "seed": seed,
                            **recover_dataset(groups, generator, rho, seed),
                        }
                    )
                print(f"Recovered {generator} at rho={rho}", flush=True)
        result = {
            "experiment_id": "common_encoding_state_v1",
            "source_commit": source["source_commit"],
            "qualification": source["qualification"],
            "rows": rows,
            "summary": summarize(rows),
        }
        write_json_exclusive(directory / "result.json", result)
    return result["summary"]


def lock_recovery(directory) -> dict:
    from fsrl.experiments.minimal_learner.locks import validate_complete
    from fsrl.experiments.training_strategy.locks import require_pushed_clean
    from fsrl.infra.provenance import load_json

    from .evidence import validate_source

    commit = require_pushed_clean()
    source = validate_source()
    validate_complete(directory)
    result = load_json(directory / "result.json")
    expected = summarize(result["rows"])
    if (
        result["summary"] != expected
        or result["source_commit"] != source["source_commit"]
        or result["qualification"] != source["qualification"]
    ):
        raise RuntimeError("generic recovery summary does not reconstruct")
    write_json_exclusive(RESULT, result)
    lock = {
        "source_commit": source["source_commit"],
        "recovery_commit": commit,
        "result": reference(RESULT),
        "selected_rho": expected["selected_rho"],
        "passed": expected["passed"],
        "liu_evaluated": False,
        "parameters_trained": False,
    }
    write_json_exclusive(LOCK, lock)
    if not lock["passed"]:
        raise RuntimeError("no rho passed generic recovery; stop before training")
    return lock
