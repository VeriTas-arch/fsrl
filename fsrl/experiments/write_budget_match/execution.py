"""Qualified source, budget-only calibration, then locked behavioral evaluation."""

import gc

import torch

from fsrl.experiments.evidence_routing.locks import require_clean
from fsrl.experiments.linear_modulation.model import load_model
from fsrl.experiments.modulation_schedule.evaluation import collect
from fsrl.experiments.training_strategy.evaluation import (
    flatten_arrays,
    json_ready,
    write_arrays,
)
from fsrl.experiments.training_strategy.locks import reference, verify_reference
from fsrl.experiments.write_cost.execution import configure_execution
from fsrl.experiments.write_cost.locks import completed
from fsrl.infra.provenance import load_json, tensor_hashes, write_json_exclusive
from fsrl.infra.run_manifest import ProspectiveRun
from fsrl.paths import REPO_ROOT

from .budgets import calibrate_network, inputs_for, measure, probes
from .protocol import (
    CALIBRATION,
    PROTOCOL,
    RECORDS,
    RUNS,
    SOURCE,
    prior_records,
    recipe,
    specification,
)


def sources():
    paths = list((REPO_ROOT / "fsrl").rglob("*.py"))
    paths += list((REPO_ROOT / "tests/experiments/write_budget_match").glob("*.py"))
    paths += [REPO_ROOT / "pyproject.toml", REPO_ROOT / ".envrc"]
    return [reference(p) for p in sorted(paths)]


def freeze():
    commit = require_clean()
    q = load_json(RECORDS / "benchmarks/qualification.json")
    assert q["passed"] and q["sources"] == sources()
    prior = prior_records()
    refs = sources() + [
        reference(PROTOCOL),
        reference(RECORDS / "benchmarks/qualification.json"),
    ]
    refs += list(prior["result"]["artifacts"].values())
    refs += list(prior["schedule_result"]["artifacts"].values())
    refs += list(specification()["parents"].values())
    refs += [
        v["file"]
        for p in prior["source"]["panels"].values()
        for v in p["inputs"].values()
    ]
    for ref in refs:
        verify_reference(ref, commit=commit)
    write_json_exclusive(
        SOURCE, {"source_commit": commit, "references": refs, "runtime": q["runtime"]}
    )
    return {"source_commit": commit, "references": len(refs)}


def validate(*, gpu=True, calibrated=False):
    commit = require_clean()
    lock = load_json(verify_reference(reference(SOURCE), commit=commit))
    for ref in lock["references"]:
        verify_reference(ref, commit=lock["source_commit"])
    if gpu:
        assert json_ready(configure_execution()) == lock["runtime"]
    if calibrated:
        c = load_json(verify_reference(reference(CALIBRATION), commit=commit))
        verify_reference(c["result"])
    return prior_records()


def calibrate():
    prior = validate()
    panel = {**prior["source"]["panels"]["1"], "id": 1}
    results = {}
    directory = RUNS / "calibration"
    with (
        ProspectiveRun.start(
            directory,
            workflow_id="write_budget_match_v1",
            execution_id="budget-only-calibration",
            producer={"source": reference(SOURCE)},
            resolved_config=specification()["solver"],
        ),
        torch.no_grad(),
    ):
        for seed in specification()["seeds"]:
            for training in ("clean", "noisy"):
                key = f"{seed}/{training}"
                net, _, seqs = load_model(
                    seed, prior["models"]["runs"][key]["files"], recipe()
                )
                before = tensor_hashes(net)
                kappa = prior["schedule_calibration"]["models"][key]["phase"]
                result, raw = calibrate_network(net, seqs, panel, kappa)
                assert tensor_hashes(net) == before
                results[key] = result
                write_arrays(directory / f"{seed}-{training}.npz", flatten_arrays(raw))
                print(
                    {
                        "model": key,
                        "matched": result["matched"],
                        "gain": result["gain"],
                        "ratios": result["ratios"],
                    },
                    flush=True,
                )
                del net, seqs
                gc.collect()
                torch.cuda.empty_cache()
            if seed == specification()["seeds"][0] and not all(
                v["matched"] for v in results.values()
            ):
                break
        result = {
            "models": results,
            "all_six_matched": len(results) == 6
            and all(v["matched"] for v in results.values()),
            "behavior_exposed": False,
        }
        write_json_exclusive(directory / "result.json", json_ready(result))
    return {
        "calibrated_models": len(results),
        "all_six_matched": result["all_six_matched"],
    }


def lock_calibration():
    validate(gpu=False)
    result = completed(RUNS / "calibration")
    value = {
        "source": reference(SOURCE),
        "result": reference(RUNS / "calibration/result.json"),
        **result,
    }
    write_json_exclusive(CALIBRATION, value)
    return {
        "locked_models": len(result["models"]),
        "evaluation_triggered": result["all_six_matched"],
    }


def evaluate():
    prior = validate(calibrated=True)
    calibration = load_json(CALIBRATION)
    assert calibration["all_six_matched"], (
        "Calibration gate failed; no behavioral evaluation."
    )
    for seed in specification()["seeds"]:
        for number in specification()["evaluation_panels"]:
            panel = {**prior["source"]["panels"][str(number)], "id": number}
            for cell, settings in prior["protocol"]["design"]["cells"].items():
                key = f"{seed}/{settings['training']}"
                files = prior["models"]["runs"][key]["files"]
                values = calibration["models"][key]["values"]
                directory = RUNS / "evaluation" / str(seed) / str(number) / cell
                with ProspectiveRun.start(
                    directory,
                    workflow_id="write_budget_match_v1",
                    execution_id=f"matched-{seed}-{number}-{cell}",
                    producer={"calibration": reference(CALIBRATION)},
                    resolved_config={"values": values},
                ):
                    result, raw, behavior = collect(
                        seed, settings["observation"], panel, files, values
                    )
                    write_arrays(directory / "raw.npz", flatten_arrays(raw))
                    write_json_exclusive(directory / "result.json", json_ready(result))
                    write_json_exclusive(
                        directory / "behavior.json", json_ready(behavior)
                    )
                print({"evaluated": f"{seed}/{number}/{cell}"}, flush=True)
                gc.collect()
                torch.cuda.empty_cache()
    return {"evaluation_units": 24}


def evaluation_budgets():
    prior = validate(calibrated=True)
    calibration = load_json(CALIBRATION)
    assert calibration["all_six_matched"]
    directory = RUNS / "evaluation_budgets"
    with (
        ProspectiveRun.start(
            directory,
            workflow_id="write_budget_match_v1",
            execution_id="three-condition-phase-budget-replay",
            producer={"calibration": reference(CALIBRATION)},
            resolved_config={"panels": specification()["evaluation_panels"]},
        ),
        torch.no_grad(),
    ):
        for key, model in prior["models"]["runs"].items():
            seed = int(key.split("/")[0])
            net, _, seqs = load_model(seed, model["files"], recipe())
            before = tensor_hashes(net)
            blocks = probes(net)
            kappa = prior["schedule_calibration"]["models"][key]["phase"]
            for number in specification()["evaluation_panels"]:
                panel = {**prior["source"]["panels"][str(number)], "id": number}
                for identity, cpu in inputs_for(panel, generic_only=False).items():
                    rows = {}
                    for mode, values in [
                        ("B", kappa),
                        ("S", kappa),
                        ("M", calibration["models"][key]["values"]),
                    ]:
                        rows[mode], _ = measure(
                            net,
                            seqs[1],
                            blocks["B" if mode == "B" else "M"],
                            cpu,
                            values,
                        )
                    path = directory / key / str(number) / (identity + ".npz")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    write_arrays(path, flatten_arrays(rows))
            assert tensor_hashes(net) == before
            print({"budget_replayed": key}, flush=True)
            del net, seqs, blocks
            gc.collect()
            torch.cuda.empty_cache()
        write_json_exclusive(
            directory / "result.json", {"passed": True, "batches": 360}
        )
    return {"batches": 360}
