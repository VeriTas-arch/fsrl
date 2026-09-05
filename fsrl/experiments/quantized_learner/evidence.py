"""Write-once source, recovery and all-nine-artifact execution barriers."""

import hashlib
import json

import numpy as np
import torch

from fsrl.experiments.minimal_learner.locks import inputs as parent_inputs
from fsrl.experiments.minimal_learner.locks import validate_complete
from fsrl.experiments.training_strategy.evaluation import json_ready
from fsrl.experiments.training_strategy.locks import (
    reference,
    require_pushed_clean,
    verify_reference,
)
from fsrl.infra.provenance import load_json, tensor_hashes, write_json_exclusive
from fsrl.paths import REPO_ROOT

from .inputs import INPUT_MANIFEST, save_evaluation_inputs
from .protocol import (
    PROTOCOL,
    PROTOCOL_COMMIT,
    PROTOCOL_HASH,
    RECORDS,
    make_model,
    resolved_specification,
    run_directory,
    specification,
)
from .qualification import CPU_TEST_LOG, CPU_TESTS, sources
from .recovery import recovery_summary

SOURCE_LOCK = RECORDS / "benchmarks/source_lock.storage_v2.json"
ARTIFACT_LOCK = RECORDS / "benchmarks/artifact_lock.json"
QUALIFICATION = RECORDS / "benchmarks/qualification.storage_v2.json"
STORAGE_REPAIR = RECORDS / "benchmarks/storage_repair.json"
RECOVERY_RESULT = RECORDS / "results/recovery.json"


def qualification_keys() -> set[str]:
    suffixes = {"margins", "w"}
    for step in range(3):
        suffixes.update(f"step-{step}/output-{i}" for i in range(5))
        suffixes.add(f"step-{step}/loss")
        for parameter in ("raw_eta", "raw_global_gain"):
            suffixes.update(
                f"step-{step}/{kind}-{parameter}"
                for kind in (
                    "gradient",
                    "nonzero-gradient",
                    "updated",
                    "actual-update",
                    "counter",
                )
            )
            suffixes.update(
                f"step-{step}/adam-{parameter}-{kind}"
                for kind in ("step", "exp_avg", "exp_avg_sq")
            )
    return {"cpu-invariants"} | {
        f"{length}/{condition}/{suffix}"
        for length in (28, 32, 36, 40)
        for condition in specification()["seeds"]["conditions"]
        for suffix in suffixes
    }


def validate_qualification(record: dict) -> None:
    expected = {
        "passed": True,
        "liu_evaluated": False,
        "protocol_sha256": PROTOCOL_HASH,
        "seed": specification()["integrity"]["qualification_seed"],
        "sources": sources(),
        "cpu_test_modules": list(CPU_TESTS),
    }
    if any(record.get(key) != value for key, value in expected.items()):
        raise RuntimeError("qualification identity/source differs")
    if set(record["checks"]) != qualification_keys() or not all(
        row["passed"] is True for row in record["checks"].values()
    ):
        raise RuntimeError("qualification lacks a required numerical check")
    runtime = record["runtime"]
    flags = {
        "compiler_threads": 1,
        "torch_intraop_threads": 1,
        "torch_interop_threads": 1,
        "blas_thread_limit": 1,
        "cuda_available": True,
        "matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
    }
    profile = {
        "device": "cuda",
        "compile": True,
        "compile_fullgraph": True,
        "compile_backend": "inductor",
        "compile_mode": "default",
    }
    if any(runtime.get(k) != v for k, v in flags.items()) or any(
        runtime["profile"].get(k) != v for k, v in profile.items()
    ):
        raise RuntimeError("qualification runtime differs from the frozen CUDA profile")


def scientific_inputs() -> list[dict]:
    spec = specification()
    records = {row["path"]: row for row in parent_inputs()}
    for path in (PROTOCOL, REPO_ROOT / spec["admission_protocol"], STORAGE_REPAIR):
        records[path.relative_to(REPO_ROOT).as_posix()] = reference(path)
    return [records[key] for key in sorted(records)]


def lock_source(qualification_directory) -> dict:
    commit = require_pushed_clean()
    validate_complete(qualification_directory)
    record = load_json(qualification_directory / "qualification.json")
    validate_qualification(record)
    for row in sources() + scientific_inputs():
        verify_reference(row, commit=commit)
    if record["source_commit"] != commit:
        raise RuntimeError("qualify the final committed implementation before locking")
    if INPUT_MANIFEST.exists():
        verify_reference(reference(INPUT_MANIFEST), commit=commit)
        inputs = load_json(INPUT_MANIFEST)
        if (
            inputs["evaluation"] != resolved_specification()["evaluation"]
            or inputs["model_rollout_performed"]
        ):
            raise RuntimeError("previously locked evaluation inputs cannot change")
        for groups in inputs["cohorts"].values():
            for row in groups.values():
                verify_reference(row["arrays"], commit=commit)
    else:
        inputs = save_evaluation_inputs(resolved_specification())
    write_json_exclusive(
        QUALIFICATION,
        {
            **record,
            "cpu_test_transcript": (qualification_directory / CPU_TEST_LOG).read_text(),
        },
    )
    result = {
        "source_commit": commit,
        "protocol_commit": PROTOCOL_COMMIT,
        "protocol_sha256": PROTOCOL_HASH,
        "sources": sources(),
        "inputs": scientific_inputs(),
        "qualification": reference(QUALIFICATION),
        "evaluation_inputs": reference(INPUT_MANIFEST),
        "cohorts": inputs["cohorts"],
    }
    write_json_exclusive(SOURCE_LOCK, result)
    return result


def validate_source() -> dict:
    commit = require_pushed_clean()
    lock = load_json(SOURCE_LOCK)
    verify_reference(reference(SOURCE_LOCK), commit=commit)
    if (
        lock["sources"] != sources()
        or lock["inputs"] != scientific_inputs()
        or lock["protocol_sha256"] != PROTOCOL_HASH
    ):
        raise RuntimeError("source lock no longer describes this implementation")
    for row in lock["sources"] + lock["inputs"]:
        verify_reference(row, commit=lock["source_commit"])
    validate_qualification(
        load_json(verify_reference(lock["qualification"], commit=commit))
    )
    saved = load_json(verify_reference(lock["evaluation_inputs"], commit=commit))
    if saved["cohorts"] != lock["cohorts"] or saved["model_rollout_performed"]:
        raise RuntimeError("evaluation input admission differs")
    for groups in saved["cohorts"].values():
        for row in groups.values():
            verify_reference(row["arrays"], commit=commit)
    return lock


def validate_recovery() -> dict:
    source = validate_source()
    result = load_json(RECOVERY_RESULT)
    commit = require_pushed_clean()
    verify_reference(reference(RECOVERY_RESULT), commit=commit)
    if (
        result["source_lock"] != reference(SOURCE_LOCK)
        or result["source_commit"] != source["source_commit"]
        or result["settings"] != specification()["identifiability"]
    ):
        raise RuntimeError("recovery was not executed under this source lock")
    for row in result["files"].values():
        verify_reference(row, commit=commit)
    with np.load(
        verify_reference(result["files"]["likelihoods"]), allow_pickle=False
    ) as saved:
        values = saved["per_episode"]
    expected = (64, 2, 27, 3, 9)
    if (
        values.shape != expected
        or json_ready(recovery_summary(values)) != result["summary"]
    ):
        raise RuntimeError(
            "recovery requires the complete registered likelihood matrix"
        )
    return result


def validate_training_record(
    config: dict, logs: list, seed: int, condition: str
) -> None:
    spec = resolved_specification()
    settings = spec["optimization"]
    expected = {
        "seed": seed,
        "condition": condition,
        "protocol_sha256": PROTOCOL_HASH,
        "optimization": settings,
        "episodes": settings["total_episode_exposures"],
    }
    if any(config.get(key) != value for key, value in expected.items()):
        raise RuntimeError("training identity/exposure differs")
    if [row["step"] for row in logs] != list(range(settings["total_steps"])):
        raise RuntimeError("training does not contain every registered step")
    for channel in ("base", "uniform", "encoded"):
        digest = hashlib.sha256()
        for row in logs:
            digest.update(bytes.fromhex(row[f"{channel}_batch_sha256"]))
            if row[f"{channel}_stream_sha256"] != digest.hexdigest():
                raise RuntimeError("training hash chain differs")
        if config[f"{channel}_stream_sha256"] != digest.hexdigest():
            raise RuntimeError("final training stream differs")
    names = {"raw_eta", "raw_global_gain"}
    if set(config["optimizer_steps"]) != names or set(
        config["optimizer_steps"].values()
    ) != {settings["total_steps"]}:
        raise RuntimeError("both scalars must receive all registered Adam updates")
    model = make_model(spec)
    model.load_state_dict(
        {
            key: torch.tensor(value, dtype=torch.float32)
            for key, value in config["raw_parameters"].items()
        }
    )
    if tensor_hashes(model) != config["final_parameters"]:
        raise RuntimeError("archived scalar values do not reconstruct final tensors")


def paired_runs(runs: dict) -> None:
    seeds = specification()["seeds"]
    expected = {
        f"{seed}/{condition}"
        for seed in seeds["mandatory"]
        for condition in seeds["conditions"]
    }
    if set(runs) != expected:
        raise RuntimeError("all nine final fits are mandatory before evaluation")
    for seed in seeds["mandatory"]:
        rows = [
            runs[f"{seed}/{condition}"]["config"] for condition in seeds["conditions"]
        ]
        for key in (
            "base_stream_sha256",
            "uniform_stream_sha256",
            "initial_parameters",
        ):
            if any(row[key] != rows[0][key] for row in rows[1:]):
                raise RuntimeError(
                    "paired base data, uniforms or initial parameters differ"
                )


def lock_artifacts() -> dict:
    recovery = validate_recovery()
    runs = {}
    for seed in specification()["seeds"]["mandatory"]:
        for condition in specification()["seeds"]["conditions"]:
            directory = run_directory(seed, condition)
            validate_complete(directory)
            config = load_json(directory / "config.json")
            logs = [
                json.loads(line)
                for line in (directory / "train_log.jsonl").read_text().splitlines()
            ]
            validate_training_record(config, logs, seed, condition)
            if config["source_commit"] != recovery["source_commit"] or config[
                "recovery"
            ] != reference(RECOVERY_RESULT):
                raise RuntimeError("training did not follow the locked recovery")
            runs[f"{seed}/{condition}"] = {"config": config, "logs": logs}
    paired_runs(runs)
    files = {}
    for identity, record in runs.items():
        path = RECORDS / "results" / f"training-{identity.replace('/', '-')}.json"
        write_json_exclusive(path, record)
        files[identity] = reference(path)
    result = {
        "source_lock": reference(SOURCE_LOCK),
        "recovery": reference(RECOVERY_RESULT),
        "source_commit": recovery["source_commit"],
        "runs": files,
    }
    write_json_exclusive(ARTIFACT_LOCK, result)
    return result


def validate_artifacts() -> dict:
    source = validate_source()
    commit = require_pushed_clean()
    lock = load_json(ARTIFACT_LOCK)
    verify_reference(reference(ARTIFACT_LOCK), commit=commit)
    if (
        lock["source_lock"] != reference(SOURCE_LOCK)
        or lock["source_commit"] != source["source_commit"]
    ):
        raise RuntimeError("artifact/source lock mismatch")
    verify_reference(lock["recovery"], commit=commit)
    runs = {}
    for identity, ref in lock["runs"].items():
        seed, condition = identity.split("/")
        run = load_json(verify_reference(ref, commit=commit))
        validate_training_record(run["config"], run["logs"], int(seed), condition)
        if (
            run["config"]["recovery"] != lock["recovery"]
            or run["config"]["source_commit"] != source["source_commit"]
        ):
            raise RuntimeError(
                "archived training was not admitted by this source/recovery"
            )
        runs[identity] = run
    paired_runs(runs)
    return {**lock, "archives": runs, "cohorts": source["cohorts"]}
