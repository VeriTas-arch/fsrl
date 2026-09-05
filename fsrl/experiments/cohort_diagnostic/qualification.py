"""Non-Liu forward parity and extraction of already exposed parent statistics."""

import subprocess
import sys

import numpy as np

from fsrl.experiments.minimal_learner.data import ModelBatch, generic_batch
from fsrl.experiments.minimal_learner.training import compiled, runtime
from fsrl.experiments.quantized_learner.analysis import readout
from fsrl.experiments.quantized_learner.encoding import encode_batch
from fsrl.experiments.quantized_learner.evaluation import load_model
from fsrl.experiments.quantized_learner.protocol import resolved_specification
from fsrl.experiments.quantized_learner.verification import reconstruct_codes
from fsrl.experiments.training_strategy.behavior import human_references
from fsrl.experiments.training_strategy.generic_validation import validation_episodes
from fsrl.experiments.training_strategy.locks import (
    reference,
    require_pushed_clean,
    verify_reference,
)
from fsrl.infra.provenance import load_json, tensor_hashes, write_json_exclusive
from fsrl.infra.run_manifest import ProspectiveRun
from fsrl.paths import REPO_ROOT

from .protocol import (
    PARENT_COMMIT,
    PARENT_RECORDS,
    RUN_ROOT,
    implementation_sources,
    load_parameters,
    specification,
)
from .statistics import cohort_record, extract_values

CPU_TEST = "tests.experiments.cohort_diagnostic.test_diagnostic"


def parent_point_parity() -> dict:
    path = PARENT_RECORDS / "results/quantized_relational_learner_v1.json"
    parent = load_json(verify_reference(reference(path), commit=PARENT_COMMIT))
    refs = human_references(resolved_specification())
    errors = {}
    for identity, fit in parent["fits"].items():
        behavior = load_json(
            verify_reference(fit["sampled_behavior"], commit=PARENT_COMMIT)
        )
        point = cohort_record(behavior, refs)
        expected = extract_values(fit["behavior"]["metrics"])
        difference = max(
            abs(point["values"][key] - value) for key, value in expected.items()
        )
        if difference > 1e-12 or point["flags"] != fit["behavior"]["flags"]:
            raise RuntimeError("original cohort point/flag extraction differs")
        if point["eligible_subjects"] != fit["behavior"]["eligible_subjects"]:
            raise RuntimeError("original eligibility denominator differs")
        if (
            point["analysis_subjects"]
            != fit["behavior"]["analysis_subjects_excluding_correct_rankers"]
        ):
            raise RuntimeError("original analysis denominator differs")
        errors[identity] = difference
    return errors


def fixture() -> tuple:
    spec = resolved_specification()
    seed = specification()["execution"]["qualification_seed"]
    spec["evaluation"]["generic"].update(episodes=256, rng_seed=seed)
    episodes = [
        row for row in validation_episodes(spec) if len(row.support_trials) == 32
    ]
    batch = generic_batch(tuple(episodes[i % len(episodes)] for i in range(77)))
    arrays = batch.arrays
    query = arrays["query_cues"]
    width = query.shape[-1] // 2
    reverse = np.concatenate((query[..., width:], query[..., :width]), axis=-1)
    batch = ModelBatch(
        {
            **arrays,
            "query_cues": np.concatenate((query, reverse), axis=1),
            "query_pairs": np.concatenate(
                (arrays["query_pairs"], arrays["query_pairs"][..., ::-1]), axis=1
            ),
            "targets": np.concatenate(
                (arrays["targets"], 1 - arrays["targets"]), axis=1
            ),
            "learned": np.tile(arrays["learned"], (1, 2)),
        }
    )
    uniforms = np.random.default_rng(seed).random(batch.arrays["signed"].shape)
    codebook = [-1, -1 / 3, 1 / 3, 1]
    encoded, witness = encode_batch(batch, "resampled", uniforms, codebook)
    independent, indices, orientation = reconstruct_codes(
        batch, uniforms, "resampled", codebook
    )
    for key, value in (
        ("internal_signed", independent),
        ("code_indices", indices),
        ("orientation", orientation),
    ):
        np.testing.assert_array_equal(witness[key], value)
    return encoded, spec


def numerical_checks(config: dict, batch, spec) -> dict:
    model = load_model(config, spec)
    before = tensor_hashes(model)
    eager = readout(model, model, batch)
    runner = compiled(model)
    compiled_output = readout(model, runner, batch)
    errors = {}
    for key in ("w", "margins"):
        np.testing.assert_allclose(
            compiled_output[key], eager[key], atol=1e-5, rtol=1e-4
        )
        errors[key] = float(np.max(np.abs(compiled_output[key] - eager[key])))
    off = ModelBatch(
        {**batch.arrays, "retention": np.zeros_like(batch.arrays["retention"])}
    )
    for value in readout(model, runner, off).values():
        np.testing.assert_array_equal(value, 0)
    reversed_queries = ModelBatch(
        {**batch.arrays, "query_cues": batch.arrays["query_cues"][:, ::-1].copy()}
    )
    reversed_output = readout(model, runner, reversed_queries)
    np.testing.assert_array_equal(reversed_output["w"], compiled_output["w"])
    np.testing.assert_allclose(
        reversed_output["margins"],
        compiled_output["margins"][:, ::-1],
        atol=1e-5,
        rtol=1e-4,
    )
    if before != tensor_hashes(model) or any(
        p.requires_grad for p in model.parameters()
    ):
        raise RuntimeError("diagnostic model is not parameter-frozen")
    return {
        "passed": True,
        "eager_compiled_errors": errors,
        "reference_parity": True,
        "z_off": True,
        "query_no_write": True,
    }


def qualify(attempt: int) -> dict:
    commit = require_pushed_clean()
    execution = runtime()
    directory = RUN_ROOT / "qualification" / f"attempt-{attempt}"
    with ProspectiveRun.start(
        directory,
        workflow_id="resampled_cohort_diagnostic_v1",
        execution_id=f"qualification-{attempt}",
        producer={"module": __name__, "source_commit": commit},
        resolved_config=execution,
    ):
        completed = subprocess.run(
            [sys.executable, "-m", "unittest", CPU_TEST],
            check=False,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        transcript = completed.stdout + completed.stderr
        (directory / "tests.txt").open("x").write(transcript)
        if completed.returncode:
            raise RuntimeError("cohort diagnostic CPU qualification failed")
        batch, spec = fixture()
        checks = {
            key: numerical_checks(config, batch, spec)
            for key, config in load_parameters().items()
        }
        result = {
            "passed": True,
            "source_commit": commit,
            "sources": implementation_sources(),
            "runtime": execution,
            "new_liu_evaluated": False,
            "new_parameters_trained": False,
            "parent_point_errors": parent_point_parity(),
            "numerical": checks,
            "cpu_test_module": CPU_TEST,
            "cpu_transcript": transcript,
        }
        write_json_exclusive(directory / "qualification.json", result)
    return {"passed": True, "record": reference(directory / "qualification.json")}
