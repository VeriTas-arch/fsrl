"""Locked non-Liu recovery screen and observation-only numerical replay."""

from copy import deepcopy

import numpy as np

from fsrl.experiments.minimal_learner.data import generic_batch
from fsrl.experiments.minimal_learner.training import runtime
from fsrl.experiments.training_strategy.evaluation import json_ready, write_arrays
from fsrl.experiments.training_strategy.generic_validation import validation_episodes
from fsrl.experiments.training_strategy.locks import reference, verify_reference
from fsrl.infra.provenance import load_json, write_json_exclusive

from .evidence import RECOVERY_RESULT, SOURCE_LOCK, validate_source
from .protocol import RECORDS, resolved_specification, specification
from .recovery import decode_choices, generate_choices, recovery_summary
from .recovery_inputs import ObservedDesign


def execute_recovery() -> dict:
    source = validate_source()
    execution = runtime()
    candidate, spec = specification(), resolved_specification()
    settings = candidate["identifiability"]
    design_spec = deepcopy(spec)
    design_spec["evaluation"]["generic"].update(
        {
            "episodes": settings["episodes_per_generating_setting"],
            "rng_seed": settings["task_rng_seed"],
        }
    )
    episodes = validation_episodes(design_spec)
    simulation_rng = np.random.default_rng(settings["simulation_rng_seed"])
    integration_rng = np.random.default_rng(settings["integration_rng_seed"])
    options = {
        "temperature": spec["evaluation"]["generic"]["temperature"],
        "epsilon": spec["model"]["epsilon"],
    }
    observations, generation, likelihoods, fingerprints = {}, {}, [], []
    for index, episode in enumerate(episodes):
        design = ObservedDesign.from_batch(generic_batch((episode,)), 0)
        simulated = generate_choices(
            design,
            simulation_rng,
            settings,
            candidate["encoding"]["codebook"],
            **options,
        )
        values, hashes = decode_choices(
            design,
            simulated["left_counts"],
            integration_rng,
            settings,
            candidate["encoding"]["codebook"],
            **options,
        )
        observations.update(
            {f"{index}__{key}": value for key, value in vars(design).items()}
        )
        generation.update(
            {f"{index}__{key}": value for key, value in simulated.items()}
        )
        likelihoods.append(values)
        fingerprints.append(hashes)
        if (index + 1) % 8 == 0:
            print("Recovery designs", index + 1, "/", len(episodes), flush=True)
    destination = RECORDS / "results"
    destination.mkdir(parents=True, exist_ok=True)
    payloads = {
        "observations": observations,
        "generation": generation,
        "likelihoods": {"per_episode": np.asarray(likelihoods)},
    }
    files = {}
    for name, arrays in payloads.items():
        path = destination / f"recovery-{name}.npz"
        write_arrays(path, arrays)
        files[name] = reference(path)
    result = {
        "source_commit": source["source_commit"],
        "source_lock": reference(SOURCE_LOCK),
        "settings": settings,
        "runtime": execution,
        "options": options,
        "files": files,
        "nuisance_fingerprints": fingerprints,
        "summary": json_ready(recovery_summary(np.asarray(likelihoods))),
        "liu_evaluated": False,
    }
    verification = verify_recovery(result)
    result["verification"] = verification
    write_json_exclusive(RECOVERY_RESULT, result)
    return {"outcome": result["summary"]["outcome"], "verification": verification}


def verify_recovery(result: dict | None = None) -> dict:
    result = load_json(RECOVERY_RESULT) if result is None else result
    arrays = {}
    for name, ref in result["files"].items():
        with np.load(verify_reference(ref), allow_pickle=False) as saved:
            arrays[name] = {key: saved[key] for key in saved.files}
    rng = np.random.default_rng(result["settings"]["integration_rng_seed"])
    codebook = specification()["encoding"]["codebook"]
    errors = []
    for index in range(result["settings"]["episodes_per_generating_setting"]):
        design = ObservedDesign(
            **{
                key: arrays["observations"][f"{index}__{key}"]
                for key in ("support_cues", "signed", "query_cues")
            }
        )
        counts = arrays["generation"][f"{index}__left_counts"]
        values, hashes = decode_choices(
            design, counts, rng, result["settings"], codebook, **result["options"]
        )
        if hashes != result["nuisance_fingerprints"][index]:
            raise RuntimeError("independent hidden-state prior pool does not replay")
        expected = arrays["likelihoods"]["per_episode"][index]
        np.testing.assert_allclose(values, expected, atol=1e-9, rtol=1e-7)
        errors.append(float(np.max(np.abs(values - expected))))
        np.testing.assert_array_equal(
            counts, arrays["generation"][f"{index}__choices"].sum(axis=1)
        )
    if (
        json_ready(recovery_summary(arrays["likelihoods"]["per_episode"]))
        != result["summary"]
    ):
        raise RuntimeError("registered recovery decision does not reconstruct")
    return {
        "passed": True,
        "designs": len(errors),
        "max_log_likelihood_error": max(errors),
    }
