"""Frozen Liu estimands on matched observed inputs and encoding replays."""

import numpy as np
import torch

from fsrl.experiments.evidence_routing.evaluation import margin_bundle
from fsrl.experiments.finite_state.liu import primary_analysis, primary_rollouts
from fsrl.experiments.finite_state.model import rollout
from fsrl.experiments.memory_structure.inputs import size_protocol
from fsrl.experiments.training_strategy.estimands import estimate
from fsrl.experiments.training_strategy.evaluation import (
    flatten_arrays,
    json_ready,
    write_arrays,
)
from fsrl.experiments.write_cost.inputs import load_input
from fsrl.experiments.write_cost.locks import completed
from fsrl.infra.provenance import write_json_exclusive
from fsrl.infra.run_manifest import ProspectiveRun

from .evaluation import loaded_model, observed
from .protocol import PROTOCOL_SHA256, RUNS, specification


def replay_summary(values, spec, seed):
    arrays = {}
    summary = {}
    for route, stack in values.items():
        stack = np.stack(stack)
        within = stack.var(0, ddof=1).mean(1)
        arrays[route] = {"margins": stack, "within_history_variance": within}
        summary[route] = {
            "within_history_variance": estimate(
                within,
                seed=spec["statistics"]["seed_offset"] + seed,
                statistics=spec["statistics"],
            ),
            "primary_replay": 0,
            "replays": len(stack),
        }
    return summary, arrays


def liu_run(seed, arm, source):
    directory = RUNS / "liu" / str(seed) / arm
    if directory.exists():
        return completed(directory)
    spec = specification()
    backbone, local, seqs = loaded_model(seed, arm, source)
    base = load_input(source["inputs"]["liu-8"])
    cpu = observed(base, arm, spec)
    protocol = size_protocol(spec, 8)
    generic = completed(RUNS / "generic/test" / str(seed) / arm)
    identity = {"seed": seed, "arm": arm, "protocol_sha256": PROTOCOL_SHA256}
    with (
        ProspectiveRun.start(
            directory,
            workflow_id="observation_uncertainty_v1",
            execution_id=f"liu-{seed}-{arm}",
            producer=identity,
            resolved_config={
                "evaluation": spec["evaluation"]["liu"],
                "observation": spec["observation"],
            },
        ),
        torch.no_grad(),
    ):
        raw = primary_rollouts(backbone, local, seqs, None, cpu, protocol, spec)
        summary, analysis, sampled = primary_analysis(
            raw, cpu, protocol, spec, seed, generic
        )
        values = {
            "full": [raw["bundles"]["intact"]["logits"]],
            "global": [raw["bundles"]["local_off"]["logits"]],
        }
        for replay in (1, 2):
            batch = observed(base, arm, spec, replay).to("cuda")
            result, _, _ = rollout(backbone, local, seqs, batch, None, 0)
            for route, logits in (
                ("full", result.logits),
                ("global", result.global_logits),
            ):
                values[route].append(
                    margin_bundle(logits, result.weights.shape[0])["logits"]
                )
        replay, replay_arrays = replay_summary(values, spec, seed)
        result = {**identity, **summary, "replay": replay}
        write_arrays(
            directory / "raw.npz",
            flatten_arrays({**raw, **analysis, "replay": replay_arrays}),
        )
        write_json_exclusive(directory / "behavior.json", json_ready(sampled))
        write_json_exclusive(directory / "result.json", json_ready(result))
        print(
            {"seed": seed, "arm": arm, "core": result["routes"]["full"]["core_flags"]},
            flush=True,
        )
    return result
