"""Conditional single-observation export of the linear modulation models."""

import torch

from fsrl.experiments.admission_hint_removal.compact import compare_rollouts
from fsrl.experiments.duplicate_observation.compact import (
    compact_batch,
    compact_model,
    control_checks,
)
from fsrl.experiments.duplicate_observation.inputs import observed
from fsrl.experiments.finite_state.model import sequences
from fsrl.experiments.local_memory_removal.model import rollout
from fsrl.experiments.training_strategy.evaluation import json_ready
from fsrl.experiments.training_strategy.locks import reference
from fsrl.experiments.write_cost.inputs import load_input
from fsrl.infra.provenance import tensor_hashes, write_json_exclusive
from fsrl.infra.run_manifest import ProspectiveRun

from .model import load_model
from .protocol import MODELS, RUNS, recipe, specification


def export(source, models):
    directory = RUNS / "compact"
    records = {}
    with (
        ProspectiveRun.start(
            directory,
            workflow_id="linear_modulation_v1",
            execution_id="compact-equivalence",
            producer={"model_lock": reference(MODELS)},
            resolved_config=specification()["compact_export"],
        ),
        torch.no_grad(),
    ):
        for key, model in models["runs"].items():
            seed = int(key.split("/")[0])
            net, _, padded = load_model(seed, model["files"], recipe())
            smaller = compact_model(net).requires_grad_(False).eval()
            compact = sequences(smaller, None, compiled=True)
            checks = {}
            for panel in specification()["design"]["panels"]:
                for arm in ("clean", "noisy"):
                    for name, ref in sorted(
                        source["panels"][str(panel)]["inputs"].items()
                    ):
                        cpu = observed(load_input(ref), arm, recipe(panel))
                        small = compact_batch(cpu)
                        checks[f"{panel}/{arm}/{name}"] = compare_rollouts(
                            rollout(net, None, padded, cpu.to("cuda"), None, 0),
                            rollout(smaller, None, compact, small.to("cuda"), None, 0),
                            cpu,
                        )
                        if name == "liu-8":
                            checks[f"{panel}/{arm}/{name}/controls"] = control_checks(
                                net, smaller, padded, compact, cpu, recipe(panel)
                            )
            path = directory / (key.replace("/", "-") + ".pth")
            with path.open("xb") as handle:
                torch.save(smaller.state_dict(), handle)
            records[key] = {
                "weights": reference(path),
                "parameters": tensor_hashes(smaller),
                "model_config": vars(smaller.model_config),
                "checks": checks,
            }
        result = {"passed": True, "models": records, "historical_ABI": False}
        write_json_exclusive(directory / "result.json", json_ready(result))
    return reference(directory / "result.json")
