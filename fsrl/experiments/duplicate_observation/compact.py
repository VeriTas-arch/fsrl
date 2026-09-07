"""Map a trained duplicate-observation model to an explicit 37-channel study interface."""

import copy
from dataclasses import replace

import numpy as np
import torch

from fsrl.experiments.admission_hint_removal.compact import compare_rollouts
from fsrl.experiments.evidence_routing.qualification import compare
from fsrl.experiments.finite_state.model import sequences
from fsrl.experiments.local_memory_removal.evaluation import load_model
from fsrl.experiments.local_memory_removal.model import rollout
from fsrl.experiments.memory_structure.inputs import size_protocol
from fsrl.experiments.memory_structure.interventions import (
    remove_relation,
    shuffle_evidence,
)
from fsrl.experiments.memory_structure.model import read_queries
from fsrl.experiments.training_strategy.batches import EpisodeBatch
from fsrl.experiments.training_strategy.evaluation import json_ready
from fsrl.experiments.training_strategy.locks import reference
from fsrl.experiments.write_cost.inputs import load_input
from fsrl.infra.provenance import tensor_hashes, write_json_exclusive
from fsrl.infra.run_manifest import ProspectiveRun

from .inputs import HINT, observed
from .protocol import MODELS, RUNS, recipe, specification


def compact_model(net):
    result = copy.deepcopy(net)
    assert result.model_config.input_size == 38
    keep = [i for i in range(38) if i != HINT]
    weight = net.i2h.weight.detach().clone()
    weight[:, 37] += weight[:, HINT]
    result.i2h.weight = torch.nn.Parameter(weight[:, keep].clone())
    result.i2h.in_features = 37
    result.model_config = replace(result.model_config, input_size=37)
    return result


def compact_batch(cpu):
    arrays = dict(cpu.arrays)
    for key in ("support_inputs", "query_inputs"):
        assert np.array_equal(arrays[key][..., HINT], arrays[key][..., 37])
        arrays[key] = np.delete(arrays[key], HINT, axis=-1)
    return EpisodeBatch(arrays)


def control_checks(net, smaller, padded, compact, cpu, spec):
    task = size_protocol(spec, 8)
    shuffled, _ = shuffle_evidence(
        cpu, task.support_blocks, spec["evaluation"]["liu"]["evidence_shuffle_seed"]
    )
    controls = {"evidence_shuffle": shuffled}
    controls.update(
        {
            f"removed/{index}": remove_relation(cpu, relation)
            for index, relation in enumerate(task.support_pairs_higher_lower)
        }
    )
    checks = {}
    for label, batch in controls.items():
        checks[label] = compare_rollouts(
            rollout(net, None, padded, batch.to("cuda"), None, 0),
            rollout(smaller, None, compact, compact_batch(batch).to("cuda"), None, 0),
            batch,
        )
    original, _, _ = rollout(net, None, padded, cpu.to("cuda"), None, 0)
    a, _ = read_queries(
        net, None, padded[1], cpu.to("cuda"), torch.zeros_like(original.weights), None
    )
    b, _ = read_queries(
        smaller,
        None,
        compact[1],
        compact_batch(cpu).to("cuda"),
        torch.zeros_like(original.weights),
        None,
    )
    checks["P_off"] = compare(a, b)
    assert torch.equal(a.argmax(-1), b.argmax(-1))
    return checks


def export(source, models):
    directory = RUNS / "compact"
    records = {}
    with (
        ProspectiveRun.start(
            directory,
            workflow_id="duplicate_observation_v1",
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
