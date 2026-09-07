"""Map a trained zero-hint model to an explicit 37-channel study interface."""

import copy
from dataclasses import replace

import numpy as np
import torch

from fsrl.experiments.evidence_routing.qualification import compare
from fsrl.experiments.finite_state.model import sequences
from fsrl.experiments.local_memory_removal.evaluation import load_model
from fsrl.experiments.local_memory_removal.model import rollout
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
    result.i2h.weight = torch.nn.Parameter(net.i2h.weight[:, keep].detach().clone())
    result.i2h.in_features = 37
    result.model_config = replace(result.model_config, input_size=37)
    return result


def compact_batch(cpu):
    arrays = dict(cpu.arrays)
    for key in ("support_inputs", "query_inputs"):
        assert not np.any(arrays[key][..., HINT])
        arrays[key] = np.delete(arrays[key], HINT, axis=-1)
    return EpisodeBatch(arrays)


def complete_order(logits, cpu):
    """Complete-graph least-squares order, allowing subject-specific orientations."""
    subjects = cpu.arrays["support_inputs"].shape[2]
    pairs = cpu.arrays["query_pairs"]
    if pairs.ndim == 2:
        pairs = np.broadcast_to(pairs[:, None], (len(pairs), subjects, 2))
    n = int(pairs.max()) + 1
    assert len(pairs) in (n * (n - 1), n * (n - 1) // 2)
    margins = (
        (logits[..., 1] - logits[..., 0]).detach().cpu().numpy().reshape(-1, subjects)
    )
    score = np.zeros((subjects, n))
    ids = np.arange(subjects)
    for index, pair in enumerate(pairs):
        score[ids, pair[:, 0]] += margins[index]
        score[ids, pair[:, 1]] -= margins[index]
    return np.argsort(-score, axis=1, kind="stable")


def compare_rollouts(first, second, cpu):
    a, cost_a, writes_a = first
    b, cost_b, writes_b = second
    checks = {
        name: compare(x, y)
        for name, x, y in (
            ("logits", a.logits, b.logits),
            ("P_T", a.weights, b.weights),
            ("first_P", a.first_write, b.first_write),
            ("cost", cost_a, cost_b),
            ("writes", writes_a, writes_b),
        )
    }
    assert torch.equal(a.logits.argmax(-1), b.logits.argmax(-1))
    assert np.array_equal(complete_order(a.logits, cpu), complete_order(b.logits, cpu))
    return {**checks, "choice_and_order_exact": True}


def export(source, models):
    directory = RUNS / "compact"
    records = {}
    with (
        ProspectiveRun.start(
            directory,
            workflow_id="admission_hint_removal_v1",
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
