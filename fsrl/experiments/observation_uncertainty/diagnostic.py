"""Identical current observations under supported and conflicting histories."""

from collections import defaultdict

import numpy as np
import torch

from fsrl.experiments.finite_state.diagnostic import history_effect
from fsrl.experiments.training_strategy.estimands import estimate
from fsrl.experiments.training_strategy.evaluation import write_arrays
from fsrl.experiments.write_cost.diagnostic import joined
from fsrl.experiments.write_cost.inputs import load_input
from fsrl.experiments.write_cost.locks import completed
from fsrl.infra.provenance import write_json_exclusive
from fsrl.infra.run_manifest import ProspectiveRun

from .evaluation import loaded_model
from .protocol import PROTOCOL_SHA256, RUNS, specification


def history_groups(source, spec):
    histories = load_input(source["inputs"]["histories"]).arrays
    groups = defaultdict(list)
    for index in range(spec["diagnostic"]["episodes"]):
        pair = {
            label: {
                key.removeprefix(f"{index}__{label}__"): value
                for key, value in histories.items()
                if key.startswith(f"{index}__{label}__")
            }
            for label in ("supported", "conflicting")
        }
        groups[len(pair["supported"]["support_inputs"])].append((index, pair))
    return groups


def diagnostic_run(seed, arm, source):
    directory = RUNS / "diagnostic" / str(seed) / arm
    if directory.exists():
        return completed(directory)
    spec = specification()
    backbone, local, seqs = loaded_model(seed, arm, source)
    identity = {"seed": seed, "arm": arm, "protocol_sha256": PROTOCOL_SHA256}
    with (
        ProspectiveRun.start(
            directory,
            workflow_id="observation_uncertainty_v1",
            execution_id=f"history-{seed}-{arm}",
            producer=identity,
            resolved_config=spec["diagnostic"],
        ),
        torch.no_grad(),
    ):
        arrays = {}
        for members in history_groups(source, spec).values():
            indices = [row[0] for row in members]
            for label in ("supported", "conflicting"):
                values = history_effect(
                    backbone,
                    local,
                    seqs,
                    None,
                    joined([row[1][label] for row in members]),
                    0,
                )
                for key, value in values.items():
                    name = f"{label}__{key}"
                    if name not in arrays:
                        arrays[name] = np.empty(
                            (spec["diagnostic"]["episodes"], *value.shape[1:])
                        )
                    arrays[name][indices] = value
        summary = {
            key.removeprefix("supported__"): estimate(
                value - arrays[key.replace("supported__", "conflicting__")],
                seed=spec["statistics"]["seed_offset"] + seed,
                statistics=spec["statistics"],
            )
            for key, value in arrays.items()
            if key.startswith("supported__") and value.ndim == 1
        }
        result = {**identity, "supported_minus_conflicting": summary}
        write_arrays(directory / "raw.npz", arrays)
        write_json_exclusive(directory / "result.json", result)
    return result
