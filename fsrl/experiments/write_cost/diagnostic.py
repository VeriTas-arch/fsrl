"""Matched current observations with or without earlier direct exposure."""

import copy
from collections import defaultdict

import numpy as np
import torch

from fsrl.experiments.evidence_routing.evaluation import margin_bundle
from fsrl.experiments.training_strategy.batches import EpisodeBatch
from fsrl.experiments.training_strategy.estimands import estimate
from fsrl.experiments.training_strategy.evaluation import write_arrays
from fsrl.infra.provenance import write_json_exclusive
from fsrl.infra.run_manifest import ProspectiveRun

from .evaluation import loaded_model
from .inputs import load_input
from .locks import completed
from .model import rollout
from .protocol import PROTOCOL_SHA256, RUNS, specification


def joined(rows):
    arrays = {
        "support_inputs": np.concatenate([r["support_inputs"] for r in rows], axis=2),
        "local_evidence": np.concatenate([r["local_evidence"] for r in rows], axis=1),
        "query_inputs": np.stack([r["query_inputs"] for r in rows], axis=2).reshape(
            2, -1, 38
        ),
        "targets": np.stack([r["targets"] for r in rows], axis=1).reshape(-1),
        "direct": np.stack([r["direct"] for r in rows]),
        "remote": np.stack([r["remote"] for r in rows]),
        "target_index": rows[0]["target_index"],
    }
    return EpisodeBatch(arrays)


def history_effect(backbone, local, seqs, cpu):
    target = int(cpu.arrays["target_index"])
    result, cost, writes = rollout(backbone, local, seqs[0], seqs[1], cpu.to("cuda"))
    altered = copy.deepcopy(cpu.arrays)
    altered["support_inputs"][target, 0, :, 34] = 0.0
    altered["support_inputs"][target, 0, :, 37] = 0.0
    removed, _, _ = rollout(
        backbone, local, seqs[0], seqs[1], EpisodeBatch(altered).to("cuda")
    )
    n = len(cost)
    full = margin_bundle(result.global_logits, n)["logits"]
    changed = margin_bundle(removed.global_logits, n)["logits"]
    signs = (2 * cpu.arrays["targets"] - 1).reshape(-1, n).T
    total = writes.sum(0).cpu().numpy().astype(float)
    target_write = writes[target].cpu().numpy().astype(float)
    output = {
        "target_write": target_write,
        "total_write": total,
        "mean_cost": cost.cpu().numpy().astype(float),
        "write_fraction": np.divide(
            target_write, total, out=np.full(n, np.nan), where=total > 0
        ),
        "global_ce_benefit": (
            np.logaddexp(0, -changed * signs) - np.logaddexp(0, -full * signs)
        ).mean(1),
        "global_margins": full,
        "removed_global_margins": changed,
    }
    for group in ("direct", "remote"):
        mask = cpu.arrays[group]
        output[group + "_influence"] = (np.abs(full - changed) * mask).sum(
            1
        ) / mask.sum(1)
    return output


def diagnostic_run(seed, arm, source):
    directory = RUNS / "diagnostic" / str(seed) / arm
    if directory.exists():
        return completed(directory)
    spec = specification()
    backbone, local, seqs = loaded_model(seed, arm, source)
    groups = defaultdict(list)
    histories = load_input(source["inputs"]["histories"]).arrays
    for index in range(spec["diagnostic"]["episodes"]):
        pair = {
            label: {
                key.removeprefix(f"{index}__{label}__"): value
                for key, value in histories.items()
                if key.startswith(f"{index}__{label}__")
            }
            for label in ("novel", "redundant")
        }
        groups[len(pair["novel"]["support_inputs"])].append((index, pair))
    identity = {"seed": seed, "arm": arm, "protocol_sha256": PROTOCOL_SHA256}
    with (
        ProspectiveRun.start(
            directory,
            workflow_id="effective_write_cost_v1",
            execution_id=f"diagnostic-{seed}-{arm}",
            producer=identity,
            resolved_config=spec["diagnostic"],
        ),
        torch.no_grad(),
    ):
        arrays = {}
        for members in groups.values():
            indices = [row[0] for row in members]
            for label in ("novel", "redundant"):
                output = history_effect(
                    backbone, local, seqs, joined([row[1][label] for row in members])
                )
                for key, value in output.items():
                    name = f"{label}__{key}"
                    if name not in arrays:
                        arrays[name] = np.empty(
                            (spec["diagnostic"]["episodes"], *value.shape[1:])
                        )
                    arrays[name][indices] = value
        differences = {
            key.removeprefix("novel__"): value
            - arrays[key.replace("novel__", "redundant__")]
            for key, value in arrays.items()
            if key.startswith("novel__") and value.ndim == 1
        }
        result = {
            **identity,
            "novel_minus_redundant": {
                key: estimate(
                    value,
                    seed=spec["statistics"]["seed_offset"] + seed,
                    statistics=spec["statistics"],
                )
                for key, value in differences.items()
            },
        }
        write_arrays(directory / "raw.npz", arrays)
        write_json_exclusive(directory / "result.json", result)
    return result
