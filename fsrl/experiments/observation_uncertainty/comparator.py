"""An information-matched least-squares observer, without fitting human data."""

import copy

import numpy as np

from fsrl.experiments.finite_state.liu import primary_analysis
from fsrl.experiments.memory_structure.inputs import size_protocol
from fsrl.experiments.memory_structure.interventions import (
    remove_relation,
    shuffle_evidence,
)
from fsrl.experiments.training_strategy.batches import EpisodeBatch
from fsrl.experiments.training_strategy.estimands import estimate
from fsrl.experiments.training_strategy.evaluation import (
    flatten_arrays,
    json_ready,
    write_arrays,
)
from fsrl.experiments.write_cost.evaluation import summarize_generic
from fsrl.experiments.write_cost.inputs import load_input
from fsrl.experiments.write_cost.locks import completed
from fsrl.infra.provenance import write_json_exclusive
from fsrl.infra.run_manifest import ProspectiveRun

from .diagnostic import history_groups
from .evaluation import observed
from .liu import replay_summary
from .protocol import PROTOCOL_SHA256, RUNS, specification


def solve(cpu, sigma):
    a = cpu.arrays
    pairs = a["support_pairs"]
    n = a["local_evidence"].shape[1]
    items = a["item_codes"].shape[1]
    queries = a["query_pairs"]
    margins = []
    ranks = []
    for subject in range(n):
        q = a["local_evidence"][:, subject].astype(float)
        design = np.zeros((len(q), items))
        design[np.arange(len(q)), pairs[:, subject, 0]] = 1
        design[np.arange(len(q)), pairs[:, subject, 1]] = -1
        present = q != 0
        scores, _, rank, _ = np.linalg.lstsq(design[present], q[present], rcond=None)
        query = queries if queries.ndim == 2 else queries[:, subject]
        margins.append((scores[query[:, 0]] - scores[query[:, 1]]) / sigma)
        ranks.append(rank)
    return np.asarray(margins), np.asarray(ranks)


def generic(source, spec, arm, split):
    rows = {}
    for name, ref in sorted(source["inputs"].items()):
        if not name.startswith(split + "-"):
            continue
        cpu = observed(load_input(ref), arm, spec)
        margin, rank = solve(cpu, spec["observation"]["sigma"])
        n = len(margin)
        values = {
            "margins": margin,
            "signs": (2 * cpu.arrays["targets"] - 1).reshape(-1, n).T,
            "learned": cpu.arrays["learned"],
            "rank": rank,
            "cost": np.zeros(n),
            "total_write": np.zeros(n),
        }
        for key, value in values.items():
            rows.setdefault(key, []).append(value)
    arrays = {key: np.concatenate(value) for key, value in rows.items()}
    summary = summarize_generic(arrays, spec, spec["seeds"]["development"])
    summary["global"] = copy.deepcopy(summary)
    return summary, arrays


def history(source, spec):
    arrays = {}
    for members in history_groups(source, spec).values():
        for index, pair in members:
            for label, a in pair.items():
                full, _ = solve(EpisodeBatch(a), spec["observation"]["sigma"])
                altered = copy.deepcopy(a)
                altered["local_evidence"][int(a["target_index"])] = 0
                removed, _ = solve(EpisodeBatch(altered), spec["observation"]["sigma"])
                signs = (2 * a["targets"] - 1)[None]
                values = {
                    "global_ce_benefit": (
                        np.logaddexp(0, -removed * signs)
                        - np.logaddexp(0, -full * signs)
                    ).mean(1),
                    "global_margins": full,
                    "removed_global_margins": removed,
                }
                for key, value in values.items():
                    name = f"{label}__{key}"
                    if name not in arrays:
                        arrays[name] = np.empty(
                            (spec["diagnostic"]["episodes"], *value.shape[1:])
                        )
                    arrays[name][index] = value[0]
    delta = (
        arrays["supported__global_ce_benefit"]
        - arrays["conflicting__global_ce_benefit"]
    )
    return estimate(
        delta,
        seed=spec["statistics"]["seed_offset"] + spec["seeds"]["development"],
        statistics=spec["statistics"],
    ), arrays


def run(source):
    spec = specification()
    results = {}
    for arm in spec["seeds"]["conditions"]:
        directory = RUNS / "comparator" / arm
        if directory.exists():
            results[arm] = completed(directory)
            continue
        identity = {
            "arm": arm,
            "observer": "least_squares",
            "protocol_sha256": PROTOCOL_SHA256,
        }
        with ProspectiveRun.start(
            directory,
            workflow_id="observation_uncertainty_v1",
            execution_id=f"comparator-{arm}",
            producer=identity,
            resolved_config=spec["comparator"],
        ):
            summaries = {}
            raw = {}
            for split in ("development", "test"):
                summaries[split], raw[split] = generic(source, spec, arm, split)
            base = load_input(source["inputs"]["liu-8"])
            cpu = observed(base, arm, spec)
            protocol = size_protocol(spec, 8)
            intact, rank = solve(cpu, spec["observation"]["sigma"])
            shuffled, _ = shuffle_evidence(
                cpu,
                protocol.support_blocks,
                spec["evaluation"]["liu"]["evidence_shuffle_seed"],
            )
            shuffled_margin, _ = solve(shuffled, spec["observation"]["sigma"])
            bundles = {
                "intact": {"logits": intact},
                "local_off": {"logits": intact},
                "P_off": {"logits": np.zeros_like(intact)},
                "evidence_shuffle": {"logits": shuffled_margin},
                "query_shuffle": {"logits": intact},
            }
            removed = np.stack(
                [
                    solve(remove_relation(cpu, relation), spec["observation"]["sigma"])[
                        0
                    ]
                    for relation in protocol.support_pairs_higher_lower
                ]
            )
            liu_raw = {
                "bundles": bundles,
                "removed": removed,
                "removed_global": removed,
                "cost": np.zeros(len(intact)),
                "storage": {"design_rank": rank},
            }
            liu_summary, analysis, sampled = primary_analysis(
                liu_raw,
                cpu,
                protocol,
                spec,
                spec["seeds"]["development"],
                summaries["test"],
            )
            values = [intact] + [
                solve(observed(base, arm, spec, replay), spec["observation"]["sigma"])[
                    0
                ]
                for replay in (1, 2)
            ]
            replay, replay_arrays = replay_summary(
                {"full": values, "global": values}, spec, spec["seeds"]["development"]
            )
            raw["liu"] = {**liu_raw, **analysis, "replay": replay_arrays}
            history_summary, raw["history"] = history(source, spec)
            result = {
                **identity,
                "generic": summaries,
                "liu": {**liu_summary, "replay": replay},
                "history": history_summary,
                "scope": "One deterministic comparator per observation condition, not three independent fitted networks; global/full same; no local path.",
            }
            write_arrays(directory / "raw.npz", flatten_arrays(raw))
            write_json_exclusive(directory / "behavior.json", json_ready(sampled))
            write_json_exclusive(directory / "result.json", json_ready(result))
        results[arm] = result
    return results
