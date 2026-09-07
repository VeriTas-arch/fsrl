"""Complete registered secondary history CE without changing primary outputs.

The initial collector saved global margins only. This separate, committed
collector replays the same frozen histories/models, requires exact equality to
those global margins, and records full-policy CE alongside global-policy CE.
"""

import json
from pathlib import Path

import numpy as np
import torch

from fsrl.experiments.evidence_routing.evaluation import margin_bundle
from fsrl.experiments.evidence_routing.locks import require_clean
from fsrl.experiments.finite_state.evaluation import arrays_at
from fsrl.experiments.finite_state.model import rollout
from fsrl.experiments.training_strategy.estimands import estimate
from fsrl.experiments.training_strategy.evaluation import (
    flatten_arrays,
    json_ready,
    write_arrays,
)
from fsrl.experiments.training_strategy.locks import reference, verify_reference
from fsrl.experiments.write_cost.diagnostic import joined
from fsrl.experiments.write_cost.locks import completed
from fsrl.infra.formal_runtime import configure_formal_runtime
from fsrl.infra.provenance import load_json, write_json_exclusive
from fsrl.infra.run_manifest import ProspectiveRun
from fsrl.infra.validation_session import validation_session

from .diagnostic import history_groups
from .evaluation import loaded_model
from .locks import MODELS, SOURCE, validate_phase
from .protocol import PROTOCOL_SHA256, RUNS, specification


def run():
    validate_phase(MODELS)
    commit = require_clean()
    source_ref = reference(Path(__file__))
    verify_reference(source_ref, commit=commit)
    source, spec = load_json(SOURCE), specification()
    directory = RUNS / "secondary_history"
    if directory.exists():
        return completed(directory)
    identity = {
        "source_commit": commit,
        "collector": source_ref,
        "protocol_sha256": PROTOCOL_SHA256,
        "model_lock": reference(MODELS),
        "primary_result": reference(RUNS / "primary.json"),
        "purpose": "Complete protocol diagnostic.secondary full/global CE; no new estimand or changed primary decision.",
    }
    summaries, arrays, checks = {}, {}, 0
    with (
        ProspectiveRun.start(
            directory,
            workflow_id="observation_uncertainty_v1",
            execution_id="secondary-history-policy",
            producer=identity,
            resolved_config=spec["diagnostic"],
        ),
        torch.no_grad(),
    ):
        groups = history_groups(source, spec)
        for seed in spec["seeds"]["mandatory"]:
            summaries[str(seed)] = {}
            arrays[str(seed)] = {}
            for arm in spec["seeds"]["conditions"]:
                backbone, local, seqs = loaded_model(seed, arm, source)
                previous = arrays_at(RUNS / "diagnostic" / str(seed) / arm)
                raw = {}
                for members in groups.values():
                    indices = [row[0] for row in members]
                    for label in ("supported", "conflicting"):
                        cpu = joined([row[1][label] for row in members])
                        result, _, _ = rollout(
                            backbone, local, seqs, cpu.to("cuda"), None, 0
                        )
                        n = result.weights.shape[0]
                        global_margin = margin_bundle(result.global_logits, n)["logits"]
                        if not np.array_equal(
                            global_margin, previous[label + "__global_margins"][indices]
                        ):
                            raise RuntimeError(
                                "secondary replay differs from frozen global margins"
                            )
                        checks += 1
                        signs = (2 * cpu.arrays["targets"] - 1).reshape(-1, n).T
                        values = {
                            "global_margins": global_margin,
                            "full_margins": margin_bundle(result.logits, n)["logits"],
                            "signs": signs,
                        }
                        for route in ("full", "global"):
                            values[route + "_ce"] = np.logaddexp(
                                0, -values[route + "_margins"] * signs
                            ).mean(1)
                        for key, value in values.items():
                            name = f"{label}__{key}"
                            if name not in raw:
                                raw[name] = np.empty(
                                    (spec["diagnostic"]["episodes"], *value.shape[1:])
                                )
                            raw[name][indices] = value
                arrays[str(seed)][arm] = raw
                summaries[str(seed)][arm] = {
                    key: estimate(
                        value,
                        seed=spec["statistics"]["seed_offset"] + seed,
                        statistics=spec["statistics"],
                    )
                    for key, value in raw.items()
                    if key.endswith("_ce")
                }
            acute = arrays_at(RUNS / "diagnostic" / str(seed) / "acute_noisy")
            clean = arrays[str(seed)]["clean"]
            for label in ("supported", "conflicting"):
                if not np.array_equal(
                    acute[label + "__global_margins"], clean[label + "__global_margins"]
                ):
                    raise RuntimeError(
                        "acute control is not identical to clean on fixed history panel"
                    )
            summaries[str(seed)]["acute_noisy"] = {
                "identical_model_and_input_as": "clean",
                "values": summaries[str(seed)]["clean"],
            }
        # The observer has no separate local route; full and global coincide.
        comparator_raw = arrays_at(RUNS / "comparator/clean")
        comparator = {}
        for label in ("supported", "conflicting"):
            signs = arrays[str(spec["seeds"]["mandatory"][0])]["clean"][
                label + "__signs"
            ]
            margin = comparator_raw[f"history__{label}__global_margins"]
            ce = np.logaddexp(0, -margin * signs).mean(1)
            comparator[label] = estimate(
                ce,
                seed=spec["statistics"]["seed_offset"] + spec["seeds"]["development"],
                statistics=spec["statistics"],
            )
        result = {
            **identity,
            "pairs": summaries,
            "least_squares_full_equals_global": comparator,
            "exact_global_replay_checks": checks,
            "acute_history_reuse": "clean and acute_noisy are the same fitted network on the identical controlled history input; no new execution needed.",
        }
        write_arrays(directory / "raw.npz", flatten_arrays(arrays))
        write_json_exclusive(directory / "result.json", json_ready(result))
    return {
        "exact_global_replay_checks": checks,
        "status": "registered_secondary_completed",
    }


def main():
    configure_formal_runtime()
    with validation_session():
        result = run()
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
