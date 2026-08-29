"""Render the deterministic encoding and P/L query snapshot used by audit v2."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Any

import numpy as np
import torch

from fsrl.core.config import TrainConfig
from fsrl.core.local_trace import ConjunctiveLocalTrace
from fsrl.core.plastic_rnn import RetroModulRNN
from fsrl.evaluation.frozen_fast_weight import (
    FastWeightIntervention,
    FrozenFastWeightEvaluator,
)
from fsrl.experiments.local_fidelity.trace_pilot import build_local_trace, query_bundle
from fsrl.tasks.protocol import ordered_pairs
from fsrl.tasks.protocol_catalog import load_registered_protocol


def _digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _digest_json(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return _digest_bytes(payload.encode("utf-8"))


def build_snapshot() -> dict[str, Any]:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(107)
    config = TrainConfig(bs=3, hs=8, cs=8, nbcues_min=8, nbcues_max=8)
    net = RetroModulRNN(config.to_model_dict(), device="cpu")
    protocol = load_registered_protocol("liu_v1")
    evaluator = FrozenFastWeightEvaluator(
        net,
        config,
        protocol,
        cue_seed=109,
        support_seed=113,
        subject_encoding_mode="stable_omission",
        subject_encoding_seed=127,
    )
    local = ConjunctiveLocalTrace(config.cs, initial_gain=0.2, device="cpu")
    fast_weights = evaluator.learn_fast_weights(FastWeightIntervention.INTACT)
    local_state = build_local_trace(evaluator, local)
    pairs = ordered_pairs(protocol.n_items)
    schedules = tuple(pairs for _ in range(config.bs))
    encoding = {
        "states": [asdict(state) for state in evaluator.subject_encoding_states or ()],
        "relation_gains": [
            [[list(pair), value] for pair, value in sorted(subject.items())]
            for subject in evaluator.subject_relation_gains or ()
        ],
        "trial_gains": evaluator.subject_trial_gains,
        "support_schedules": [
            [asdict(trial) for trial in schedule]
            for schedule in evaluator.support_schedules
        ],
    }
    result: dict[str, Any] = {
        "cue_codes": _digest_bytes(np.ascontiguousarray(evaluator.cue_codes).tobytes()),
        "encoding": _digest_json(encoding),
        "fast_weights": _digest_bytes(
            fast_weights.detach().cpu().contiguous().numpy().tobytes()
        ),
        "local_state": _digest_bytes(
            local_state.detach().cpu().contiguous().numpy().tobytes()
        ),
        "query_conditions": {},
    }
    for condition in (
        "original_v1_local_off",
        "dual_intact",
        "local_query_key_shuffle",
        "global_P_off_local_intact",
    ):
        bundle = query_bundle(
            evaluator,
            local,
            fast_weights,
            local_state,
            schedules,
            condition=condition,
            shuffle_seed=131,
        )
        result["query_conditions"][condition] = {
            name: _digest_bytes(np.ascontiguousarray(value).tobytes())
            for name, value in sorted(bundle.items())
        }
    return result


def main() -> None:
    print(json.dumps(build_snapshot(), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
