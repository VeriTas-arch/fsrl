"""High-level causal-suite execution over the frozen evaluator."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np

from fsrl.infra.runtime import (
    ExecutionProfile,
    configure_runtime,
    default_device,
)
from fsrl.tasks.protocol import load_ranking_protocol
from fsrl.tasks.protocol_catalog import LIU_V1_PROTOCOL_PATH
from fsrl.tasks.subject_encoding import SubjectEncodingConfig
from fsrl.training.checkpoints import load_training_provenance
from fsrl.training.legacy_checkpoints import load_frozen_retro_checkpoint

from .contracts import (
    FastWeightIntervention,
    FrozenEvaluationBackend,
)
from .frozen_fast_weight import FrozenFastWeightEvaluator

DEFAULT_PROTOCOL_PATH = LIU_V1_PROTOCOL_PATH


def run_causal_suite(
    checkpoint: Path | str,
    *,
    batch_size: int,
    cue_seed: int,
    support_seed: int,
    order_seed: int,
    order_schedules: int,
    cue_mode: str,
    subject_encoding_mode: str,
    subject_encoding_seed: int,
    protocol_path: Path | str = DEFAULT_PROTOCOL_PATH,
    evaluation_backend: FrozenEvaluationBackend | str = (
        FrozenEvaluationBackend.BATCHED_SEQUENCE
    ),
    execution_profile: ExecutionProfile | None = None,
) -> dict:
    protocol_path = Path(protocol_path)
    protocol = load_ranking_protocol(protocol_path)
    backend = FrozenEvaluationBackend(evaluation_backend)
    runtime = None
    if backend == FrozenEvaluationBackend.BATCHED_SEQUENCE:
        selected_device = default_device()
        execution_profile = execution_profile or ExecutionProfile(
            device=selected_device,
            compile=selected_device == "cuda",
            require_cuda=selected_device == "cuda",
        )
        runtime = configure_runtime(execution_profile)
    net, config, checkpoint_info = load_frozen_retro_checkpoint(
        checkpoint,
        batch_size,
        device=(execution_profile.device if execution_profile is not None else None),
    )
    evaluator = FrozenFastWeightEvaluator(
        net,
        config,
        protocol,
        cue_seed=cue_seed,
        support_seed=support_seed,
        cue_mode=cue_mode,
        subject_encoding_mode=subject_encoding_mode,
        subject_encoding_seed=subject_encoding_seed,
        backend=backend,
        execution_profile=execution_profile,
    )
    conditions = {}
    condition_winners = {}
    for intervention in FastWeightIntervention:
        metrics, winners = evaluator.condition_evaluation(intervention)
        conditions[intervention.value] = asdict(metrics)
        condition_winners[intervention.value] = winners
    intact_winners = condition_winners[FastWeightIntervention.INTACT.value]
    for intervention, winners_by_subject in condition_winners.items():
        agreements = []
        for subject, winners in enumerate(winners_by_subject):
            agreements.extend(
                int(winner == intact_winners[subject][pair])
                for pair, winner in winners.items()
            )
        conditions[intervention]["mean_pair_decision_agreement_to_intact"] = float(
            np.mean(agreements)
        )
    intact_fast_weights = evaluator.learn_fast_weights(FastWeightIntervention.INTACT)
    invariance = evaluator.order_invariance(
        intact_fast_weights, schedules=order_schedules, seed=order_seed
    )
    provenance = load_training_provenance(Path(checkpoint), checkpoint_info.sha256)
    result = {
        "protocol_id": protocol.protocol_id,
        "protocol_path": str(protocol_path.resolve()),
        "checkpoint": asdict(checkpoint_info),
        "batch_size": batch_size,
        "cue_seed": cue_seed,
        "cue_mode": cue_mode,
        "subject_encoding": {
            "mode": subject_encoding_mode,
            "seed": subject_encoding_seed,
            "configuration": SubjectEncodingConfig().to_dict(),
        },
        "training_provenance": provenance,
        "support_seed": support_seed,
        "conditions": conditions,
        "order_invariance": asdict(invariance),
    }
    if evaluator.backend != FrozenEvaluationBackend.LEGACY_STEPWISE:
        result["evaluation_execution"] = evaluator.evaluation_execution_record()
        result["evaluation_execution"]["runtime"] = runtime
    return result
