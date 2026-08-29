"""Registered checkpoint-to-evaluator loading for maintained read-only analyses."""

from __future__ import annotations

from fsrl.infra.provenance import load_json
from fsrl.infra.study_registry import resolve_registered_path
from fsrl.tasks.protocol import RankingProtocol
from fsrl.training.checkpoints import validate_meta_checkpoint

from .frozen_fast_weight import (
    FrozenFastWeightEvaluator,
    load_frozen_retro_checkpoint,
)


def load_registered_frozen_evaluator(
    registration: dict,
    training_specification: dict,
    protocol: RankingProtocol,
) -> tuple[FrozenFastWeightEvaluator, dict]:
    """Load one registered evaluator and its matched behavioral result."""

    seed = int(registration["seed"])
    checkpoint = resolve_registered_path(registration["checkpoint_path"])
    validate_meta_checkpoint(checkpoint, training_specification, seed)
    behavior = load_json(resolve_registered_path(registration["behavior_path"]))
    net, config, checkpoint_info = load_frozen_retro_checkpoint(
        checkpoint, len(behavior["subjects"])
    )
    if behavior["checkpoint"]["sha256"] != checkpoint_info.sha256:
        raise RuntimeError(f"seed {seed} behavior and checkpoint do not match")
    evaluator = FrozenFastWeightEvaluator(
        net,
        config,
        protocol,
        cue_seed=int(behavior["cue_seed"]),
        support_seed=int(behavior["support_seed"]),
        cue_mode="permuted_shared",
        subject_encoding_mode=behavior["subject_encoding_mode"],
        subject_encoding_seed=int(behavior["subject_encoding_seed"]),
    )
    return evaluator, behavior
