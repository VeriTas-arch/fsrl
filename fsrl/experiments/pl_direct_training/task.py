"""Unchanged generic task distribution for direct P/L training."""

from __future__ import annotations

from fsrl.tasks.sparse_ranking import GenericRankingTaskGenerator
from fsrl.training.backbone import registered_excluded_signatures


def make_task_generator(specification: dict) -> GenericRankingTaskGenerator:
    task = specification["task"]
    return GenericRankingTaskGenerator(
        n_items=task["n_items"],
        cue_size=task["cue_size"],
        min_edges=task["min_edges"],
        max_edges=task["max_edges"],
        support_blocks=task["support_blocks"],
        excluded_signatures=registered_excluded_signatures(),
        subject_encoding_mode=task["subject_encoding_mode"],
    )
