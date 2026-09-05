"""Fixed held-out graph stream, grouped only for same-length CUDA execution."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

from fsrl.core.sequence import RecurrentSequence
from fsrl.infra.runtime import compile_module
from fsrl.tasks.sparse_ranking import GenericRankingTaskGenerator
from fsrl.training.backbone import registered_excluded_signatures

from .batches import prepare_batch, sample_episodes
from .estimands import query_endpoints
from .execution import PROFILE
from .optimization import forward_batch, query_from_state


def validation_episodes(specification: dict) -> tuple:
    task = specification["task"]
    generator = GenericRankingTaskGenerator(
        cue_size=task["cue_size"],
        min_edges=task["min_edges"],
        max_edges=task["max_edges"],
        support_blocks=task["support_blocks"],
        excluded_signatures=registered_excluded_signatures(),
        subject_encoding_mode=task["subject_encoding_mode"],
    )
    settings = specification["evaluation"]["generic"]
    rng = np.random.default_rng(settings["rng_seed"])
    return tuple(
        sample_episodes(generator, rng, 1, validation=True)[0]
        for _ in range(settings["episodes"])
    )


def validation_groups(episodes: tuple) -> dict:
    groups = defaultdict(list)
    for index, episode in enumerate(episodes):
        groups[len(episode.support_trials)].append(index)
    return dict(groups)


def _group_rollout(backbone, local, sequence, episodes: tuple) -> dict:
    cpu = prepare_batch(episodes)
    batch = cpu.to(next(backbone.parameters()).device)
    size = len(episodes)
    with torch.no_grad():
        result = forward_batch(
            backbone, local, sequence, batch, local_active=True, fast_weight_penalty=0.0
        )
        p_off, _ = query_from_state(
            backbone,
            local,
            sequence,
            batch,
            torch.zeros_like(result.fast_weights),
            result.local_state,
            local_active=True,
        )
        logits = {
            "intact": result.logits,
            "local_off": result.global_logits,
            "P_off": p_off,
        }
        margins = {
            name: (value[:, 1] - value[:, 0])
            .reshape(28, size)
            .T.cpu()
            .numpy()
            .astype(np.float64)
            for name, value in logits.items()
        }
    learned = np.asarray(
        [
            [
                tuple(sorted((query.left_item, query.right_item)))
                in {
                    tuple(sorted((trial.left_item, trial.right_item)))
                    for trial in episode.support_trials
                }
                for query in episode.query_trials
            ]
            for episode in episodes
        ],
        dtype=bool,
    )
    return {
        "margins": margins,
        "correct_signs": (2 * cpu.arrays["targets"] - 1).reshape(28, size).T,
        "learned": learned,
        "inputs": cpu.arrays,
        "fingerprint": cpu.fingerprint(),
    }


def evaluate_generic(backbone, local, specification: dict) -> dict:
    episodes = validation_episodes(specification)
    sequence = compile_module(RecurrentSequence(backbone), PROFILE)
    settings = specification["evaluation"]["generic"]
    margins = {name: np.empty((len(episodes), 28)) for name in settings["conditions"]}
    signs = np.empty((len(episodes), 28))
    learned = np.empty((len(episodes), 28), dtype=bool)
    groups = {}
    for length, indices in validation_groups(episodes).items():
        result = _group_rollout(
            backbone, local, sequence, tuple(episodes[index] for index in indices)
        )
        for name, values in result["margins"].items():
            margins[name][indices] = values
        signs[indices] = result["correct_signs"]
        learned[indices] = result["learned"]
        groups[str(length)] = {
            "episode_indices": np.asarray(indices),
            "inputs": result["inputs"],
            "fingerprint": result["fingerprint"],
        }
    endpoints = {
        name: query_endpoints(
            values[:, :, None],
            signs[:, :, None],
            {"learned": learned, "nonlearned": ~learned},
            temperature=settings["temperature"],
        )
        for name, values in margins.items()
    }
    return {
        "endpoints": endpoints,
        "margins": margins,
        "correct_signs": signs,
        "learned": learned,
        "groups": groups,
    }
