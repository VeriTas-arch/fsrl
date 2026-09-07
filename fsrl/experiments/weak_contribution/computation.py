"""Global-only removals and local reconstruction at frozen parameters."""

import numpy as np
import torch

from fsrl.experiments.training_strategy.batches import EpisodeBatch


def relation_mask(cpu, relation):
    return np.all(
        np.sort(cpu.arrays["support_pairs"], axis=-1) == sorted(relation), axis=-1
    )


def remove_global(cpu, mask):
    arrays = dict(cpu.arrays)
    inputs = arrays["support_inputs"].copy()
    for channel in (34, 37):
        inputs[:, 0, :, channel][mask] = 0
    arrays["support_inputs"] = inputs
    return EpisodeBatch(arrays)


def remove_all_weak(cpu):
    mask = cpu.arrays["trial_retention"] == 0
    if np.any(cpu.arrays["support_inputs"][:, 0, :, 34][mask] != 0):
        raise RuntimeError("weak cells must have zero strong-channel evidence")
    return remove_global(cpu, mask)


def local_margin(local, cpu, omitted=None):
    batch = cpu.to(local.raw_gain.device)
    subjects = batch.local_evidence.shape[1]
    state = local.initial_state(subjects)
    for index, inputs in enumerate(batch.support_inputs.unbind()):
        evidence = batch.local_evidence[index]
        if omitted is not None:
            evidence = evidence.masked_fill(
                torch.as_tensor(omitted[index], device=evidence.device), 0
            )
        state = local.write(state, inputs[0, :, : 2 * local.cue_size], evidence)
    count = batch.targets.numel() // subjects
    cues = batch.query_inputs[0, :, : 2 * local.cue_size]
    _, correction = local.read(state.repeat(count, 1), cues)
    return correction.reshape(count, subjects).T.cpu().numpy().astype(float)


def reconstruct(local, cpu, relations, removed):
    local_removed = np.stack(
        [local_margin(local, cpu, relation_mask(cpu, r)) for r in relations]
    )
    return removed - local_removed, local_removed


def compare(first, second):
    from .protocol import specification

    np.testing.assert_allclose(
        first, second, **specification()["execution"]["tolerance"]
    )
    return {"passed": True, "max_absolute_error": float(np.max(np.abs(first - second)))}


def input_checks(cpu, changed, relations):
    a, b = cpu.arrays, changed.arrays
    weak_trials = a["trial_retention"] == 0
    expected = a["support_inputs"].copy()
    expected[:, 0, :, 37][weak_trials] = 0
    np.testing.assert_array_equal(expected, b["support_inputs"])
    for key in a:
        if key != "support_inputs":
            np.testing.assert_array_equal(a[key], b[key])
    counts = []
    for index, relation in enumerate(relations):
        mask = relation_mask(cpu, relation)
        np.testing.assert_array_equal(mask.sum(0), np.full(mask.shape[1], 4))
        retained = a["retention"][:, index]
        np.testing.assert_array_equal(
            a["trial_retention"][mask], np.broadcast_to(retained, mask.shape)[mask]
        )
        counts.append(~retained)
    weak = np.stack(counts)
    return {
        "passed": True,
        "weak_relation_counts": weak.sum(0).tolist(),
        "zero_weak_subjects": np.flatnonzero(~weak.any(0)).tolist(),
        "changed_trial_cells": int(weak_trials.sum()),
        "input_fingerprint": cpu.fingerprint(),
        "intervention_fingerprint": changed.fingerprint(),
    }
