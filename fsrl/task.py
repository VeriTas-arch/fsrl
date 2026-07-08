import numpy as np
import torch

from .config import ADDINPUT, DEVICE, NUMRESPONSESTEP


def generate_cue_data(config, nbcues):
    """Generate unique random binary cue vectors for each batch element."""
    cue_data = []
    for batch_index in range(config.bs):
        cue_data.append([])
        for cue_index in range(nbcues):
            candidate = sample_unique_cue(config, cue_data[batch_index], cue_index)
            cue_data[batch_index].append(candidate)
    return cue_data


def sample_unique_cue(config, existing_cues, cue_index):
    attempts = 0
    while True:
        attempts += 1
        if attempts > 10000:
            raise ValueError("Could not generate a full list of different cues")

        candidate = np.random.randint(2, size=config.cs) * 2 - 1
        is_too_similar = False
        for previous_index in range(cue_index):
            if np.mean(existing_cues[previous_index] == candidate) > 0.66:
                is_too_similar = True
                break
        if not is_too_similar:
            return candidate


def sample_trial_pair(nbcues, is_train_trial):
    cue_pair = list(np.random.choice(range(nbcues), 2, replace=False))
    if is_train_trial:
        while abs(cue_pair[0] - cue_pair[1]) > 1:
            cue_pair = list(np.random.choice(range(nbcues), 2, replace=False))
    return cue_pair


def build_step_inputs(
    config, nbcues, cue_data, cues, reward, previous_actions, numstep, numstep_ep
):
    inputs = np.zeros((config.bs, config.inputsize), dtype="float32")

    for batch_index in range(config.bs):
        cue = cues[batch_index][numstep]
        if isinstance(cue, (list, tuple, np.ndarray)):
            inputs[batch_index, : config.nbstimbits - 1] = np.concatenate(
                (cue_data[batch_index][cue[0]][:], cue_data[batch_index][cue[1]][:])
            )
        elif cue == nbcues:
            inputs[batch_index, config.nbstimbits - 1] = 1

        inputs[batch_index, config.nbstimbits + 0] = 1.0
        inputs[batch_index, config.nbstimbits + 1] = numstep_ep / config.eplen
        inputs[batch_index, config.nbstimbits + 2] = reward[batch_index]

        if numstep == NUMRESPONSESTEP + 1:
            inputs[
                batch_index,
                config.nbstimbits + ADDINPUT + previous_actions[batch_index],
            ] = 1

    return torch.from_numpy(inputs).detach().to(DEVICE)


def prepare_trial(config, nbcues, nbtrials):
    cues = []
    cue_pairs = []
    correct_order = np.zeros(config.bs)
    adjacent = np.zeros(config.bs)

    for batch_index in range(config.bs):
        cue_pair = sample_trial_pair(
            nbcues, nbtrials[batch_index] < config.nbtraintrials
        )
        assert nbtrials[batch_index] == int(nbtrials[batch_index])

        correct_order[batch_index] = 1 if cue_pair[0] < cue_pair[1] else 0
        adjacent[batch_index] = 1 if abs(cue_pair[0] - cue_pair[1]) == 1 else 0
        cue_pairs.append(cue_pair)
        cues.append([cue_pair, nbcues, -1, -1])

    return cues, cue_pairs, correct_order, adjacent
