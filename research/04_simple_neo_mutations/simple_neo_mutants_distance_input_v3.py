"""
Simple-NEO mutant suite for stepwise task/feedback replacements.

The default `exact_simple_neo` mode is intended to preserve the uploaded
simple_neo.py baseline: random nbcues in [4, 8], adjacent-only training trials,
all-pair rewarded test trials, four-step trial timing, per-trial reset of hidden
state and eligibility trace, episode-persistent plastic weights, and A2C-style
meta-training.

Mutants change one task ingredient at a time so we can locate where the
Miconi-style solution stops working when moving toward the behavioral-paper task.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

ROOT_DIR = Path(__file__).resolve().parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ADDINPUT = 4
# ADDINPUT slots after cue bits:
#   +0 bias, +1 normalized episode time, +2 previous reward, +3 signed rank-distance observation.
# The +3 slot was unused in v2; v3 uses it to provide behavioral-paper
# learning-pair distance independently of reward/action feedback.
DISTANCE_INPUT_OFFSET = 3
NUMRESPONSESTEP = 1
SCRIPT_VERSION = "single_neo_mutants_distance_input_v3_checked_2026-07-09"

VariantName = Literal[
    "exact_simple_neo",
    "n8_fixed",
    "behavior_graph_rewarded",
    "behavior_graph_no_test_feedback",
    "behavior_graph_no_test_feedback_with_test_loss",
    "observational_learning",
    "observational_learning_no_test_feedback",
    "observational_learning_no_test_feedback_with_train_aux",
    "observational_learning_no_test_feedback_with_test_loss",
    "observational_learning_no_test_feedback_with_train_aux_test_loss",
]

# Behavioral-paper sparse graph in rank/item-index coordinates.
# Lower index is higher/correct in the simple_neo convention.
BEHAVIOR_GRAPH_N8 = [
    (0, 5),  # A-F
    (1, 2),  # B-C
    (1, 4),  # B-E
    (2, 6),  # C-G
    (3, 5),  # D-F
    (3, 6),  # D-G
    (4, 7),  # E-H
    (0, 7),  # A-H
]


def log(message: str) -> None:
    print(message, flush=True)


@dataclass
class TrainConfig:
    rngseed: int = -1
    variant: VariantName = "exact_simple_neo"

    rew: float = 1.0
    wp: float = 0.0
    bent: float = 0.1
    blossv: float = 0.1
    gr: float = 0.9
    hs: int = 200
    bs: int = 32
    gc: float = 2.0
    eps: float = 1e-6
    nbiter: int = 30000
    save_every: int = 200
    pe: int = 101
    cs: int = 15
    triallen: int = 4
    nbtraintrials: int = 20
    nbtesttrials: int = 10
    testlmult: float = 3.0
    l2: float = 0.0
    lr: float = 1e-4
    lpw: float = 1e-4
    nbcues_min: int = 4
    nbcues_max: int = 8

    # Mutant-specific controls. Defaults preserve simple_neo where applicable.
    behavior_nbcues: int = 8
    observational_train_orients_correct_first: bool = True
    train_supervised_loss_weight: float = 1.0
    test_supervised_loss_weight: float = 1.0
    write_csv: bool = True
    num_threads: int = 1
    distance_input: bool = True
    # Distance is signed by displayed orientation and normalized to [-1, 1].
    # Positive means the first displayed cue is higher/correct; negative means the second is higher/correct.
    distance_input_train_only: bool = True

    @property
    def nbcuesrange(self):
        return range(self.nbcues_min, self.nbcues_max + 1)

    @property
    def nbtrials(self):
        return self.nbtraintrials + self.nbtesttrials

    @property
    def eplen(self):
        return self.nbtrials * self.triallen

    @property
    def nbstimbits(self):
        return 2 * self.cs + 1

    @property
    def outputsize(self):
        return 2

    @property
    def inputsize(self):
        return self.nbstimbits + ADDINPUT + self.outputsize

    def to_model_dict(self):
        return {
            "rngseed": self.rngseed,
            "rew": self.rew,
            "wp": self.wp,
            "bent": self.bent,
            "blossv": self.blossv,
            "gr": self.gr,
            "hs": self.hs,
            "bs": self.bs,
            "gc": self.gc,
            "eps": self.eps,
            "nbiter": self.nbiter,
            "save_every": self.save_every,
            "pe": self.pe,
            "nbcuesrange": self.nbcuesrange,
            "cs": self.cs,
            "triallen": self.triallen,
            "nbtraintrials": self.nbtraintrials,
            "nbtesttrials": self.nbtesttrials,
            "nbtrials": self.nbtrials,
            "eplen": self.eplen,
            "testlmult": self.testlmult,
            "l2": self.l2,
            "lr": self.lr,
            "lpw": self.lpw,
            "outputsize": self.outputsize,
            "inputsize": self.inputsize,
        }


class RetroModulRNN(nn.Module):
    """RNN with neuromodulated recurrent plasticity, matching simple_neo."""

    def __init__(self, config):
        super().__init__()
        for paramname in ["outputsize", "inputsize", "hs", "bs"]:
            if paramname not in config:
                raise KeyError("Must provide missing key in config: " + paramname)

        nbda = 2
        self.GG = config
        self.activ = torch.tanh
        self.i2h = torch.nn.Linear(config["inputsize"], config["hs"]).to(DEVICE)
        self.w = torch.nn.Parameter(
            (
                (1.0 / np.sqrt(config["hs"]))
                * (2.0 * torch.rand(config["hs"], config["hs"]) - 1.0)
            ).to(DEVICE),
            requires_grad=True,
        )
        self.alpha = torch.nn.Parameter(
            (0.01 * (2.0 * torch.rand(config["hs"], config["hs"]) - 1.0)).to(DEVICE),
            requires_grad=True,
        )
        self.etaet = torch.nn.Parameter((0.7 * torch.ones(1)).to(DEVICE), requires_grad=True)
        self.DAmult = torch.nn.Parameter((1.0 * torch.ones(1)).to(DEVICE), requires_grad=True)
        self.h2DA = torch.nn.Linear(config["hs"], nbda).to(DEVICE)
        self.h2o = torch.nn.Linear(config["hs"], config["outputsize"]).to(DEVICE)
        self.h2v = torch.nn.Linear(config["hs"], 1).to(DEVICE)

    def forward(self, inputs, hidden, et, pw):
        batch_size = inputs.shape[0]
        hidden_size = self.GG["hs"]
        assert pw.shape[0] == hidden.shape[0] == et.shape[0] == batch_size

        hactiv = self.activ(
            self.i2h(inputs).view(batch_size, hidden_size, 1)
            + torch.matmul(
                (self.w + torch.mul(self.alpha, pw)),
                hidden.view(batch_size, hidden_size, 1),
            )
        ).view(batch_size, hidden_size)

        activout = self.h2o(hactiv)
        valueout = self.h2v(hactiv)

        daout2 = torch.tanh(self.h2DA(hactiv))
        daout = self.DAmult * (daout2[:, 0] - daout2[:, 1])[:, None]

        pw = pw + daout.view(batch_size, 1, 1) * et
        torch.clip_(pw, min=-50.0, max=50.0)

        deltaet = torch.bmm(
            hactiv.view(batch_size, hidden_size, 1),
            hidden.view(batch_size, 1, hidden_size),
        )
        deltaet = torch.tanh(deltaet)
        et = (1 - self.etaet) * et + self.etaet * deltaet

        return activout, valueout, daout, hactiv, et, pw

    def initialZeroET(self, batch_size):
        return torch.zeros(batch_size, self.GG["hs"], self.GG["hs"], requires_grad=False).to(DEVICE)

    def initialZeroPlasticWeights(self, batch_size):
        return torch.zeros(batch_size, self.GG["hs"], self.GG["hs"], requires_grad=False).to(DEVICE)

    def initialZeroState(self, batch_size):
        return torch.zeros(batch_size, self.GG["hs"], requires_grad=False).to(DEVICE)


@dataclass
class EpisodeStats:
    loss: torch.Tensor
    loss_value: float
    loss_objective: float
    train_reward_mean: float
    test_reward_mean: float
    nbtraintrials: int
    nbtesttrials: int
    train_perf: float | None
    test_perf: float | None
    test_perf_adjacent: float | None
    test_perf_nonadjacent: float | None
    train_aux_loss: float
    test_ce_loss: float
    final_pw: torch.Tensor


def set_seed(seed):
    if seed < 0:
        log("[setup] No random seed.")
        return
    log(f"[setup] Setting random seed {seed}")
    np.random.seed(seed)
    torch.manual_seed(seed)


def generate_cue_data(config: TrainConfig, nbcues: int):
    cue_data = []
    for batch_index in range(config.bs):
        cue_data.append([])
        for cue_index in range(nbcues):
            candidate = sample_unique_cue(config, cue_data[batch_index], cue_index)
            cue_data[batch_index].append(candidate)
    return cue_data


def sample_unique_cue(config: TrainConfig, existing_cues, cue_index):
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


def sample_random_nonidentical_pair(nbcues: int) -> list[int]:
    return list(np.random.choice(range(nbcues), 2, replace=False))


def sample_adjacent_pair(nbcues: int) -> list[int]:
    cue_pair = sample_random_nonidentical_pair(nbcues)
    while abs(cue_pair[0] - cue_pair[1]) > 1:
        cue_pair = sample_random_nonidentical_pair(nbcues)
    return cue_pair


def sample_behavior_graph_pair(random_orientation: bool = True) -> list[int]:
    a, b = BEHAVIOR_GRAPH_N8[np.random.randint(len(BEHAVIOR_GRAPH_N8))]
    if random_orientation and np.random.rand() < 0.5:
        return [b, a]
    return [a, b]


def choose_nbcues_for_episode(config: TrainConfig) -> int:
    if config.variant == "exact_simple_neo":
        return int(np.random.choice(list(config.nbcuesrange)))
    if config.variant == "n8_fixed":
        return 8
    # Behavioral-graph mutants use the behavioral-paper 8-item graph.
    return config.behavior_nbcues


def is_behavior_graph_variant(variant: str) -> bool:
    return variant.startswith("behavior_graph") or variant.startswith("observational_learning")


def is_passive_observational_variant(variant: str) -> bool:
    return variant.startswith("observational_learning")


def suppress_test_feedback(variant: str) -> bool:
    return (
        variant == "behavior_graph_no_test_feedback"
        or variant == "behavior_graph_no_test_feedback_with_test_loss"
        or variant.startswith("observational_learning_no_test_feedback")
    )


def signed_rank_distance(cue_pair: list[int], nbcues: int) -> float:
    """Return displayed signed rank distance normalized to [-1, 1].

    simple_neo uses lower cue index as higher/correct. For a displayed pair
    [first, second], second-first > 0 means the first displayed cue is higher.
    The magnitude is rank distance divided by max possible distance.
    """
    if nbcues <= 1:
        return 0.0
    return float(cue_pair[1] - cue_pair[0]) / float(nbcues - 1)


def should_provide_distance_input(config: TrainConfig, is_train_trial: bool) -> bool:
    if not config.distance_input:
        return False
    if config.distance_input_train_only and not is_train_trial:
        return False
    # Use distance input for behavioral-paper variants; exact_simple_neo/n8_fixed stay
    # comparable to the original reward-only controls unless explicitly changed later.
    return is_behavior_graph_variant(config.variant)


def use_train_aux_loss(variant: str) -> bool:
    return variant in (
        "observational_learning_no_test_feedback_with_train_aux",
        "observational_learning_no_test_feedback_with_train_aux_test_loss",
    )


def use_test_supervised_loss(variant: str) -> bool:
    return variant in (
        "behavior_graph_no_test_feedback_with_test_loss",
        "observational_learning_no_test_feedback_with_test_loss",
        "observational_learning_no_test_feedback_with_train_aux_test_loss",
    )


def prepare_trial(config: TrainConfig, nbcues: int, numtrial: int):
    cues = []
    cue_pairs = []
    correct_order = np.zeros(config.bs)
    adjacent = np.zeros(config.bs)
    requires_choice = np.ones(config.bs, dtype="float32")
    reward_available = np.ones(config.bs, dtype="float32")
    distance_observation = np.zeros(config.bs, dtype="float32")

    is_train_trial = numtrial < config.nbtraintrials
    is_test_trial = not is_train_trial

    for batch_index in range(config.bs):
        if config.variant in ("exact_simple_neo", "n8_fixed"):
            cue_pair = sample_adjacent_pair(nbcues) if is_train_trial else sample_random_nonidentical_pair(nbcues)
        elif config.variant.startswith("behavior_graph"):
            cue_pair = sample_behavior_graph_pair(random_orientation=True) if is_train_trial else sample_random_nonidentical_pair(nbcues)
        elif is_passive_observational_variant(config.variant):
            if is_train_trial:
                # Passive demonstration: no required response and no reward/feedback
                # re-enters during learning. In distance-input v3, the signed distance
                # is the observation signal, so orientation should be randomized;
                # otherwise a trivial "first displayed cue is correct" shortcut remains.
                # If distance input is disabled, preserve the original v2 default
                # correct-first demonstration unless the user explicitly disables it.
                train_random_orientation = (
                    should_provide_distance_input(config, is_train_trial)
                    or use_train_aux_loss(config.variant)
                    or not config.observational_train_orients_correct_first
                )
                cue_pair = sample_behavior_graph_pair(random_orientation=train_random_orientation)
                requires_choice[batch_index] = 0.0
                reward_available[batch_index] = 0.0
            else:
                cue_pair = sample_random_nonidentical_pair(nbcues)
        else:
            raise ValueError(f"Unknown variant: {config.variant}")

        if suppress_test_feedback(config.variant) and is_test_trial:
            reward_available[batch_index] = 0.0

        correct_order[batch_index] = 1 if cue_pair[0] < cue_pair[1] else 0
        adjacent[batch_index] = 1 if abs(cue_pair[0] - cue_pair[1]) == 1 else 0
        if should_provide_distance_input(config, is_train_trial):
            distance_observation[batch_index] = signed_rank_distance(cue_pair, nbcues)
        cue_pairs.append(cue_pair)
        cues.append([cue_pair, nbcues, -1, -1])

    return (
        cues,
        cue_pairs,
        correct_order,
        adjacent,
        requires_choice,
        reward_available,
        distance_observation,
    )


def build_step_inputs(
    config: TrainConfig,
    nbcues: int,
    cue_data,
    cues,
    reward,
    previous_actions,
    distance_observation,
    numstep: int,
    numstep_ep: int,
    include_previous_action,
):
    inputs = np.zeros((config.bs, config.inputsize), dtype="float32")
    include_previous_action = np.asarray(include_previous_action, dtype="float32")

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
        # v3: independent signed rank-distance observation. It is presented only
        # with the cue pair itself, not as reward feedback after the response. It
        # remains zero during no-feedback test when distance_input_train_only=True.
        if isinstance(cue, (list, tuple, np.ndarray)):
            inputs[batch_index, config.nbstimbits + DISTANCE_INPUT_OFFSET] = distance_observation[batch_index]

        if numstep == NUMRESPONSESTEP + 1 and include_previous_action[batch_index] > 0.5:
            inputs[
                batch_index,
                config.nbstimbits + ADDINPUT + previous_actions[batch_index],
            ] = 1

    return torch.from_numpy(inputs).detach().to(DEVICE)


def run_episode(config: TrainConfig, net: RetroModulRNN, nbcues: int, print_trace: bool = False):
    hidden = net.initialZeroState(config.bs)
    et = net.initialZeroET(config.bs)
    pw = net.initialZeroPlasticWeights(config.bs)
    cue_data = generate_cue_data(config, nbcues)

    reward = np.zeros(config.bs, dtype="float32")
    sumrewardtrain = np.zeros(config.bs)
    sumrewardtest = np.zeros(config.bs)
    rewards = []
    values = []
    logprobs = []

    previous_actions = np.zeros(config.bs, dtype="int32")

    nbtraintrials = 0
    nbtraintrials_correct = 0
    nbtesttrials = 0
    nbtesttrials_correct = 0
    nbtesttrials_adjacent = 0
    nbtesttrials_adjacent_correct = 0
    nbtesttrials_nonadjacent = 0
    nbtesttrials_nonadjacent_correct = 0

    loss = 0
    lossv = 0
    train_aux_loss_sum = torch.zeros((), device=DEVICE)
    test_ce_loss_sum = torch.zeros((), device=DEVICE)
    train_aux_steps = 0
    test_ce_steps = 0

    blank_inputs = torch.zeros(config.bs, config.inputsize, requires_grad=False).to(DEVICE)
    for _ in range(2):
        _, _, _, hidden, et, pw = net(blank_inputs, hidden, et, pw)

    numstep_ep = 0
    for numtrial in range(config.nbtrials):
        is_train_trial = numtrial < config.nbtraintrials
        is_test_trial = not is_train_trial

        # Crucial Miconi/simple_neo detail: hidden state and eligibility trace reset every trial;
        # plastic weights persist across the episode.
        hidden = net.initialZeroState(config.bs)
        et = net.initialZeroET(config.bs)

        (
            cues,
            cue_pairs,
            correct_order,
            adjacent,
            requires_choice,
            reward_available,
            distance_observation,
        ) = prepare_trial(config, nbcues, numtrial)
        include_previous_action = requires_choice.copy()
        correct_answer = np.zeros(config.bs)

        for numstep in range(config.triallen):
            inputs = build_step_inputs(
                config,
                nbcues,
                cue_data,
                cues,
                reward,
                previous_actions,
                distance_observation,
                numstep,
                numstep_ep,
                include_previous_action,
            )
            y_raw, value, daout, hidden, et, pw = net(inputs, hidden, et, pw)

            y = torch.softmax(y_raw, dim=1)
            distrib = torch.distributions.Categorical(y)
            actions = distrib.sample()
            logprob = distrib.log_prob(actions)
            # Preserve original behavior for all choice trials: log-prob from every step enters
            # the A2C objective. For passive observation trials, no response is required, so all
            # policy terms are masked out for that trial.
            logprob = logprob * torch.from_numpy(requires_choice).detach().to(DEVICE)
            logprobs.append(logprob)

            # Optional outer-loop supervised losses. Labels are used only in the loss,
            # never as episode inputs. Train auxiliary is restricted to observed train
            # pairs; test CE is the meta-training target for all-pair no-feedback tests.
            if numstep == NUMRESPONSESTEP:
                correct_action = torch.from_numpy(correct_order.astype("int64")).detach().to(DEVICE)
                if is_train_trial and use_train_aux_loss(config.variant):
                    train_aux_loss_sum = train_aux_loss_sum + F.cross_entropy(y_raw, correct_action, reduction="mean")
                    train_aux_steps += 1
                if is_test_trial and use_test_supervised_loss(config.variant):
                    test_ce_loss_sum = test_ce_loss_sum + F.cross_entropy(y_raw, correct_action, reduction="mean")
                    test_ce_steps += 1

            sampled_actions = actions.detach().cpu().numpy()
            previous_actions = sampled_actions.copy()

            if print_trace:
                log_trace(
                    config,
                    numtrial,
                    numstep,
                    inputs,
                    y,
                    previous_actions,
                    correct_order,
                    reward,
                    daout,
                    cues,
                    distance_observation,
                    requires_choice,
                    reward_available,
                )

            reward = np.zeros(config.bs, dtype="float32")
            for batch_index in range(config.bs):
                if numstep == NUMRESPONSESTEP:
                    if requires_choice[batch_index] > 0.5:
                        correct_answer[batch_index] = 1
                        chose_item_1 = previous_actions[batch_index] == 1
                        is_correct = (correct_order[batch_index] and chose_item_1) or (
                            (not correct_order[batch_index]) and not chose_item_1
                        )
                        if is_correct:
                            if reward_available[batch_index] > 0.5:
                                reward[batch_index] += config.rew
                        else:
                            correct_answer[batch_index] = 0
                            if reward_available[batch_index] > 0.5:
                                reward[batch_index] -= config.rew
                    else:
                        # Passive demonstration: no action feedback should be re-entered at step 2.
                        previous_actions[batch_index] = 0
                        correct_answer[batch_index] = 0

            rewards.append(reward)
            values.append(value)
            if is_train_trial:
                sumrewardtrain += reward
            else:
                sumrewardtest += reward

            loss = loss + config.bent * y.pow(2).sum() / config.bs
            numstep_ep += 1

        if is_train_trial:
            # Count train performance only when a choice was actually required.
            train_choice_count = np.sum(requires_choice)
            if train_choice_count > 0:
                nbtraintrials += int(train_choice_count)
                nbtraintrials_correct += np.sum(correct_answer)
        else:
            nbtesttrials += config.bs
            nbtesttrials_correct += np.sum(correct_answer)
            nbtesttrials_adjacent += np.sum(adjacent)
            nbtesttrials_adjacent_correct += np.sum(adjacent * correct_answer)
            nbtesttrials_nonadjacent += np.sum(1 - adjacent)
            nbtesttrials_nonadjacent_correct += np.sum((1 - adjacent) * correct_answer)

    bootstrap_return = torch.zeros(config.bs, requires_grad=False).to(DEVICE)
    for numstepb in reversed(range(config.eplen)):
        bootstrap_return = config.gr * bootstrap_return + torch.from_numpy(rewards[numstepb]).detach().to(DEVICE)
        advantage = bootstrap_return - values[numstepb][:, 0]
        lossv = lossv + advantage.pow(2).sum() / config.bs
        loss_multiplier = config.testlmult if numstepb > config.eplen - config.triallen * config.nbtesttrials else 1.0
        loss = loss - loss_multiplier * (logprobs[numstepb] * advantage.detach()).sum() / config.bs

    loss_objective = float(loss.detach())
    train_aux_loss = train_aux_loss_sum / max(1, train_aux_steps)
    test_ce_loss = test_ce_loss_sum / max(1, test_ce_steps)
    loss = loss + config.blossv * lossv
    loss = loss / config.eplen
    if use_train_aux_loss(config.variant):
        loss = loss + config.train_supervised_loss_weight * train_aux_loss
    if use_test_supervised_loss(config.variant):
        loss = loss + config.test_supervised_loss_weight * test_ce_loss
    loss = loss + torch.mean(pw**2) * config.lpw

    train_perf = None if nbtraintrials == 0 else nbtraintrials_correct / nbtraintrials
    test_perf = None if nbtesttrials == 0 else nbtesttrials_correct / nbtesttrials
    test_perf_adjacent = None
    if nbtesttrials_adjacent > 0:
        test_perf_adjacent = nbtesttrials_adjacent_correct / nbtesttrials_adjacent
    test_perf_nonadjacent = None
    if nbtesttrials_nonadjacent > 0:
        test_perf_nonadjacent = nbtesttrials_nonadjacent_correct / nbtesttrials_nonadjacent

    return EpisodeStats(
        loss=loss,
        loss_value=float(loss.detach()),
        loss_objective=loss_objective,
        train_reward_mean=float(sumrewardtrain.mean()),
        test_reward_mean=float(sumrewardtest.mean()),
        nbtraintrials=nbtraintrials,
        nbtesttrials=nbtesttrials,
        train_perf=train_perf,
        test_perf=test_perf,
        test_perf_adjacent=test_perf_adjacent,
        test_perf_nonadjacent=test_perf_nonadjacent,
        train_aux_loss=float(train_aux_loss.detach()),
        test_ce_loss=float(test_ce_loss.detach()),
        final_pw=pw.detach(),
    )


def log_trace(
    config,
    numtrial,
    numstep,
    inputs,
    y,
    actions,
    correct_order,
    reward,
    daout,
    cues,
    distance_observation,
    requires_choice,
    reward_available,
):
    log(
        "Tr {} Step {} Cue1(0): {} Cue2(0): {} Other inputs: {}\n"
        " - Outputs(0): {} - action chosen(0): {} TrialLen: {} numstep {} "
        "TTHCC(0): {} Reward(prev): {} DistanceObs: {} DAout: {} cues(0): {} "
        "requires_choice(0): {} reward_available(0): {}".format(
            numtrial,
            numstep,
            inputs[0, : config.cs].detach().cpu().numpy(),
            inputs[0, config.cs : 2 * config.cs].detach().cpu().numpy(),
            inputs[0, 2 * config.cs :].detach().cpu().numpy(),
            y.detach().cpu().numpy()[0, :],
            actions[0],
            config.triallen,
            numstep,
            correct_order[0],
            reward[0],
            distance_observation[0],
            float(daout[0].detach()),
            cues[0],
            requires_choice[0],
            reward_available[0],
        )
    )


def print_episode_summary(config: TrainConfig, episode_index: int, stats: EpisodeStats, start_time: float):
    elapsed = time.time() - start_time
    log(f"Episode {episode_index} [{config.variant}] ====")
    log(f"Time spent on last {config.pe} iters: {elapsed:.2f}s")
    log(f"Mean loss: {stats.loss_value:.6f}; objective: {stats.loss_objective:.6f}")
    log(
        "Train perf: {} | train reward mean: {:.3f}".format(
            "N/A" if stats.train_perf is None else f"{stats.train_perf:.3f}",
            stats.train_reward_mean,
        )
    )
    log(
        "Test performance: {} | adjacent: {} | nonadjacent: {} | test reward mean: {:.3f}".format(
            "N/A" if stats.test_perf is None else f"{stats.test_perf:.3f}",
            "N/A" if stats.test_perf_adjacent is None else f"{stats.test_perf_adjacent:.3f}",
            "N/A" if stats.test_perf_nonadjacent is None else f"{stats.test_perf_nonadjacent:.3f}",
            stats.test_reward_mean,
        )
    )
    log(f"Aux losses: train_aux={stats.train_aux_loss:.6f} | test_ce={stats.test_ce_loss:.6f}")
    pw = stats.final_pw
    log(f"mean-abs pw: {float(torch.mean(torch.abs(pw))):.6f}")


def save_checkpoint(config: TrainConfig, net: RetroModulRNN, output_dir: Path, test_rewards):
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), output_dir / ("netAE" + str(config.rngseed) + ".dat"))
    torch.save(net.state_dict(), output_dir / "net.dat")
    with open(output_dir / ("tAE" + str(config.rngseed) + ".txt"), "w") as thefile:
        for item in test_rewards[::10]:
            thefile.write(f"{item}\n")
    log(f"[save] Wrote checkpoint and test-reward log to {output_dir}")


def append_csv(csv_path: Path, episode_index: int, stats: EpisodeStats):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "episode": episode_index,
        "loss": stats.loss_value,
        "loss_objective": stats.loss_objective,
        "train_reward_mean": stats.train_reward_mean,
        "test_reward_mean": stats.test_reward_mean,
        "train_perf": "" if stats.train_perf is None else stats.train_perf,
        "test_perf": "" if stats.test_perf is None else stats.test_perf,
        "test_perf_adjacent": "" if stats.test_perf_adjacent is None else stats.test_perf_adjacent,
        "test_perf_nonadjacent": "" if stats.test_perf_nonadjacent is None else stats.test_perf_nonadjacent,
        "train_aux_loss": stats.train_aux_loss,
        "test_ce_loss": stats.test_ce_loss,
        "mean_abs_pw": float(torch.mean(torch.abs(stats.final_pw))),
    }
    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def train(config: TrainConfig, output_dir: Path, trace_steps: bool = False):
    model_config = config.to_model_dict()
    net = RetroModulRNN(model_config)
    optimizer = torch.optim.Adam(net.parameters(), lr=config.lr, eps=config.eps, weight_decay=config.l2)
    test_rewards = []

    log(f"[setup] Script version: {SCRIPT_VERSION}")
    log(f"[setup] Device: {DEVICE}; torch_num_threads={torch.get_num_threads()}")
    log(f"[setup] Variant: {config.variant}")
    log(f"[setup] Batch size: {config.bs}; episodes: {config.nbiter}; output: {output_dir}")
    log(
        f"[setup] hs={config.hs}; cs={config.cs}; train_trials={config.nbtraintrials}; "
        f"test_trials={config.nbtesttrials}; triallen={config.triallen}"
    )
    if is_behavior_graph_variant(config.variant):
        log(f"[setup] behavior graph n=8 train pairs: {BEHAVIOR_GRAPH_N8}")
    log(
        f"[setup] train_aux_loss={use_train_aux_loss(config.variant)} "
        f"w={config.train_supervised_loss_weight}; "
        f"test_supervised_loss={use_test_supervised_loss(config.variant)} "
        f"w={config.test_supervised_loss_weight}; "
        f"suppress_test_feedback={suppress_test_feedback(config.variant)}"
    )
    log(
        f"[setup] distance_input={config.distance_input}; "
        f"train_only={config.distance_input_train_only}; "
        f"distance_slot=nbstimbits+{DISTANCE_INPUT_OFFSET}"
    )
    log(f"[setup] Parameter shapes: {[x.size() for x in net.parameters()]}")

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.json", "w") as f:
        # Avoid non-JSON range by using scalar fields only.
        cfg_dict = asdict(config)
        cfg_dict["device"] = DEVICE
        f.write(str(cfg_dict) + "\n")

    trace_start = time.time()
    episode_iter = tqdm(range(config.nbiter), desc="training episodes", unit="episode", dynamic_ncols=True, file=sys.stdout)
    for episode_index in episode_iter:
        should_print_summary = episode_index % config.pe == 0 or episode_index == config.nbiter - 1
        print_trace = trace_steps and should_print_summary
        nbcues = choose_nbcues_for_episode(config)

        optimizer.zero_grad()
        stats = run_episode(config, net, nbcues=nbcues, print_trace=print_trace)
        stats.loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), config.gc)
        if episode_index > 100:
            optimizer.step()

        test_rewards.append(stats.test_reward_mean)
        if should_print_summary:
            print_episode_summary(config, episode_index, stats, trace_start)
            trace_start = time.time()
            if config.write_csv:
                append_csv(output_dir / "train_log.csv", episode_index, stats)

        if config.save_every > 0 and episode_index % config.save_every == 0 and episode_index > 0:
            save_checkpoint(config, net, output_dir, test_rewards)

    return net


def parse_args():
    parser = argparse.ArgumentParser(description="Stepwise simple_neo mutant suite.")
    parser.add_argument(
        "--variant",
        choices=[
            "exact_simple_neo",
            "n8_fixed",
            "behavior_graph_rewarded",
            "behavior_graph_no_test_feedback",
            "behavior_graph_no_test_feedback_with_test_loss",
            "observational_learning",
            "observational_learning_no_test_feedback",
            "observational_learning_no_test_feedback_with_train_aux",
            "observational_learning_no_test_feedback_with_test_loss",
            "observational_learning_no_test_feedback_with_train_aux_test_loss",
        ],
        default="exact_simple_neo",
    )
    parser.add_argument("--nbiter", type=int, default=30000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--print-every", type=int, default=101)
    parser.add_argument("--output-dir", default=str(ROOT_DIR))
    parser.add_argument("--hidden-size", type=int, default=200)
    parser.add_argument("--cue-size", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lpw", type=float, default=1e-4)
    parser.add_argument("--nbtraintrials", type=int, default=20)
    parser.add_argument("--nbtesttrials", type=int, default=10)
    parser.add_argument("--testlmult", type=float, default=3.0)
    parser.add_argument("--train-supervised-loss-weight", type=float, default=1.0)
    parser.add_argument("--test-supervised-loss-weight", type=float, default=1.0)
    parser.add_argument("--nmin", type=int, default=4)
    parser.add_argument("--nmax", type=int, default=8)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument(
        "--no-distance-input",
        action="store_true",
        help="Disable signed rank-distance input for behavioral learning trials.",
    )
    parser.add_argument(
        "--distance-input-all-phases",
        action="store_true",
        help="Also expose signed distance during test trials; normally keep this off to preserve no-feedback test.",
    )
    parser.add_argument("--no-csv", action="store_true")
    parser.add_argument("--trace-steps", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    nmin = args.nmin
    nmax = args.nmax
    if args.variant == "n8_fixed":
        nmin = nmax = 8
    elif args.variant.startswith("behavior_graph") or args.variant.startswith("observational_learning"):
        nmin = nmax = 8

    config = TrainConfig(
        rngseed=args.seed,
        variant=args.variant,
        bs=args.batch_size,
        nbiter=args.nbiter,
        save_every=args.save_every,
        pe=args.print_every,
        hs=args.hidden_size,
        cs=args.cue_size,
        lr=args.lr,
        lpw=args.lpw,
        nbtraintrials=args.nbtraintrials,
        nbtesttrials=args.nbtesttrials,
        testlmult=args.testlmult,
        train_supervised_loss_weight=args.train_supervised_loss_weight,
        test_supervised_loss_weight=args.test_supervised_loss_weight,
        nbcues_min=nmin,
        nbcues_max=nmax,
        write_csv=not args.no_csv,
        num_threads=args.num_threads,
        distance_input=not args.no_distance_input,
        distance_input_train_only=not args.distance_input_all_phases,
    )
    np.set_printoptions(precision=5)
    if config.num_threads and config.num_threads > 0:
        torch.set_num_threads(config.num_threads)
    set_seed(config.rngseed)
    train(config, Path(args.output_dir), trace_steps=args.trace_steps)


if __name__ == "__main__":
    main()
