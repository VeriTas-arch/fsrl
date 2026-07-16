import argparse
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from tqdm.auto import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEST_REPEATS = 10
TRAIN_REPEATS = 4
NBCUES = 8
NBPAIRS = NBCUES * (NBCUES - 1) // 2
MAXTILE = 10


def log(message):
    print(message, flush=True)


@dataclass
class TrainConfig:
    rngseed: int = 0
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
    pe: int = 50
    cs: int = 15
    triallen: int = 4
    nbtraintrials: int = TRAIN_REPEATS * NBCUES
    nbtesttrials: int = TEST_REPEATS * NBPAIRS
    sample: int = 0
    testlmult: float = 3.0
    l2: float = 0.0
    lr: float = 1e-4
    lpw: float = 1e-4

    @property
    def nbtrials(self):
        return self.nbtraintrials + self.nbtesttrials

    @property
    def eplen(self):
        return self.nbtrials * self.triallen

    @property
    def nbstimbits(self):
        return 2 * self.cs + 2

    @property
    def outputsize(self):
        return 2

    @property
    def inputsize(self):
        return self.nbstimbits

    def to_model_dict(self):
        """Return the dict shape expected by the original model code."""
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
    """RNN with neuromodulated recurrent plasticity.

    ``et`` is the Hebbian eligibility trace. ``pw`` is the within-episode
    plastic recurrent weight matrix. Only ``pw`` changes during an episode.
    """

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
        self.etaet = torch.nn.Parameter(
            (0.7 * torch.ones(1)).to(DEVICE), requires_grad=True
        )
        self.DAmult = torch.nn.Parameter(
            (1.0 * torch.ones(1)).to(DEVICE), requires_grad=True
        )
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
        return torch.zeros(
            batch_size, self.GG["hs"], self.GG["hs"], requires_grad=False
        ).to(DEVICE)

    def initialZeroPlasticWeights(self, batch_size):
        return torch.zeros(
            batch_size, self.GG["hs"], self.GG["hs"], requires_grad=False
        ).to(DEVICE)

    def initialZeroState(self, batch_size):
        return torch.zeros(batch_size, self.GG["hs"], requires_grad=False).to(DEVICE)


@dataclass
class EpisodeStats:
    loss: torch.Tensor
    loss_value: float
    loss_objective: float
    test_reward_mean: float
    nbtesttrials: int
    test_perf: float | None
    test_perf_adjacent: float | None
    test_perf_nonadjacent: float | None
    final_pw: torch.Tensor


def set_seed(seed):
    if seed < 0:
        log("[setup] No random seed.")
        return
    log(f"[setup] Setting random seed {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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


class InputsGenerator:
    def _shuffle(self, trials):
        np.random.shuffle(trials)
        for pair in trials:
            np.random.shuffle(pair)

    def __init__(self, config):
        self.orders = np.ndarray((config.bs, config.nbtrials, 2), dtype=np.int64)
        for batch in range(config.bs):
            order = np.ndarray((config.nbtrials, 2), dtype=np.int64)
            train_trials = self._generate_train_trials()
            for i in range(TRAIN_REPEATS):
                self._shuffle(train_trials)
                order[i * NBCUES : (i + 1) * NBCUES] = train_trials
            order[config.nbtraintrials :] = self._generate_test_trials(config)
            self.orders[batch] = order

    def prepare_trial(self, config, nbtrials):
        cue_pairs = []
        correct_order = np.zeros(config.bs, dtype=np.int64)
        adjacent = np.zeros(config.bs, dtype=np.int64)

        for batch_index in range(config.bs):
            cue_pair = self.orders[batch_index][int(nbtrials[batch_index])]
            assert nbtrials[batch_index] == int(nbtrials[batch_index])

            correct_order[batch_index] = 1 if cue_pair[0] < cue_pair[1] else 0
            adjacent[batch_index] = 1 if abs(cue_pair[0] - cue_pair[1]) == 1 else 0
            cue_pairs.append(cue_pair)

        return cue_pairs, correct_order, adjacent

    def build_step_inputs(
        self,
        config,
        cue_data,
        cue_pairs,
        numstep,
        numstep_ep,
    ):
        inputs = np.zeros((config.bs, config.inputsize), dtype="float32")
        for batch_index in range(config.bs):
            cue = cue_pairs[batch_index]
            if numstep == 0:
                inputs[batch_index, :-2] = np.concatenate(
                    (cue_data[batch_index][cue[0]][:], cue_data[batch_index][cue[1]][:])
                )
                if numstep_ep < config.nbtraintrials * config.triallen:
                    d = -min(cue) + random.randint(0, MAXTILE - abs(cue[0] - cue[1]))
                    inputs[batch_index, -2] = cue[0] + d
                    inputs[batch_index, -1] = cue[1] + d
        return torch.from_numpy(inputs).detach().to(DEVICE)
