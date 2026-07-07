"""
Readable training script for the plastic RNN transitive-inference task.

This file keeps the learning rule, task structure, and saved model format of
``simple.py`` while organizing the code into small functions for learning.
"""

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from tqdm.auto import tqdm

ROOT_DIR = Path(__file__).resolve().parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ADDINPUT = 4
NUMRESPONSESTEP = 1


def log(message):
    print(message, flush=True)


@dataclass
class TrainConfig:
    rngseed: int = -1
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


def run_episode(config, net, nbcues, print_trace=False):
    """Run one full episode and return the differentiable training loss."""
    hidden = net.initialZeroState(config.bs)
    et = net.initialZeroET(config.bs)
    pw = net.initialZeroPlasticWeights(config.bs)
    cue_data = generate_cue_data(config, nbcues)

    reward = np.zeros(config.bs, dtype="float32")
    sumrewardtest = np.zeros(config.bs)
    rewards = []
    values = []
    logprobs = []

    correct_thisep = np.zeros((config.bs, config.nbtrials))
    istest_thisep = np.zeros((config.bs, config.nbtrials))
    nbtrials = np.zeros(config.bs)
    previous_actions = np.zeros(config.bs, dtype="int32")

    nbtesttrials = 0
    nbtesttrials_correct = 0
    nbtesttrials_adjacent = 0
    nbtesttrials_adjacent_correct = 0
    nbtesttrials_nonadjacent = 0
    nbtesttrials_nonadjacent_correct = 0

    loss = 0
    lossv = 0

    # Two blank steps before the episode, matching the original script.
    blank_inputs = torch.zeros(config.bs, config.inputsize, requires_grad=False).to(
        DEVICE
    )
    for _ in range(2):
        _, _, _, hidden, et, pw = net(blank_inputs, hidden, et, pw)

    numstep_ep = 0
    for numtrial in range(config.nbtrials):
        hidden = net.initialZeroState(config.bs)
        et = net.initialZeroET(config.bs)
        cues, cue_pairs, correct_order, adjacent = prepare_trial(
            config, nbcues, nbtrials
        )
        istest_thisep[:, numtrial] = 1 if numtrial >= config.nbtraintrials else 0

        correct_answer = np.zeros(config.bs)

        for numstep in range(config.triallen):
            inputs = build_step_inputs(
                config,
                nbcues,
                cue_data,
                cues,
                reward,
                previous_actions,
                numstep,
                numstep_ep,
            )
            y_raw, value, daout, hidden, et, pw = net(inputs, hidden, et, pw)

            y = torch.softmax(y_raw, dim=1)
            distrib = torch.distributions.Categorical(y)
            actions = distrib.sample()
            logprobs.append(distrib.log_prob(actions))
            previous_actions = actions.detach().cpu().numpy()

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
                )

            reward = np.zeros(config.bs, dtype="float32")
            for batch_index in range(config.bs):
                if numstep == NUMRESPONSESTEP:
                    correct_answer[batch_index] = 1
                    chose_item_1 = previous_actions[batch_index] == 1
                    if (correct_order[batch_index] and chose_item_1) or (
                        (not correct_order[batch_index]) and not chose_item_1
                    ):
                        reward[batch_index] += config.rew
                    else:
                        reward[batch_index] -= config.rew
                        correct_answer[batch_index] = 0
                    correct_thisep[batch_index, numtrial] = correct_answer[batch_index]

                if numstep == config.triallen - 1:
                    nbtrials[batch_index] += 1

            rewards.append(reward)
            values.append(value)
            if numtrial >= config.nbtrials - config.nbtesttrials:
                sumrewardtest += reward

            loss = loss + config.bent * y.pow(2).sum() / config.bs
            numstep_ep += 1

        if numtrial >= config.nbtrials - config.nbtesttrials:
            sumrewardtest += reward
            nbtesttrials += config.bs
            nbtesttrials_correct += np.sum(correct_answer)
            nbtesttrials_adjacent += np.sum(adjacent)
            nbtesttrials_adjacent_correct += np.sum(adjacent * correct_answer)
            nbtesttrials_nonadjacent += np.sum(1 - adjacent)
            nbtesttrials_nonadjacent_correct += np.sum((1 - adjacent) * correct_answer)

    bootstrap_return = torch.zeros(config.bs, requires_grad=False).to(DEVICE)
    for numstepb in reversed(range(config.eplen)):
        bootstrap_return = config.gr * bootstrap_return + torch.from_numpy(
            rewards[numstepb]
        ).detach().to(DEVICE)
        advantage = bootstrap_return - values[numstepb][:, 0]
        lossv = lossv + advantage.pow(2).sum() / config.bs
        loss_multiplier = (
            config.testlmult
            if numstepb > config.eplen - config.triallen * config.nbtesttrials
            else 1.0
        )
        loss = (
            loss
            - loss_multiplier
            * (logprobs[numstepb] * advantage.detach()).sum()
            / config.bs
        )

    loss_objective = float(loss.detach())
    loss = loss + config.blossv * lossv
    loss = loss / config.eplen
    loss = loss + torch.mean(pw**2) * config.lpw

    test_perf = None if nbtesttrials == 0 else nbtesttrials_correct / nbtesttrials
    test_perf_adjacent = None
    if nbtesttrials_adjacent > 0:
        test_perf_adjacent = nbtesttrials_adjacent_correct / nbtesttrials_adjacent
    test_perf_nonadjacent = None
    if nbtesttrials_nonadjacent > 0:
        test_perf_nonadjacent = (
            nbtesttrials_nonadjacent_correct / nbtesttrials_nonadjacent
        )

    return EpisodeStats(
        loss=loss,
        loss_value=float(loss.detach()),
        loss_objective=loss_objective,
        test_reward_mean=float(sumrewardtest.mean()),
        nbtesttrials=nbtesttrials,
        test_perf=test_perf,
        test_perf_adjacent=test_perf_adjacent,
        test_perf_nonadjacent=test_perf_nonadjacent,
        final_pw=pw.detach(),
    )


def log_trace(
    config, numtrial, numstep, inputs, y, actions, correct_order, reward, daout, cues
):
    log(
        "Tr {} Step {} Cue1(0): {} Cue2(0): {} Other inputs: {}\n"
        " - Outputs(0): {} - action chosen(0): {} TrialLen: {} numstep {} "
        "TTHCC(0): {} Reward(prev): {} DAout: {} cues(0): {}".format(
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
            float(daout[0].detach()),
            cues[0],
        )
    )


def print_episode_summary(config, episode_index, stats, start_time):
    elapsed = time.time() - start_time
    log(f"Episode {episode_index} ====")
    log(f"Time spent on last {config.pe} iters: {elapsed:.2f}s")
    log(f"Mean loss: {stats.loss_value:.6f}")
    log(
        "Test performance: {} | adjacent: {} | nonadjacent: {}".format(
            "N/A" if stats.test_perf is None else f"{stats.test_perf:.3f}",
            (
                "N/A"
                if stats.test_perf_adjacent is None
                else f"{stats.test_perf_adjacent:.3f}"
            ),
            (
                "N/A"
                if stats.test_perf_nonadjacent is None
                else f"{stats.test_perf_nonadjacent:.3f}"
            ),
        )
    )
    pw = stats.final_pw
    log(f"mean-abs pw: {float(torch.mean(torch.abs(pw))):.6f}")


def save_checkpoint(config, net, output_dir, test_rewards):
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), output_dir / ("netAE" + str(config.rngseed) + ".dat"))
    torch.save(net.state_dict(), output_dir / "net.dat")
    with open(output_dir / ("tAE" + str(config.rngseed) + ".txt"), "w") as thefile:
        for item in test_rewards[::10]:
            thefile.write(f"{item}\n")
    log(f"[save] Wrote checkpoint and test-reward log to {output_dir}")


def train(config, output_dir, trace_steps=False):
    model_config = config.to_model_dict()
    net = RetroModulRNN(model_config)
    optimizer = torch.optim.Adam(
        net.parameters(), lr=config.lr, eps=config.eps, weight_decay=config.l2
    )
    test_rewards = []

    log(f"[setup] Device: {DEVICE}")
    log(
        f"[setup] Batch size: {config.bs}; episodes: {config.nbiter}; output: {output_dir}"
    )
    log(f"[setup] Parameter shapes: {[x.size() for x in net.parameters()]}")

    trace_start = time.time()
    episode_iter = tqdm(
        range(config.nbiter),
        desc="training episodes",
        unit="episode",
        dynamic_ncols=True,
        file=sys.stdout,
    )
    for episode_index in episode_iter:
        should_print_summary = episode_index % config.pe == 0
        print_trace = trace_steps and should_print_summary
        nbcues = np.random.choice(list(config.nbcuesrange))

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

        if episode_index % config.save_every == 0 and episode_index > 0:
            save_checkpoint(config, net, output_dir, test_rewards)

    return net


def parse_args():
    parser = argparse.ArgumentParser(
        description="Readable training script for the plastic RNN transitive-inference task."
    )
    parser.add_argument("--nbiter", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--print-every", type=int, default=101)
    parser.add_argument("--output-dir", default=str(ROOT_DIR))
    parser.add_argument(
        "--trace-steps",
        action="store_true",
        help="Print per-step debugging details on summary episodes.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = TrainConfig(
        rngseed=args.seed,
        bs=args.batch_size,
        nbiter=args.nbiter,
        save_every=args.save_every,
        pe=args.print_every,
    )
    np.set_printoptions(precision=5)
    set_seed(config.rngseed)
    train(config, Path(args.output_dir), trace_steps=args.trace_steps)


if __name__ == "__main__":
    main()
