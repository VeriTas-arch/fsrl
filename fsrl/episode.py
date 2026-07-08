from dataclasses import dataclass

import numpy as np
import torch

from .config import DEVICE, NUMRESPONSESTEP
from .logging import log
from .task import build_step_inputs, generate_cue_data, prepare_trial


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
