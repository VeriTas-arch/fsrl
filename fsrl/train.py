import sys
import time

import numpy as np
import torch
from tqdm.auto import tqdm

from .config import DEVICE
from .episode import run_episode
from .logging import log
from .model import RetroModulRNN


def set_seed(seed):
    if seed < 0:
        log("[setup] No random seed.")
        return
    log(f"[setup] Setting random seed {seed}")
    np.random.seed(seed)
    torch.manual_seed(seed)


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
