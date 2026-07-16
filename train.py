from model import *


class TrainGenerator(InputsGenerator):
    def _generate_train_trials(self):
        def invalid(train_trials):
            e = [[] for _ in range(8)]
            for i in range(8):
                x, y = train_trials[i * 2], train_trials[i * 2 + 1]
                e[x].append(y)
                e[y].append(x)
            vis = [False] * 8
            p = 0
            while not vis[p]:
                vis[p] = True
                p = e[p][1] if vis[e[p][0]] else e[p][0]
            return not all(vis)

        train_trials = [i // 2 for i in range(NBCUES * 2)]
        while invalid(train_trials):
            random.shuffle(train_trials)
        return np.array(
            [[train_trials[i * 2], train_trials[i * 2 + 1]] for i in range(NBCUES)]
        )

    def _generate_test_trials(self, config):
        test_trials = np.ndarray((config.nbtesttrials, 2), dtype=np.int64)
        for i in range(config.nbtesttrials):
            x, y = 0, 0
            while x == y:
                x, y = random.randint(0, NBCUES - 1), random.randint(0, NBCUES - 1)
            test_trials[i, 0], test_trials[i, 1] = x, y
        return test_trials


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
    previous_actions = np.zeros(config.bs, dtype="int64")

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

    gen = TrainGenerator(config)

    numstep_ep = 0
    for numtrial in range(config.nbtrials):
        hidden = net.initialZeroState(config.bs)
        et = net.initialZeroET(config.bs)
        cue_pairs, correct_order, adjacent = gen.prepare_trial(config, nbtrials)
        istest_thisep[:, numtrial] = 1 if numtrial >= config.nbtraintrials else 0

        correct_answer = np.zeros(config.bs)

        for numstep in range(config.triallen):
            inputs = gen.build_step_inputs(
                config,
                cue_data,
                cue_pairs,
                numstep,
                numstep_ep,
            )
            y_raw, value, daout, hidden, et, pw = net(inputs, hidden, et, pw)

            y = torch.softmax(y_raw, dim=1)
            distrib = torch.distributions.Categorical(y)
            actions = distrib.sample()
            logprobs.append(distrib.log_prob(actions))
            previous_actions = actions.detach().cpu().numpy()

            reward = np.zeros(config.bs, dtype="float32")
            for batch_index in range(config.bs):
                if numtrial >= config.nbtraintrials and numstep == 1:
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


def print_episode_summary(config, episode_index, stats, start_time):
    elapsed = time.time() - start_time
    log(
        f"{episode_index}, {elapsed:.2f}, {stats.loss_value:.6f}, "
        "{}, {}, {}".format(
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


def save_checkpoint(config, net, output_dir, test_rewards, episode_index):
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        net.state_dict(), output_dir / f"net_{config.rngseed}_{episode_index}.dat"
    )


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
    log("")
    log("episode_index, elapsed, loss, perf, perf_adj, perf_nonadj")

    trace_start = time.time()
    episode_iter = tqdm(
        range(config.nbiter),
        desc="training episodes",
        unit="episode",
        dynamic_ncols=True,
        file=sys.stdout,
    )
    for episode_index in episode_iter:
        should_print_summary = (episode_index + 1) % config.pe == 0
        print_trace = trace_steps and should_print_summary
        nbcues = NBCUES

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

        if (episode_index + 1) % config.save_every == 0 and episode_index > 0:
            save_checkpoint(config, net, output_dir, test_rewards, episode_index)

    return net


def parse_args():
    parser = argparse.ArgumentParser(
        description="Readable training script for the plastic RNN transitive-inference task."
    )
    parser.add_argument("--nbiter", type=int, default=30000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--print-every", type=int, default=50)
    parser.add_argument("--output-dir", default="models")
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
