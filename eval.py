import matplotlib.pyplot as plt
from scipy.stats import beta
from model import *


class TestGenerator(InputsGenerator):
    def _generate_train_trials(self):
        return np.array(
            [[0, 5], [1, 2], [1, 4], [2, 6], [3, 5], [3, 6], [4, 7], [0, 7]]
        )

    def _generate_test_trials(self, config):
        test_trials = np.array([[i, j] for i in range(NBCUES) for j in range(i)])
        order = np.ndarray((config.nbtesttrials, 2), dtype=np.int64)
        for i in range(TEST_REPEATS):
            self._shuffle(test_trials)
            order[i * NBPAIRS : (i + 1) * NBPAIRS] = test_trials
        return order


def evalution(config, net: nn.Module):
    nbcues = NBCUES
    was_training = net.training
    net.eval()

    cue_data = generate_cue_data(config, nbcues)
    gen = TestGenerator(config)

    hidden = net.initialZeroState(config.bs)
    et = net.initialZeroET(config.bs)
    pw = net.initialZeroPlasticWeights(config.bs)

    output = np.ndarray((config.bs, config.nbtesttrials))
    correct_thisep = np.zeros((config.bs, config.nbtesttrials))
    nbtrials = np.zeros(config.bs)
    previous_actions = np.zeros(config.bs, dtype="int64")

    # Two blank steps before the episode, matching the original script.
    blank_inputs = torch.zeros(config.bs, config.inputsize, requires_grad=False).to(
        DEVICE
    )
    for _ in range(2):
        _, _, _, hidden, et, pw = net(blank_inputs, hidden, et, pw)

    numstep_ep = 0
    for numtrial in range(config.nbtrials):
        print(f"[TRIAL] {numtrial}", end="\r")
        hidden = net.initialZeroState(config.bs)
        et = net.initialZeroET(config.bs)
        cue_pairs, correct_order, _ = gen.prepare_trial(config, nbtrials)

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
            actions = distrib.sample() if config.sample == 1 else torch.argmax(y, dim=1)
            previous_actions = actions.detach().cpu().numpy()

            for batch_index in range(config.bs):
                numtesttrial = numtrial - config.nbtraintrials
                if numtesttrial >= 0 and numstep == 1:
                    chose_item_1 = previous_actions[batch_index] == 1
                    correct = int(correct_order[batch_index] == chose_item_1)
                    correct_thisep[batch_index, numtesttrial] = correct
                    output[batch_index, numtesttrial] = y[
                        batch_index, correct_order[batch_index]
                    ]

                if numstep == config.triallen - 1:
                    nbtrials[batch_index] += 1

            numstep_ep += 1

    if was_training:
        net.train()
    return gen.orders, correct_thisep, output


def generate_result(config: TrainConfig, orders, is_correct, correction):
    position_correct = is_correct.sum(axis=0)
    print("[POSITION CORRECT]")
    for position in range(config.nbtesttrials):
        print(
            position_correct[position],
            end="\n" if (position + 1) % NBPAIRS == 0 else " ",
        )
    corr_mat = np.corrcoef(position_correct, np.arange(config.nbtesttrials))
    print(f"[correlation] {corr_mat[0,1]:.2f}")

    pair_correct = np.zeros((config.bs, NBCUES, NBCUES), dtype=np.int64)
    pair_count = np.zeros((config.bs, NBCUES, NBCUES), dtype=np.int64)
    for batch_index in range(config.bs):
        for trial_index in range(config.nbtesttrials):
            pair = orders[batch_index, trial_index + config.nbtraintrials]
            index = (batch_index, max(pair), min(pair))
            pair_correct[index] += is_correct[batch_index, trial_index]
            pair_count[index] += 1
    print("[PAIR CORRECT]")
    for i in range(1, NBCUES):
        for j in range(i):
            print(pair_correct[:, i, j].sum(), end=" ")
        print()

    print("[BETA DISTURBUTION]")
    for i in range(1, NBCUES):
        for j in range(i):
            data = (
                pair_correct[:, i, j] / TEST_REPEATS * (config.bs - 1) + 0.5
            ) / config.bs
            a, b, _, _ = beta.fit(data, floc=0, fscale=1)
            if min(a, b) > 10:
                print("(>10, >10)", end="\t")
            else:
                print(f"({a:.1f}, {b:.1f})", end="\t")
        print()

    """fig, axes = plt.subplots(
        nrows=NBCUES - 1, ncols=NBCUES - 1, sharex=True, sharey=True
    )
    for i in range(1, NBCUES):
        for j in range(i):
            cnt = np.bincount(pair_correct[:, i, j], minlength=TEST_REPEATS + 1)
            axes[i - 1, j].plot(cnt / cnt.max())
    plt.tight_layout()
    plt.show()"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="net.dat")
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample", type=int, default=0)
    # parser.add_argument("--output-dir", default="figures")
    return parser.parse_args()


def main():
    args = parse_args()
    config = TrainConfig(rngseed=args.seed, bs=args.batch_size, sample=args.sample)
    np.set_printoptions(precision=5)
    set_seed(config.rngseed)
    net = RetroModulRNN(config.to_model_dict())
    net.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
    with torch.no_grad():
        orders, is_correct, correction = evalution(config, net)
    generate_result(config, orders, is_correct, correction)


if __name__ == "__main__":
    main()
