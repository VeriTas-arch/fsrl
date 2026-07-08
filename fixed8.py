import itertools
import numpy as np
import torch
from pathlib import Path

from simple_neo import (
    RetroModulRNN,
    TrainConfig,
    DEVICE,
    generate_cue_data,
    set_seed,
    NUMRESPONSESTEP,
)

FIXED_PAIRS = [(0, 5), (1, 2), (1, 4), (2, 6), (3, 5), (3, 6), (4, 7), (0, 7)]
ALL_PAIRS = list(itertools.combinations(range(8), 2))

NUM_PARTICIPANTS = 100
NUM_BLOCKS = 4
REPEATS_PER_PAIR = 10


def build_trial_inputs(config, nbcues, cue_data, cues, reward,
                       previous_action, time_enc, step_in_trial):
    inputs = np.zeros((config.bs, config.inputsize), dtype="float32")
    for batch_index in range(config.bs):
        cue = cues[batch_index][step_in_trial]
        if isinstance(cue, (list, tuple, np.ndarray)):
            inputs[batch_index, : config.nbstimbits - 1] = np.concatenate(
                (cue_data[batch_index][cue[0]][:], cue_data[batch_index][cue[1]][:])
            )
        elif cue == nbcues:
            inputs[batch_index, config.nbstimbits - 1] = 1

        inputs[batch_index, config.nbstimbits + 0] = 1.0
        inputs[batch_index, config.nbstimbits + 1] = time_enc
        inputs[batch_index, config.nbstimbits + 2] = reward[batch_index]

        if step_in_trial == NUMRESPONSESTEP + 1:
            inputs[
                batch_index,
                config.nbstimbits + 4 + previous_action[batch_index],
            ] = 1

    return torch.from_numpy(inputs).detach().to(DEVICE)


def run_single_participant(net, config, cue_data_full, participant_seed):
    set_seed(participant_seed)

    original_bs = config.bs
    config.bs = 1
    cue_data = [cue_data_full[0]]
    batch_size = config.bs
    nbcues = 8

    pw = net.initialZeroPlasticWeights(batch_size)

    train_sequence = []
    rng_train = np.random.RandomState(participant_seed)
    for _ in range(NUM_BLOCKS):
        block = FIXED_PAIRS.copy()
        rng_train.shuffle(block)
        train_sequence.extend(block)

    test_sequence = ALL_PAIRS * REPEATS_PER_PAIR
    rng_test = np.random.RandomState(participant_seed + 9999)
    rng_test.shuffle(test_sequence)

    total_trials = len(train_sequence) + len(test_sequence)
    total_steps = total_trials * config.triallen

    global_step = 0
    reward = np.zeros(batch_size, dtype="float32")
    previous_action = np.zeros(batch_size, dtype="int32")

    pair_correct_counts = {pair: 0 for pair in ALL_PAIRS}
    pair_total_counts = {pair: 0 for pair in ALL_PAIRS}

    for trial_idx, (a, b) in enumerate(train_sequence + test_sequence):
        is_test = trial_idx >= len(train_sequence)
        if is_test:
            reward[:] = 0.0

        hidden = net.initialZeroState(batch_size)
        et = net.initialZeroET(batch_size)
        previous_action.fill(0)

        if (rng_test if is_test else rng_train).rand() < 0.5:
            i, j = b, a
        else:
            i, j = a, b
        cues = [[[i, j], nbcues, -1, -1]]
        correct_is_item1 = 1 if i < j else 0

        for step_in_trial in range(config.triallen):
            time_enc = global_step / total_steps
            inputs = build_trial_inputs(
                config, nbcues, cue_data, cues, reward,
                previous_action, time_enc, step_in_trial
            )
            y_raw, value, daout, hidden, et, pw = net(inputs, hidden, et, pw)

            if step_in_trial == NUMRESPONSESTEP:
                y = torch.softmax(y_raw, dim=1)
                if is_test:
                    reward[0] = 0.0
                    chose_item1 = (torch.argmax(y, dim=1).item() == 1)
                    previous_action[0] = 1 if chose_item1 else 0

                    pair = (a, b)
                    pair_total_counts[pair] += 1
                    if (correct_is_item1 and chose_item1) or \
                       (not correct_is_item1 and not chose_item1):
                        pair_correct_counts[pair] += 1
                else:
                    distrib = torch.distributions.Categorical(y)
                    actions = distrib.sample()
                    previous_action = actions.detach().cpu().numpy()

                    chose_item1 = (previous_action[0] == 1)
                    if (correct_is_item1 and chose_item1) or \
                       (not correct_is_item1 and not chose_item1):
                        reward[0] = config.rew
                    else:
                        reward[0] = -config.rew

            global_step += 1

    config.bs = original_bs

    accuracies = []
    for pair in ALL_PAIRS:
        total = pair_total_counts.get(pair, 0)
        acc = pair_correct_counts.get(pair, 0) / total if total > 0 else 0.0
        accuracies.append(acc)

    return np.array(accuracies)


def main():
    config = TrainConfig()
    net = RetroModulRNN(config.to_model_dict())
    model_path = Path("net.dat")
    if not model_path.exists():
        raise FileNotFoundError(
            "请先运行 simple_neo.py 完成元训练，生成 net.dat 文件！"
        )
    net.load_state_dict(torch.load(model_path, map_location=DEVICE))
    net.to(DEVICE)

    for param in net.parameters():
        param.requires_grad = False
    net.eval()

    print(f"[加载] 模型已加载，参数量: {sum(p.numel() for p in net.parameters())}")

    fixed_seed = 42
    np.random.seed(fixed_seed)
    cue_data_full = generate_cue_data(config, nbcues=8)
    print(f"[刺激] 已生成 8 个固定的随机刺激编码 (seed={fixed_seed})")

    all_accuracies = []
    print(f"[模拟] 开始模拟 {NUM_PARTICIPANTS} 个虚拟受试者...")
    for pid in range(NUM_PARTICIPANTS):
        participant_seed = pid * 12345 + 67890
        acc = run_single_participant(net, config, cue_data_full, participant_seed)
        all_accuracies.append(acc)

        if (pid + 1) % 10 == 0:
            print(f"  已完成 {pid + 1}/{NUM_PARTICIPANTS}")

    result_matrix = np.array(all_accuracies)
    np.save("results.npy", result_matrix)
    print(f"[保存] 结果已保存至 results.npy")
    print(f"  矩阵形状: {result_matrix.shape}")
    print(f"  总体平均正确率: {result_matrix.mean():.3f}")

    pair_labels = [f"{chr(65+a)}{chr(65+b)}" for a, b in ALL_PAIRS]
    print("\n各配对的平均正确率（28个）:")
    for idx in range(28):
        print(f"  {pair_labels[idx]}: {result_matrix[:, idx].mean():.3f} ± {result_matrix[:, idx].std():.3f}")


if __name__ == "__main__":
    main()