"""
run_liu_replication.py (最终修正版)
独立脚本：加载预训练的元学习网络，复现 Liu 等人 (2026) 的固定 8 对排序实验。

修正内容：
1. batch_size 匹配：将 config.bs 临时设为 1，与手动构造的单样本 cues 一致。
2. 消除位置偏好：每个试次随机交换刺激呈现顺序，迫使模型真正学习相对等级。
3. 可重复性：每个虚拟受试者开始时调用 set_seed() 固定 torch/numpy/random 状态。
4. 注释全面补充，提升可读性。
"""

import itertools
import numpy as np
import torch
from pathlib import Path

# 从 simple_neo 中导入必要的类和函数（不修改任何原有代码）
from simple_neo import (
    RetroModulRNN,
    TrainConfig,
    DEVICE,
    build_step_inputs,
    generate_cue_data,
    NUMRESPONSESTEP,
    set_seed,
)

# ============= 固定参数 =============
# Liu 实验中的 8 个非相邻配对（索引 0~7 对应 A~H）
FIXED_PAIRS = [(0, 5), (1, 2), (1, 4), (2, 6), (3, 5), (3, 6), (4, 7), (0, 7)]

# 全部 28 个配对（用于测试）
ALL_PAIRS = list(itertools.combinations(range(8), 2))

NUM_PARTICIPANTS = 100      # 虚拟受试者数量
NUM_BLOCKS = 4              # 训练 Block 数
REPEATS_PER_PAIR = 10       # 每个配对测试次数


def run_single_participant(net, config, cue_data_full, participant_seed):
    """
    运行一个虚拟受试者的完整流程：
    1. 重置网络内部状态 (hidden, et, pw)
    2. 4 个 Block 的固定配对训练 (有反馈，允许学习)
    3. 28 对 × 10 次测试 (无反馈，禁止学习)
    返回: 长度为 28 的数组，每个元素是该配对上的平均正确率 (0~1)

    参数：
        net: 预训练的 RetroModulRNN 网络
        config: TrainConfig 对象（会临时修改 bs=1，结束后恢复）
        cue_data_full: 完整 batch 的刺激编码，我们取 [0] 作为单样本刺激
        participant_seed: 该虚拟受试者的随机种子
    """
    # ---- 固定随机种子，确保该虚拟受试者的实验完全可重复 ----
    set_seed(participant_seed)

    # 保存原始 batch_size，并改为 1（因为我们只模拟一个受试者）
    original_bs = config.bs
    config.bs = 1
    # 提取单 batch 的刺激编码（所有刺激编码是固定的，不同 batch 只是重复）
    cue_data = [cue_data_full[0]]  # 保持列表格式，长度为 1

    batch_size = config.bs          # 现在为 1
    nbcues = 8

    # ---- 1. 重置 RNN 状态与塑性相关变量 ----
    hidden = net.initialZeroState(batch_size)
    et = net.initialZeroET(batch_size)
    pw = net.initialZeroPlasticWeights(batch_size)
    previous_action = np.zeros(batch_size, dtype="int32")
    reward = np.zeros(batch_size, dtype="float32")
    numstep_ep = 0

    # ---- 2. 生成训练试次序列（4 个 Block，每个 Block 内随机打乱） ----
    train_sequence = []
    rng_train = np.random.RandomState(participant_seed)  # 独立随机源
    for block_idx in range(NUM_BLOCKS):
        block = FIXED_PAIRS.copy()
        rng_train.shuffle(block)
        train_sequence.extend(block)

    # ---- 3. 训练阶段（32 个试次，有奖励） ----
    for trial_idx, (a, b) in enumerate(train_sequence):
        # 随机交换刺激呈现位置，消除“总选第一个”的简单策略
        if rng_train.rand() < 0.5:
            i, j = b, a       # 交换，使较高等级可能出现在第二位置
        else:
            i, j = a, b
        # 构造本试次的提示序列，格式：[刺激对, 提示符, -1, -1]，batch 维度在前
        cues = [[[i, j], nbcues, -1, -1]]
        correct_is_item1 = 1 if i < j else 0            # i<j 表示 i 是更高的等级

        for numstep in range(config.triallen):
            # 构建当前时间步的输入张量
            inputs = build_step_inputs(
                config,
                nbcues,
                cue_data,
                cues,
                reward,
                previous_action,
                numstep,
                numstep_ep,
            )
            # 网络前向传播
            y_raw, value, daout, hidden, et, pw = net(inputs, hidden, et, pw)
            y = torch.softmax(y_raw, dim=1)
            distrib = torch.distributions.Categorical(y)
            actions = distrib.sample()
            previous_action = actions.detach().cpu().numpy()

            # 在响应步给予奖励
            if numstep == NUMRESPONSESTEP:
                chose_item1 = (previous_action[0] == 1)
                # 若正确选择了较高等级的刺激，给予正奖励，否则负奖励
                if (correct_is_item1 and chose_item1) or (not correct_is_item1 and not chose_item1):
                    reward[0] = config.rew
                else:
                    reward[0] = -config.rew

            # 试次最后一步将 reward 清零，避免跨试次干扰
            if numstep == config.triallen - 1:
                reward[0] = 0.0

            numstep_ep += 1

    # ---- 4. 测试阶段（280 个试次，无反馈，禁止塑性更新） ----
    # 生成 28 对 × 10 次，随机打乱顺序
    test_pairs = ALL_PAIRS * REPEATS_PER_PAIR
    rng_test = np.random.RandomState(participant_seed + 9999)
    rng_test.shuffle(test_pairs)

    # 临时禁用塑性：将 DAmult 置零，防止测试时 pw 继续更新
    original_damult = net.DAmult.data.clone()
    net.DAmult.data = torch.zeros_like(net.DAmult.data)

    # 记录每个配对的正确次数
    pair_correct_counts = {pair: 0 for pair in ALL_PAIRS}
    pair_total_counts = {pair: 0 for pair in ALL_PAIRS}

    for trial_idx, (a, b) in enumerate(test_pairs):
        # 同样随机交换刺激位置
        if rng_test.rand() < 0.5:
            i, j = b, a
        else:
            i, j = a, b
        # 构造提示序列（格式同训练阶段）
        cues = [[[i, j], nbcues, -1, -1]]
        correct_is_item1 = 1 if i < j else 0

        reward = np.zeros(batch_size, dtype="float32")  # 测试期间奖励恒为 0

        for numstep in range(config.triallen):
            inputs = build_step_inputs(
                config,
                nbcues,
                cue_data,
                cues,
                reward,
                previous_action,
                numstep,
                numstep_ep,
            )
            y_raw, value, daout, hidden, et, pw = net(inputs, hidden, et, pw)

            # 仅在响应步记录模型的选择
            if numstep == NUMRESPONSESTEP:
                y = torch.softmax(y_raw, dim=1)
                chose_item1 = (torch.argmax(y, dim=1).item() == 1)

                pair_total_counts[(a, b)] += 1          # 注意：配对身份仍以 (a,b) 记录
                if (correct_is_item1 and chose_item1) or (not correct_is_item1 and not chose_item1):
                    pair_correct_counts[(a, b)] += 1

            numstep_ep += 1

    # 恢复塑性参数和原始 batch_size
    net.DAmult.data = original_damult
    config.bs = original_bs

    # ---- 5. 计算每个配对的平均正确率 ----
    accuracies = []
    for pair in ALL_PAIRS:
        total = pair_total_counts[pair]
        acc = pair_correct_counts[pair] / total if total > 0 else 0.0
        accuracies.append(acc)

    return np.array(accuracies)


def main():
    # ---- 1. 加载配置和训练好的网络 ----
    config = TrainConfig()
    net = RetroModulRNN(config.to_model_dict())
    model_path = Path("net.dat")
    if not model_path.exists():
        raise FileNotFoundError(
            "请先运行 simple_neo.py 完成元训练，生成 net.dat 文件！"
        )
    net.load_state_dict(torch.load(model_path, map_location=DEVICE))
    net.to(DEVICE)

    # 冻结所有基础参数（元学习得到的算法不应在实验中被修改）
    for param in net.parameters():
        param.requires_grad = False
    net.eval()

    print(f"[加载] 模型已加载，参数量: {sum(p.numel() for p in net.parameters())}")

    # ---- 2. 生成固定的 8 个刺激编码（所有虚拟受试者共用） ----
    fixed_seed = 42
    np.random.seed(fixed_seed)
    # generate_cue_data 使用 config.bs（此处为32），返回 32 个 batch 的刺激
    cue_data = generate_cue_data(config, nbcues=8)
    print(f"[刺激] 已生成 8 个固定的随机刺激编码 (seed={fixed_seed})")

    # ---- 3. 运行 N 个虚拟受试者 ----
    all_accuracies = []  # 存储每个参与者的 28 个准确率

    print(f"[模拟] 开始模拟 {NUM_PARTICIPANTS} 个虚拟受试者...")
    for pid in range(NUM_PARTICIPANTS):
        participant_seed = pid * 12345 + 67890
        acc = run_single_participant(net, config, cue_data, participant_seed)
        all_accuracies.append(acc)

        if (pid + 1) % 10 == 0:
            print(f"  已完成 {pid + 1}/{NUM_PARTICIPANTS}")

    # ---- 4. 保存结果 ----
    result_matrix = np.array(all_accuracies)  # shape: (100, 28)
    np.save("liu_replication_accuracies.npy", result_matrix)
    print(f"[保存] 结果已保存至 liu_replication_accuracies.npy")
    print(f"  矩阵形状: {result_matrix.shape}")
    print(f"  总体平均正确率: {result_matrix.mean():.3f}")

    # 简单输出部分配对的表现
    pair_labels = [f"{chr(65+a)}{chr(65+b)}" for a, b in ALL_PAIRS]
    print("\n[示例] 各配对的平均正确率（前10个）:")
    for idx in range(10):
        print(f"  {pair_labels[idx]}: {result_matrix[:, idx].mean():.3f} ± {result_matrix[:, idx].std():.3f}")


if __name__ == "__main__":
    main()