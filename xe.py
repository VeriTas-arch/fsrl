#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整实验脚本（定制化绘图 + 强化学习 / 交叉熵 双模式）
- 支持预训练模型或随机初始化（白模型）
- 支持纯可塑性学习（gradient_update_freq=0）
- 支持梯度下降（强化学习损失或交叉熵）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from scipy.stats import beta, ttest_1samp
import pickle
import os
import time
from tqdm import tqdm
import itertools
from collections import deque
from matplotlib.patches import Patch

# ----------------------------- 设备 -----------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"使用设备: {device}")

# ----------------------------- 网络定义（来自原版） -----------------------------
class RetroModulRNN(nn.Module):
    def __init__(self, params):
        super(RetroModulRNN, self).__init__()
        for paramname in ['outputsize', 'inputsize', 'hs', 'bs']:
            if paramname not in params.keys():
                raise KeyError("Must provide missing key in argument 'params': "+paramname)
        self.params = params
        self.activ = torch.tanh
        self.i2h = torch.nn.Linear(self.params['inputsize'], params['hs']).to(device)
        self.w = torch.nn.Parameter(( (1.0 / np.sqrt(params['hs'])) * (2.0 * torch.rand(params['hs'], params['hs']) - 1.0) ).to(device), requires_grad=True)
        self.alpha = torch.nn.Parameter(.01 * (2.0 * torch.rand(params['hs'], params['hs']) - 1.0).to(device), requires_grad=True)
        self.etaet = torch.nn.Parameter((.7 * torch.ones(1)).to(device), requires_grad=True)
        self.DAmult = torch.nn.Parameter((1.0 * torch.ones(1)).to(device), requires_grad=True)
        self.h2DA = torch.nn.Linear(params['hs'], 2).to(device)
        self.h2o = torch.nn.Linear(params['hs'], self.params['outputsize']).to(device)
        self.h2v = torch.nn.Linear(params['hs'], 1).to(device)

    def forward(self, inputs, hidden, et, pw):
        BATCHSIZE = inputs.shape[0]
        HS = self.params['hs']
        hactiv = self.activ(
            self.i2h(inputs).view(BATCHSIZE, HS, 1) +
            torch.matmul((self.w + torch.mul(self.alpha, pw)),
                         hidden.view(BATCHSIZE, HS, 1))
        ).view(BATCHSIZE, HS)
        activout = self.h2o(hactiv)
        valueout = self.h2v(hactiv)
        DAout2 = torch.tanh(self.h2DA(hactiv))
        DAout = self.DAmult * (DAout2[:,0] - DAout2[:,1])[:,None]
        deltapw = DAout.view(BATCHSIZE,1,1) * et
        pw = pw + deltapw
        torch.clip_(pw, min=-50.0, max=50.0)
        deltaet = torch.bmm(hactiv.view(BATCHSIZE, HS, 1), hidden.view(BATCHSIZE, 1, HS))
        deltaet = torch.tanh(deltaet)
        et = (1 - self.etaet) * et + self.etaet * deltaet
        hidden = hactiv
        return activout, valueout, DAout, hidden, et, pw

    def initialZeroET(self, mybs):
        return torch.zeros(mybs, self.params['hs'], self.params['hs'], requires_grad=False).to(device)

    def initialZeroPlasticWeights(self, mybs):
        return torch.zeros(mybs, self.params['hs'], self.params['hs'], requires_grad=False).to(device)

    def initialZeroState(self, mybs):
        return torch.zeros(mybs, self.params['hs'], requires_grad=False).to(device)

# ----------------------------- 加载模型 -----------------------------
def load_model(path='net_active.dat'):
    state_dict = torch.load(path, map_location='cpu')
    input_size = state_dict['i2h.weight'].shape[1]
    hidden_size = state_dict['i2h.weight'].shape[0]
    output_size = state_dict['h2o.weight'].shape[0]
    params = {
        'inputsize': input_size,
        'hs': hidden_size,
        'outputsize': output_size,
        'bs': 1
    }
    model = RetroModulRNN(params)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

# ----------------------------- 辅助函数 -----------------------------
def all_pairs(n_items=8):
    return [(i, j) for i in range(n_items) for j in range(i+1, n_items)]

def select_learning_pairs_for_order(true_order, n_items=8, max_adjacent=2, max_attempts=2000):
    all_possible = [(i, j) for i in range(n_items) for j in range(i+1, n_items)]
    for _ in range(max_attempts):
        chosen = list(np.random.choice(len(all_possible), size=8, replace=False))
        pairs = [all_possible[idx] for idx in chosen]
        adj_count = sum(1 for (i,j) in pairs if abs(i-j) == 1)
        if adj_count > max_adjacent:
            continue
        adj = np.zeros((n_items, n_items), dtype=int)
        for (i,j) in pairs:
            if true_order[i] < true_order[j]:
                adj[i, j] = 1
            else:
                adj[j, i] = 1
        indeg = np.sum(adj, axis=0)
        q = deque([k for k in range(n_items) if indeg[k] == 0])
        order = []
        unique = True
        while q:
            if len(q) > 1:
                unique = False
                break
            u = q.popleft()
            order.append(u)
            for v in range(n_items):
                if adj[u, v]:
                    indeg[v] -= 1
                    if indeg[v] == 0:
                        q.append(v)
        if unique and len(order) == n_items:
            return pairs
    fallback = [(0,3), (1,4), (2,5), (3,6), (4,7), (0,6), (1,7), (2,7)]
    return fallback

def build_input(stim1, stim2, step, reward, action_prev):
    stim_part = np.concatenate([stim1, stim2])  # 30
    go = 1.0 if step == 1 else 0.0
    time = step / 3.0
    extra = np.array([1.0, time, reward, 0.0])  # 4
    action_onehot = np.zeros(2)
    action_onehot[action_prev] = 1.0
    return np.concatenate([stim_part, [go], extra, action_onehot])  # 37

def run_trial_batch(model, stim1_batch, stim2_batch, target_batch, hidden, et, pw, reward_enabled=True):
    batch_size = stim1_batch.shape[0]
    action_prev = np.zeros(batch_size, dtype=np.int64)
    choice = np.full(batch_size, -1, dtype=np.int64)
    correct = np.zeros(batch_size, dtype=bool)
    reward_val = np.zeros(batch_size, dtype=np.float32)

    for step in range(4):
        reward_input = reward_val if step == 3 else np.zeros(batch_size, dtype=np.float32)
        inp_list = []
        for b in range(batch_size):
            inp = build_input(stim1_batch[b], stim2_batch[b], step, reward_input[b], action_prev[b])
            inp_list.append(inp)
        inp_np = np.stack(inp_list, axis=0)
        inp_t = torch.tensor(inp_np, dtype=torch.float32, device=device)

        with torch.set_grad_enabled(False):
            logits, value, DAout, hidden, et, pw = model(inp_t, hidden, et, pw)

        if step == 1:
            probs = F.softmax(logits, dim=1).cpu().numpy()
            for b in range(batch_size):
                choice[b] = np.random.choice([0, 1], p=probs[b])
            action_prev = choice.copy()
            for b in range(batch_size):
                correct[b] = (choice[b] == 0 and target_batch[b] == 1) or (choice[b] == 1 and target_batch[b] == -1)
            if reward_enabled:
                reward_val = np.where(correct, 1.0, -1.0).astype(np.float32)
            else:
                reward_val = np.zeros(batch_size, dtype=np.float32)

    return choice, correct, hidden, et, pw

# ----------------------------- 训练函数（强化学习版） -----------------------------
def train_trial_batch_rl(model, stim1_batch, stim2_batch, target_batch, hidden, et, pw):
    """
    强化学习损失（策略梯度），仅对决策步骤计算损失，后续步骤不参与梯度。
    """
    batch_size = stim1_batch.shape[0]
    action_prev = np.zeros(batch_size, dtype=np.int64)
    choice = np.full(batch_size, -1, dtype=np.int64)
    correct = np.zeros(batch_size, dtype=bool)
    reward_val = np.zeros(batch_size, dtype=np.float32)
    log_probs = []
    rewards = []

    for step in range(4):
        reward_input = reward_val if step == 3 else np.zeros(batch_size, dtype=np.float32)
        inp_list = []
        for b in range(batch_size):
            inp = build_input(stim1_batch[b], stim2_batch[b], step, reward_input[b], action_prev[b])
            inp_list.append(inp)
        inp_np = np.stack(inp_list, axis=0)
        inp_t = torch.tensor(inp_np, dtype=torch.float32, device=device)

        with torch.set_grad_enabled(True):
            logits, value, DAout, hidden, et, pw = model(inp_t, hidden, et, pw)

        if step == 1:
            probs = F.softmax(logits, dim=1)
            log_prob = F.log_softmax(logits, dim=1)
            actions = torch.multinomial(probs, 1).squeeze(1)
            chosen_log_prob = log_prob.gather(1, actions.unsqueeze(1)).squeeze(1)
            log_probs.append(chosen_log_prob)

            action_np = actions.cpu().numpy()
            choice[:] = action_np
            action_prev = choice.copy()
            for b in range(batch_size):
                correct[b] = (choice[b] == 0 and target_batch[b] == 1) or (choice[b] == 1 and target_batch[b] == -1)
            reward_val = np.where(correct, 1.0, -1.0).astype(np.float32)
            rewards.append(torch.tensor(reward_val, dtype=torch.float32, device=device))

            # 切断后续步骤的梯度
            hidden = hidden.detach()
            et = et.detach()
            pw = pw.detach()

    log_probs = torch.stack(log_probs)  # (1, batch_size)
    rewards = torch.stack(rewards)      # (1, batch_size)
    loss = - (log_probs * rewards.detach()).mean()
    # 可选：熵正则
    # loss -= 0.01 * (probs * log_prob).sum(dim=1).mean()
    return choice, correct, hidden, et, pw, loss

# ----------------------------- 训练函数（交叉熵版，保留以作备选） -----------------------------
def train_trial_batch(model, stim1_batch, stim2_batch, target_batch, hidden, et, pw):
    """
    交叉熵损失（监督学习），使用方式见 run_experiment 中的 use_rl_loss=False。
    此版本已修正梯度传播问题（在步骤1后 detach 状态）。
    """
    batch_size = stim1_batch.shape[0]
    action_prev = np.zeros(batch_size, dtype=np.int64)
    choice = np.full(batch_size, -1, dtype=np.int64)
    correct = np.zeros(batch_size, dtype=bool)
    reward_val = np.zeros(batch_size, dtype=np.float32)
    loss = None

    for step in range(4):
        reward_input = reward_val if step == 3 else np.zeros(batch_size, dtype=np.float32)
        inp_list = []
        for b in range(batch_size):
            inp = build_input(stim1_batch[b], stim2_batch[b], step, reward_input[b], action_prev[b])
            inp_list.append(inp)
        inp_np = np.stack(inp_list, axis=0)
        inp_t = torch.tensor(inp_np, dtype=torch.float32, device=device)

        with torch.set_grad_enabled(True):
            logits, value, DAout, hidden, et, pw = model(inp_t, hidden, et, pw)

        if step == 1:
            targets_t = torch.tensor(target_batch, dtype=torch.long, device=device)
            labels = (targets_t == -1).long()
            loss = F.cross_entropy(logits, labels)

            probs = F.softmax(logits, dim=1).detach().cpu().numpy()
            for b in range(batch_size):
                choice[b] = np.random.choice([0, 1], p=probs[b])
            action_prev = choice.copy()
            for b in range(batch_size):
                correct[b] = (choice[b] == 0 and target_batch[b] == 1) or (choice[b] == 1 and target_batch[b] == -1)
            reward_val = np.where(correct, 1.0, -1.0).astype(np.float32)

            # 切断后续步骤的梯度
            hidden = hidden.detach()
            et = et.detach()
            pw = pw.detach()

    return choice, correct, hidden, et, pw, loss

# ----------------------------- 实验主函数 -----------------------------
def run_experiment(model, n_subjects=77, n_items=8, stim_dim=14,
                   noise_std=0.05, n_learning_blocks=4, n_test_blocks=10,
                   learning_use_val=False, learning_use_noise=False,
                   test_use_val=False, test_use_noise=False,
                   max_adjacent=2, gradient_update_freq=4, use_rl_loss=True):
    """
    运行实验。
    gradient_update_freq: 每多少次试次更新参数（0 表示不更新）
    use_rl_loss: True 使用强化学习损失，False 使用交叉熵（需定义 train_trial_batch）
    """
    all_pairs_list = all_pairs(n_items)
    n_pairs = len(all_pairs_list)

    all_accuracies = []
    all_subject_rankings = []
    all_hidden_rankings = []

    if gradient_update_freq > 0:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, eps=1e-6)
        model.train()
    else:
        optimizer = None
        model.eval()

    for subj in tqdm(range(n_subjects), desc="被试"):
        bases = np.random.randint(0, 2, size=(n_items, stim_dim)).astype(np.float32) * 2 - 1
        true_order = np.random.permutation(n_items)
        true_vals = (true_order / (n_items - 1)) * 2 - 1

        pairs = select_learning_pairs_for_order(true_order, n_items, max_adjacent=max_adjacent)
        learning_pair_indices = []
        for (i, j) in pairs:
            for idx, (pi, pj) in enumerate(all_pairs_list):
                if (pi == i and pj == j) or (pi == j and pj == i):
                    learning_pair_indices.append(idx)
                    break
        assert len(learning_pair_indices) == 8

        targets = np.zeros(n_pairs, dtype=np.int8)
        for idx, (i, j) in enumerate(all_pairs_list):
            targets[idx] = 1 if true_vals[i] > true_vals[j] else -1

        pair_noise = np.random.normal(0, noise_std, size=n_pairs).astype(np.float32)

        learning_order = []
        for _ in range(n_learning_blocks):
            block = learning_pair_indices.copy()
            np.random.shuffle(block)
            learning_order.extend(block)

        test_order = []
        for _ in range(n_test_blocks):
            block = list(range(n_pairs))
            np.random.shuffle(block)
            test_order.extend(block)

        hidden = model.initialZeroState(1)
        et = model.initialZeroET(1)
        pw = model.initialZeroPlasticWeights(1)

        correct_counts = np.zeros(n_pairs, dtype=np.int64)
        total_counts = np.zeros(n_pairs, dtype=np.int64)

        def get_stim(idx, pair_idx, use_val=True, use_noise=True):
            base = bases[idx]
            if use_val:
                val = true_vals[idx] + (pair_noise[pair_idx] if use_noise else 0.0)
            else:
                val = 0.0
            return np.concatenate([base, [val]])

        grad_counter = 0
        for pair_idx in learning_order:
            i, j = all_pairs_list[pair_idx]
            stim1 = get_stim(i, pair_idx, use_val=learning_use_val, use_noise=learning_use_noise)
            stim2 = get_stim(j, pair_idx, use_val=learning_use_val, use_noise=learning_use_noise)
            target = targets[pair_idx]
            stim1_b = stim1[np.newaxis, :]
            stim2_b = stim2[np.newaxis, :]
            target_b = np.array([target])

            if gradient_update_freq > 0:
                if use_rl_loss:
                    _, _, hidden, et, pw, loss = train_trial_batch_rl(
                        model, stim1_b, stim2_b, target_b, hidden, et, pw
                    )
                else:
                    _, _, hidden, et, pw, loss = train_trial_batch(
                        model, stim1_b, stim2_b, target_b, hidden, et, pw
                    )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                grad_counter += 1
                if grad_counter % gradient_update_freq == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                hidden = hidden.detach()
                et = et.detach()
                pw = pw.detach()
            else:
                _, _, hidden, et, pw = run_trial_batch(
                    model, stim1_b, stim2_b, target_b, hidden, et, pw, reward_enabled=True
                )

        if gradient_update_freq > 0:
            model.eval()

        for pair_idx in test_order:
            i, j = all_pairs_list[pair_idx]
            stim1 = get_stim(i, pair_idx, use_val=test_use_val, use_noise=test_use_noise)
            stim2 = get_stim(j, pair_idx, use_val=test_use_val, use_noise=test_use_noise)
            target = targets[pair_idx]
            stim1_b = stim1[np.newaxis, :]
            stim2_b = stim2[np.newaxis, :]
            target_b = np.array([target])
            _, correct, hidden, et, pw = run_trial_batch(
                model, stim1_b, stim2_b, target_b, hidden, et, pw, reward_enabled=False
            )
            if correct[0]:
                correct_counts[pair_idx] += 1
            total_counts[pair_idx] += 1

        acc = correct_counts / total_counts
        all_accuracies.append(acc)

        wins = np.zeros((n_items, n_items))
        for idx, (i, j) in enumerate(all_pairs_list):
            if acc[idx] > 0.5:
                wins[i, j] += 1
            else:
                wins[j, i] += 1
        scores = np.sum(wins, axis=1)
        ranking = np.argsort(-scores)
        all_subject_rankings.append(ranking)
        all_hidden_rankings.append(true_order)

    all_accuracies = np.array(all_accuracies)
    return {
        'accuracies': all_accuracies,
        'learning_pair_indices': learning_pair_indices,
        'all_pairs': all_pairs_list,
        'n_items': n_items,
        'subject_rankings': all_subject_rankings,
        'hidden_rankings': all_hidden_rankings,
        'n_subjects': n_subjects
    }

# ----------------------------- 分析函数（保持不变） -----------------------------
def analyze_results(data):
    accs = data['accuracies']
    n_subjects, n_pairs = accs.shape
    n_items = data['n_items']
    pairs = data['all_pairs']

    alpha_vals = []
    beta_vals = []
    for idx in range(n_pairs):
        a = accs[:, idx]
        m = np.mean(a)
        v = np.var(a)
        if v == 0:
            alpha_, beta_ = 1.0, 1.0
        else:
            alpha_ = m * (m*(1-m)/v - 1)
            beta_ = (1-m) * (m*(1-m)/v - 1)
            alpha_ = max(alpha_, 0.01)
            beta_ = max(beta_, 0.01)
        alpha_vals.append(alpha_)
        beta_vals.append(beta_)

    color_codes = []
    for a, b in zip(alpha_vals, beta_vals):
        if a < 1 and b < 1:
            color_codes.append(0)
        elif a > 1 and b > 1:
            color_codes.append(1)
        elif a > 1 and b < 1:
            color_codes.append(2)
        else:
            color_codes.append(3)

    def count_circular_triads_from_judgments(judg):
        n = len(judg)
        circular = 0
        for triple in itertools.combinations(range(n), 3):
            a, b, c = triple
            if judg[a, b] and judg[b, c] and judg[c, a]:
                circular += 1
            elif judg[a, c] and judg[c, b] and judg[b, a]:
                circular += 1
        return circular

    circular_counts = []
    for subj in range(n_subjects):
        true_order = data['hidden_rankings'][subj]
        judg = np.zeros((n_items, n_items), dtype=int)
        for idx, (i, j) in enumerate(pairs):
            if true_order[i] > true_order[j]:
                ground_dir = 1
            else:
                ground_dir = -1
            if accs[subj, idx] > 0.5:
                subj_dir = ground_dir
            elif accs[subj, idx] < 0.5:
                subj_dir = -ground_dir
            else:
                subj_dir = 0
            if subj_dir == 1:
                judg[i, j] = 1
                judg[j, i] = 0
            elif subj_dir == -1:
                judg[i, j] = 0
                judg[j, i] = 1
            else:
                judg[i, j] = 0
                judg[j, i] = 0
        circular_counts.append(count_circular_triads_from_judgments(judg))

    hard_pairs = [idx for idx, c in enumerate(color_codes) if c == 0]
    error_consistency = []
    for subj_idx in range(n_subjects):
        if not hard_pairs:
            consistency = 1.0
        else:
            consistent = 0
            total = len(hard_pairs)
            for idx in hard_pairs:
                if accs[subj_idx, idx] < 0.1 or accs[subj_idx, idx] > 0.9:
                    consistent += 1
            consistency = consistent / total if total > 0 else 1.0
        error_consistency.append(consistency)

    from scipy.stats import kendalltau
    similarities = []
    for i in range(n_subjects):
        for j in range(i+1, n_subjects):
            tau, _ = kendalltau(data['subject_rankings'][i], data['subject_rankings'][j])
            similarities.append(tau)

    return {
        'beta_color_codes': color_codes,
        'circular_counts': circular_counts,
        'error_consistency': error_consistency,
        'inter_subject_similarity': similarities
    }

# ----------------------------- 绘图函数（与之前一致） -----------------------------
def plot_results(data, results, save_dir='result_custom'):
    os.makedirs(save_dir, exist_ok=True)
    accs = data['accuracies']
    n_subjects, n_pairs = accs.shape
    n_items = data['n_items']
    learning_idx = data['learning_pair_indices']
    pairs = data['all_pairs']

    # 图1
    learned_accs = []
    unlearned_accs = []
    for subj in range(n_subjects):
        learned = [accs[subj, idx] for idx in learning_idx]
        unlearned = [accs[subj, idx] for idx in range(n_pairs) if idx not in learning_idx]
        learned_accs.append(np.mean(learned))
        unlearned_accs.append(np.mean(unlearned))

    plt.figure(figsize=(6, 4))
    bp = plt.boxplot([learned_accs, unlearned_accs], tick_labels=['Learned', 'Unlearned'],
                     patch_artist=True, showmeans=False)
    for i, data_ in enumerate([learned_accs, unlearned_accs], start=1):
        mean_val = np.mean(data_)
        std_val = np.std(data_)
        min_val = np.min(data_)
        max_val = np.max(data_)
        plt.text(i, 0.02, f'std={std_val:.3f}\nrange={max_val-min_val:.3f}',
                 ha='center', va='bottom', fontsize=8)
    plt.ylabel('Average Accuracy')
    plt.title('Figure 1: Learned vs Unlearned Pairs')
    plt.savefig(f'{save_dir}/fig1_learned_vs_unlearned.png', dpi=150)
    plt.close()

    # 图2
    item_accs = []
    for item in range(n_items):
        idxs = [idx for idx, (i, j) in enumerate(pairs) if i == item or j == item]
        subj_accs = np.mean(accs[:, idxs], axis=1)
        item_accs.append(subj_accs)
    item_accs = np.array(item_accs)
    mean_item = np.mean(item_accs, axis=1)
    std_item = np.std(item_accs, axis=1)
    plt.figure(figsize=(8, 5))
    plt.bar(range(n_items), mean_item, yerr=std_item, capsize=3, alpha=0.7, color='skyblue')
    plt.xlabel('Item index')
    plt.ylabel('Mean accuracy')
    plt.title('Figure 2: Accuracy per item')
    plt.xticks(range(n_items))
    plt.savefig(f'{save_dir}/fig2_accuracy_per_item.png', dpi=150)
    plt.close()

    # 图3
    mean_pair = np.mean(accs, axis=0)
    dist_dict = {}
    for idx, (i, j) in enumerate(pairs):
        d = abs(i - j)
        dist_dict.setdefault(d, []).append(mean_pair[idx])
    dists = sorted(dist_dict.keys())
    mean_dist = [np.mean(dist_dict[d]) for d in dists]
    std_dist = [np.std(dist_dict[d]) for d in dists]
    plt.figure(figsize=(6, 4))
    plt.errorbar(dists, mean_dist, yerr=std_dist, fmt='o-', capsize=3)
    plt.xlabel('Relative distance')
    plt.ylabel('Mean accuracy')
    plt.title('Figure 3: Accuracy by relative distance')
    plt.savefig(f'{save_dir}/fig3_distance_effect.png', dpi=150)
    plt.close()

    # 图4
    mat = np.full((n_items, n_items), np.nan)
    for idx, (i, j) in enumerate(pairs):
        mat[i, j] = mean_pair[idx]
        mat[j, i] = mean_pair[idx]
    mask = np.triu(np.ones_like(mat, dtype=bool), k=0)
    annot = np.full((n_items, n_items), '', dtype='<U10')
    for idx, (i, j) in enumerate(pairs):
        star = '*' if idx in learning_idx else ''
        annot[i, j] = f'{mean_pair[idx]:.2f}{star}'
        annot[j, i] = f'{mean_pair[idx]:.2f}{star}'
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(mat, mask=mask, annot=annot, fmt='s', cmap='coolwarm',
                     cbar_kws={'label': 'Accuracy'}, square=True)
    ax.set_title('Figure 4: Accuracy matrix (* = learned pair)')
    plt.savefig(f'{save_dir}/fig4_accuracy_matrix.png', dpi=150)
    plt.close()

    # 图5
    fig, axes = plt.subplots(4, 7, figsize=(20, 12))
    axes = axes.flatten()
    for idx in range(n_pairs):
        axes[idx].hist(accs[:, idx], bins=np.linspace(0, 1, 11), alpha=0.7,
                       color='skyblue', edgecolor='black')
        axes[idx].set_title(f'Pair {idx}')
        axes[idx].set_xlabel('Accuracy')
        axes[idx].set_ylabel('Count')
    for idx in range(n_pairs, len(axes)):
        axes[idx].axis('off')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/fig5_accuracy_distribution_per_pair.png', dpi=150)
    plt.close()

    # 图6
    color_codes = results['beta_color_codes']
    mat_beta = np.full((n_items, n_items), np.nan)
    for idx, (i, j) in enumerate(pairs):
        mat_beta[i, j] = color_codes[idx]
        mat_beta[j, i] = color_codes[idx]
    mask_beta = np.triu(np.ones_like(mat_beta, dtype=bool), k=0)
    cmap_beta = sns.color_palette(['forestgreen', 'lightgray', 'sandybrown', 'saddlebrown'])
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(mat_beta, mask=mask_beta, annot=False, fmt='',
                     cmap=cmap_beta, cbar=False, square=True)
    for idx, (j, i) in enumerate(pairs):
        if idx in learning_idx:
            ax.text(j+0.5, i+0.5, '★', ha='center', va='center', color='black', fontsize=12)
    legend_elements = [
        Patch(facecolor='forestgreen', label='Bimodal (α<1, β<1)'),
        Patch(facecolor='lightgray', label='Unimodal (α>1, β>1)'),
        Patch(facecolor='sandybrown', label='High accuracy (α>1, β<1)'),
        Patch(facecolor='saddlebrown', label='Low accuracy (α<1, β>1)')
    ]
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_title('Figure 6: Beta distribution types (★ = learned pair)')
    plt.savefig(f'{save_dir}/fig6_beta_types.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 图7
    thresholds = [0.6, 0.7, 0.8, 0.9, 1.0]
    counts = []
    for th in thresholds:
        cnt = sum(1 for subj in range(n_subjects) if np.any(accs[subj] < th))
        counts.append(cnt)
    plt.figure(figsize=(6, 4))
    plt.bar([str(t) for t in thresholds], counts, color='teal')
    plt.xlabel('Error rate threshold')
    plt.ylabel('Number of subjects')
    plt.title('Figure 7: Subjects with at least one pair error > threshold')
    for i, v in enumerate(counts):
        plt.text(i, v+0.5, str(v), ha='center')
    plt.savefig(f'{save_dir}/fig7_threshold_counts.png', dpi=150)
    plt.close()

    # 图8
    err_mat = 1 - accs
    plt.figure(figsize=(14, 10))
    sns.heatmap(err_mat, cmap='Reds', cbar_kws={'label': 'Error rate'},
                xticklabels=range(n_pairs), yticklabels=[f'S{i+1}' for i in range(n_subjects)])
    plt.xlabel('Pair index')
    plt.ylabel('Subject')
    plt.title('Figure 8: Error rates per subject and per pair')
    plt.savefig(f'{save_dir}/fig8_error_heatmap.png', dpi=150)
    plt.close()

    # 图9（新增）
    circular_counts = results['circular_counts']
    max_possible = 20
    self_consistency_scores = [1 - c / max_possible for c in circular_counts]
    n_perfect = sum(1 for s in self_consistency_scores if s == 1.0)
    plt.figure(figsize=(14, 6))
    colors = ['green' if s == 1.0 else 'orange' if s >= 0.8 else 'red' for s in self_consistency_scores]
    plt.bar(range(1, n_subjects+1), self_consistency_scores, color=colors, alpha=0.7,
            edgecolor='black', linewidth=0.5)
    plt.axhline(1.0, color='green', linestyle='--', alpha=0.7, label='Perfect consistency (1.0)')
    plt.axhline(0.8, color='orange', linestyle=':', alpha=0.7, label='High consistency (≥0.8)')
    plt.xlabel('Subject ID')
    plt.ylabel('Self-consistency coefficient')
    plt.title('Figure 9: Self-consistency per subject')
    plt.ylim(0, 1.1)
    plt.text(0.02, 0.95, f'Perfectly consistent (score=1.0): {n_perfect}/{n_subjects}',
             transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/fig9_self_consistency.png', dpi=150)
    plt.close()
    print(f"所有图表已保存到 {save_dir}/")

# ----------------------------- 主程序 -----------------------------
def main():
    np.random.seed(42)
    torch.manual_seed(42)

    # ===== 模型选择开关 =====
    load_pretrained = True   # True 加载预训练模型 net_active.dat，False 随机初始化

    if load_pretrained:
        model_path = 'net_active.dat'
        if not os.path.exists(model_path):
            print(f"错误：找不到模型文件 {model_path}")
            return
        model = load_model(model_path)
        print("使用预训练模型")
    else:
        # 随机初始化白模型，需要与原模型参数一致
        params = {
            'inputsize': 37,          # 根据 build_input 计算
            'hs': 200,                # 与原模型一致
            'outputsize': 2,
            'bs': 1
        }
        model = RetroModulRNN(params)
        model.to(device)
        print("使用随机初始化白模型")

    print("开始实验（77个被试）...")
    t0 = time.time()
    data = run_experiment(model, n_subjects=77, n_items=8, stim_dim=14, noise_std=0.05,
                          n_learning_blocks=300, n_test_blocks=10, max_adjacent=2,
                          learning_use_val=False, learning_use_noise=False,
                          test_use_val=False, test_use_noise=False,
                          gradient_update_freq=4,          # 设为 0 关闭梯度下降
                          use_rl_loss=True)                # True 强化学习，False 交叉熵
    print(f"实验完成，耗时 {time.time()-t0:.2f} 秒")

    with open('results_custom.pkl', 'wb') as f:
        pickle.dump(data, f)

    results = analyze_results(data)
    plot_results(data, results)
    print("全部完成！")

if __name__ == '__main__':
    main()