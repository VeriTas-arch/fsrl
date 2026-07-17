#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
独立绘图脚本：从已保存的 results_custom.pkl 加载数据并绘图
无需重新训练，直接使用缓存数据生成9张图
"""

import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
import itertools
from scipy.stats import kendalltau

# ----------------------------- 分析函数 -----------------------------
def analyze_results(data):
    """
    分析实验结果，计算所有需要的统计量
    返回 results 字典
    """
    accs = data['accuracies']
    n_subjects, n_pairs = accs.shape
    n_items = data['n_items']
    pairs = data['all_pairs']

    # ---------- 1. Beta分布拟合 ----------
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
            color_codes.append(0)   # 双峰 -> 森绿
        elif a > 1 and b > 1:
            color_codes.append(1)   # 单峰 -> 浅灰
        elif a > 1 and b < 1:
            color_codes.append(2)   # 高准 -> 棕黄
        else:
            color_codes.append(3)   # 低准 -> 深褐

    # ---------- 2. 自洽性分析（基于判断矩阵） ----------
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

    # ---------- 3. 错误一致性（针对困难配对） ----------
    hard_pairs = [idx for idx, c in enumerate(color_codes) if c == 0]  # 双峰
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

    # ---------- 4. 被试间排名相似性 ----------
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

# ----------------------------- 绘图函数 -----------------------------
def plot_results(data, results, save_dir='result_custom'):
    os.makedirs(save_dir, exist_ok=True)

    accs = data['accuracies']  # (n_subjects, n_pairs)
    n_subjects, n_pairs = accs.shape
    n_items = data['n_items']
    learning_idx = data['learning_pair_indices']
    pairs = data['all_pairs']

    # ========== 统计量预计算 ==========
    mean_pair = np.mean(accs, axis=0)
    std_pair = np.std(accs, axis=0)

    # 图2：按物品统计
    item_accs = []
    for item in range(n_items):
        idxs = [idx for idx, (i, j) in enumerate(pairs) if i == item or j == item]
        subj_accs = np.mean(accs[:, idxs], axis=1)
        item_accs.append(subj_accs)
    item_accs = np.array(item_accs)  # (n_items, n_subjects)
    mean_item = np.mean(item_accs, axis=1)
    std_item = np.std(item_accs, axis=1)

    # ============ 图1 ============
    learned_accs = []
    unlearned_accs = []
    for subj in range(n_subjects):
        learned = [accs[subj, idx] for idx in learning_idx]
        unlearned = [accs[subj, idx] for idx in range(n_pairs) if idx not in learning_idx]
        learned_accs.append(np.mean(learned))
        unlearned_accs.append(np.mean(unlearned))

    plt.figure(figsize=(6, 4))
    plt.boxplot([learned_accs, unlearned_accs], tick_labels=['Learned', 'Unlearned'],
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

    # ============ 图2：按物品 ============
    plt.figure(figsize=(8, 5))
    plt.bar(range(n_items), mean_item, yerr=std_item, capsize=3, alpha=0.7, color='skyblue')
    plt.xlabel('Item index')
    plt.ylabel('Mean accuracy')
    plt.title('Figure 2: Accuracy per item (averaged over pairs containing that item)')
    plt.xticks(range(n_items))
    plt.savefig(f'{save_dir}/fig2_accuracy_per_item.png', dpi=150)
    plt.close()

    # ============ 图3 ============
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

    # ============ 图4 ============
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

    # ============ 图5 ============
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

    # ============ 图6 ============
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

    # 在下三角区域添加星号（仅学习对）
    for idx, (i, j) in enumerate(pairs):
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

    # ============ 图7 ============
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

    # ============ 图8 ============
    err_mat = 1 - accs
    plt.figure(figsize=(14, 10))
    sns.heatmap(err_mat, cmap='Reds', cbar_kws={'label': 'Error rate'},
                xticklabels=range(n_pairs), yticklabels=[f'S{i+1}' for i in range(n_subjects)])
    plt.xlabel('Pair index')
    plt.ylabel('Subject')
    plt.title('Figure 8: Error rates per subject and per pair')
    plt.savefig(f'{save_dir}/fig8_error_heatmap.png', dpi=150)
    plt.close()

    # ============ 图9：自洽性系数 ============
    circular_counts = results['circular_counts']
    max_possible = 20
    self_consistency_scores = [1 - c / max_possible for c in circular_counts]
    n_subjects = len(self_consistency_scores)
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
    plt.text(0.02, 0.88, f'High consistency (score≥0.8): {n_perfect + sum(1 for s in self_consistency_scores if 0.8 <= s < 1.0)}/{n_subjects}',
             transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/fig9_self_consistency.png', dpi=150)
    plt.close()

    print(f"所有图表已保存到 {save_dir}/")

# ----------------------------- 主程序 -----------------------------
if __name__ == '__main__':
    # 确保 results_custom.pkl 存在
    if not os.path.exists('results_custom.pkl'):
        print("错误：未找到 results_custom.pkl，请先运行实验生成数据文件。")
        exit(1)

    print("从缓存加载数据...")
    with open('results_custom.pkl', 'rb') as f:
        data = pickle.load(f)

    print("数据分析中...")
    results = analyze_results(data)

    print("绘图生成中...")
    plot_results(data, results)

    print("全部完成！图表保存在 result_custom/ 目录下。")