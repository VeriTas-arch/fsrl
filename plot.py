import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations

# ===================== 加载数据 =====================
data = np.load('liu_replication_accuracies.npy')  # shape: (100, 28)

# ===================== 确定连续对的索引 =====================
all_pairs = list(combinations(range(8), 2))
pair_labels = [f"{chr(65+a)}{chr(65+b)}" for a, b in all_pairs]

# 连续对：AB(0,1), BC(1,2), CD(2,3), DE(3,4), EF(4,5), FG(5,6), GH(6,7)
adjacent_pairs = [(i, i+1) for i in range(7)]
adjacent_labels = [f"{chr(65+a)}{chr(65+b)}" for a, b in adjacent_pairs]

# 提取对应的列索引
adjacent_idx = [all_pairs.index(pair) for pair in adjacent_pairs]
adjacent_data = data[:, adjacent_idx]  # shape: (100, 7)

# ===================== 绘制密度曲线 =====================
plt.figure(figsize=(10, 6))

# 定义颜色映射（可选）
colors = sns.color_palette("viridis", 7)

for i, (label, col) in enumerate(zip(adjacent_labels, adjacent_data.T)):
    sns.kdeplot(
        col,
        label=label,
        color=colors[i],
        linewidth=2,
        alpha=0.8
    )

# 添加所有数据叠加的整体分布（可选）
# sns.kdeplot(adjacent_data.ravel(), label='Overall', color='black', linestyle='--', linewidth=1)

plt.xlabel('Accuracy', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.title('Distribution of accuracy for adjacent pairs (AB to GH)', fontsize=14)
plt.legend(title='Pair', title_fontsize=12, fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()

# 保存图片
plt.savefig('adjacent_pairs_accuracy_distribution.png', dpi=200)
plt.show()