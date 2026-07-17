import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置绘图风格
sns.set_theme(style='whitegrid', font_scale=1.2)

# ---------- 1. 读取数据 ----------
df = pd.read_csv('models/results/batch_test_results_seed_40.csv', index_col=0)

# 配对列表和网络名称
pairs = df.index.tolist()
networks = df.columns.tolist()

# ---------- 2. 定义学习对（8个非相邻对） ----------
learned_pairs = ['A-F', 'B-C', 'B-E', 'C-G', 'D-F', 'D-G', 'E-H', 'A-H']

# 标记每个配对是否为学习对
df['is_learned'] = df.index.isin(learned_pairs)

# ---------- 3. 计算每个网络的平均准确率（学习对 vs 非学习对） ----------
learned_mean = df[df['is_learned'] == True][networks].mean(axis=0)
nonlearned_mean = df[df['is_learned'] == False][networks].mean(axis=0)

group_learned_mean = learned_mean.mean()
group_learned_sem = learned_mean.sem()
group_nonlearned_mean = nonlearned_mean.mean()
group_nonlearned_sem = nonlearned_mean.sem()

# ---------- 4. 计算距离效应 ----------
items = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
item_to_rank = {item: i+1 for i, item in enumerate(items)}

distances = []
for pair in pairs:
    a, b = pair.split('-')
    dist = abs(item_to_rank[a] - item_to_rank[b])
    distances.append(dist)

df['distance'] = distances

distance_means = df.groupby('distance')[networks].mean()
distance_group_mean = distance_means.mean(axis=1)
distance_group_sem = distance_means.sem(axis=1)

# ---------- 5. 绘制在同一画布上 ----------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))  # 创建一行两列的子图

# ---- 图1：柱状图（学习对 vs 非学习对） ----
ax1 = axes[0]
bars = ax1.bar(['Learned pairs', 'Non-learned pairs'],
               [group_learned_mean, group_nonlearned_mean],
               yerr=[group_learned_sem, group_nonlearned_sem],
               capsize=5, color=['grey', 'lightblue'], edgecolor='black')
ax1.set_ylabel('Mean Accuracy')
ax1.set_ylim(0, 1.0)
ax1.set_title('Grand averaged ranking accuracy')
ax1.grid(axis='y', alpha=0.3)

# ---- 图2：折线图（距离效应） ----
ax2 = axes[1]
ax2.errorbar(distance_group_mean.index, distance_group_mean.values,
             yerr=distance_group_sem.values,
             fmt='o-', capsize=5, color='blue', ecolor='gray', elinewidth=2, capthick=2)
ax2.set_xlabel('Rank Distance')
ax2.set_ylabel('Mean Accuracy')
ax2.set_ylim(0, 1.0)
ax2.set_xticks(range(1, 8))
ax2.set_title('Symbolic Distance Effect')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ---------- 6. 打印数值 ----------
print("学习对平均准确率: {:.3f} ± {:.3f}".format(group_learned_mean, group_learned_sem))
print("非学习对平均准确率: {:.3f} ± {:.3f}".format(group_nonlearned_mean, group_nonlearned_sem))
print("距离效应各距离准确率:")
for d, m in distance_group_mean.items():
    print(f"距离 {d}: {m:.3f} ± {distance_group_sem[d]:.3f}")