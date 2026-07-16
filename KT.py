import pandas as pd
import numpy as np
from scipy.stats import kendalltau
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- 1. 读取数据 ----------
df = pd.read_csv('models/results/batch_test_results_seed_123.csv', index_col=0)

# 所有配对名称（按字母升序，共28个）
pairs = df.index.tolist()
# 所有网络名称
networks = df.columns.tolist()

# 真实顺序（假设字母顺序为真实排名，从低到高：A < B < ... < H）
items = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
item_to_idx = {item: i for i, item in enumerate(items)}
n_items = len(items)

# ---------- 2. 构建每个网络的偏好矩阵（8x8） ----------
def build_preference_matrix(net_name):
    """
    根据某个网络对所有配对的正确率，构建8x8偏好矩阵 P，
    其中 P[i,j] 表示网络认为 item_i 优于 item_j 的强度（概率）。
    如果真实顺序中 i < j（即 i 排在 j 前面），
    则 P[i,j] = 正确率（判断 i<j 正确的概率），P[j,i] = 1 - 正确率。
    如果真实顺序中 i > j，则对称处理。
    """
    P = np.zeros((n_items, n_items))
    for pair in pairs:
        # 解析配对，如 "A-B"
        a, b = pair.split('-')
        i = item_to_idx[a]
        j = item_to_idx[b]
        # 获取该网络在此配对的正确率
        acc = df.loc[pair, net_name]
        # 真实顺序中 i 应该在 j 前面（因为字母顺序）
        if i < j:
            P[i, j] = acc
            P[j, i] = 1 - acc
        else:
            # 如果实际配对顺序是反的（但我们的pairs都是升序，所以不会进入这里）
            P[i, j] = acc
            P[j, i] = 1 - acc
    # 对角线设为0.5（无偏好）
    np.fill_diagonal(P, 0.5)
    return P

# ---------- 3. HodgeRank：从偏好矩阵估计排名分数 ----------
def hodge_rank_from_preference(P):
    """
    输入：偏好矩阵 P (n x n)，P[i,j] 表示 i 优于 j 的强度
    输出：排名分数向量 s，数值越大表示排名越靠后（或越优，取决于设定）
    这里我们求解最小二乘：min_s sum_{i,j} (s_i - s_j - P[i,j])^2，且 sum(s)=0
    """
    n = P.shape[0]
    # 构建超定方程组 A * s = b
    A_list = []
    b_list = []
    for i in range(n):
        for j in range(n):
            if i != j:
                # 方程：s_i - s_j = P[i,j]
                row = np.zeros(n)
                row[i] = 1
                row[j] = -1
                A_list.append(row)
                b_list.append(P[i, j])
    A = np.array(A_list)
    b = np.array(b_list)
    
    # 加入约束 sum(s) = 0
    A_aug = np.vstack([A, np.ones(n)])
    b_aug = np.hstack([b, 0])
    
    # 最小二乘求解
    s, residuals, rank, s_vals = np.linalg.lstsq(A_aug, b_aug, rcond=None)
    return s

# ---------- 4. 计算每个网络的排名分数 ----------
rank_scores = {}
for net in networks:
    P = build_preference_matrix(net)
    s = hodge_rank_from_preference(P)
    rank_scores[net] = s  # s是numpy数组，长度为8

# 将分数转化为排名顺序（从小到大，分数低的排名靠前）
# 但Kendall tau可以直接比较分数向量，因为单调变换不影响秩相关。
# 所以我们直接用分数向量计算Kendall tau。

# ---------- 5. 计算网络间相似性（Kendall's tau） ----------
n_nets = len(networks)
sim_matrix = np.zeros((n_nets, n_nets))

for i, net_i in enumerate(networks):
    for j, net_j in enumerate(networks):
        # 使用Kendall tau比较两个网络的分数向量
        tau, p_value = kendalltau(rank_scores[net_i], rank_scores[net_j])
        sim_matrix[i, j] = tau

# 转换为DataFrame方便查看
sim_df = pd.DataFrame(sim_matrix, index=networks, columns=networks)

plt.figure(figsize=(10, 8))
sns.heatmap(sim_df, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            square=True, linewidths=0.5, cbar_kws={'shrink': 0.8})
plt.title('Network Similarity based on Kendall\'s tau (HodgeRank reconstructed ranks)')
plt.tight_layout()
plt.show()

# 接上一步的 sim_df（20x20矩阵）
# 提取上三角矩阵（排除对角线，避免重复计算）
upper_tri_indices = np.triu_indices_from(sim_df, k=1)
all_similarities = sim_df.values[upper_tri_indices]

print("=== 网络间相似性描述性统计 ===")
print(f"网络数量: {len(networks)}")
print(f"有效网络对数量: {len(all_similarities)}")
print(f"平均相似性 (Mean Kendall's tau): {np.mean(all_similarities):.4f}")
print(f"中位数 (Median): {np.median(all_similarities):.4f}")
print(f"标准差 (Std): {np.std(all_similarities):.4f}")
print(f"最小值 / 最大值: {np.min(all_similarities):.4f} / {np.max(all_similarities):.4f}")