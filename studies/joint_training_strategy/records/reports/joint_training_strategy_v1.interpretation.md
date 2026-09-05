# 单阶段配对训练：结果解释与冻结结论

本说明解释已完成的注册比较，不修改协议、结果或判定阈值。数值与逐 seed
判定以[正式结果](../results/joint_training_strategy_v1.json)和
[完整报告](joint_training_strategy_v1.md)为准；方法见
[冻结协议](../benchmarks/joint_training_strategy_v1.json)。

## 结论先行

**单阶段联合训练能够完成当前任务，并在三枚新网络上保留全部九项定性行为规则；
但尚未通过“可完整替代 matched 两阶段流程”的注册标准。**

注册总 outcome 为 `competent_but_not_noninferior`，不是训练失败，也不是
所有单阶段方案不可行。三对条件全部具有任务能力；2108 的 omitted 非劣
区间未达到预设下界，因此不能用另外两对网络或平均结果修复全 seed 结论。

## 已获得的正结果

- 六个模型都通过 generic/Liu 能力门和 combined-policy transitivity 门。
  每个模型训练 48,000 个 episode，严格使用固定最终 checkpoint。
- 三枚 joint 的九项定性行为规则全部通过；2108、2110 同时保留全部六项指定的
  历史定量匹配。2109 的定量缺口仅为 learned accuracy。
- joint 相对 matched staged 的 Liu nonlearned 正确概率分别改善约
  0.98、1.18、0.92 个百分点；各网络 participant-bootstrap 的下界均大于零。
  generic learned/nonlearned 的配对改善下界也均大于零。
- 两种训练方式的六个模型都通过 global necessity、remote reassembly、
  query/evidence specificity 和 local-only partition 四个完整机制链接。
  omitted 的局部直接收益在六个模型中也都通过预设的 materiality 门。

这些结果支持：预设的 P/L 结构可以接受一个共同任务目标的联合优化；
先后冻结参数不是完成本任务或获得这些行为现象的必要条件。它们不证明
双通道自主涌现、结构最简、普遍必要，或 BPTT 是人脑学习机制。

## 为什么没有达到完整替代标准

### 1. 一项配对非劣证据不足，而非已经证明显著退化

以下单位为正确概率的百分点；每个网络独立对 69 名具有 omitted learned
pairs 的虚拟被试进行 complete-case bootstrap，没有跨网络 pooling。

| Seed | omitted joint−staged 均值 | 95% CI | 下界 ≥ −2 pp |
| --- | --- | --- | --- |
| 2108 | −0.3133 | [−2.7008, +1.9130] | FAIL |
| 2109 | +0.5123 | [−1.3807, +2.3417] | PASS |
| 2110 | +1.3290 | [−0.0825, +2.8487] | PASS |

2108 的区间跨越零。正确解释是无法在当前设计下排除超过容忍值的损失，
而不是已经证实 joint 在 omitted 上显著更差。不得据此移动 −2 pp 的门限、
更换 seed、延长训练或把更多被试/网络追加进当前注册比较。

### 2. retained 局部收益的严格门在两种流程中都未通过

注册要求 intact−local_off 在 retained learned pairs 上的正确概率收益
**95% 下界至少为 +1 pp**。joint 的均值约 +0.66 至 +0.83 pp，下界约
+0.35 至 +0.49 pp；matched staged 的均值约 +0.79 至 +0.95 pp，下界约
+0.45 至 +0.59 pp。因此六个模型的 `direct_local_fidelity` 完整链接均失败。

所有这些区间下界仍为正。因此不能说局部路径无效，也不能把两种流程共有的
materiality 缺口归因于共同优化。其余四个机制链接和 omitted rescue 必须保留。

2109/2110 的注册分支标签是 `alternative_computational_solution`，因为它们
通过非劣但未通过完整共同机制门。这个标签是 outcome tree 的分类，不是
“发现全新计算算法”的独立证据，更不是旧功能分工已消失的证明。

### 3. 2109 的一项描述性行为匹配未通过

2109 joint 的 sampled learned accuracy 为 0.936851，冻结人类参考区间上界为
0.933766，超出约 0.31 pp。它仍通过 learned accuracy 的定性规则，但未通过
该行历史定量分类，因此不能算全三网完整行为保留。

这里判断的是点估计是否落入既有人类区间，不是正式的模型—人类等效检验；
也不证明统计显著不一致。不得因为差距小或准确率更高而将此行改判为 PASS。
Liu 温度仍为历史校准的 0.25，没有在新结果后重新拟合。

## 计算成本

不含编译/warmup 的实测训练时间：matched staged 每枚约 49.2–49.4 秒，
joint 约 59.0–59.8 秒；两者峰值 allocated 显存均约 2.33–2.34 GB。
因此单阶段简化了优化流程，但本次没有计算效率优势。相同 episode 暴露量
不代表相同主干更新数或 FLOPs；joint 多执行了 500 次主干更新。

## 证据完整性与边界

实现与最终非 Liu CUDA 校验先锁定，六模型联合 artifact 锁随后在任何新评估前
提交并推送。配对初始化、逐步与累计训练数据指纹、实际 Adam 计数、staged
主干冻结哈希及所有最终文件通过检查。完整科学结果及六份数值型 NPZ 保留在
本 study；没有覆盖历史模型或旧报告。

另从注册 NPZ 直接重算了 24 项 Liu intact 分组概率向量和全部 18 项配对非劣
区间，共 42 项独立数值核对，与正式结果在 1e-14 容差内一致。该核对仅验证
已有结果，没有新增评估、阈值或科学指标。

## 理论更新与停止规则

当前正结果已经把研究推进到“同一任务目标下的共同可学习性”：保持结构先验时，
全局关系构建、局部经验贡献和 Liu 定性现象不必依赖先全局、后局部的优化流程。
但当前配方尚不能升级为通过全部严格门的正式替代主模型。

本轮比较在此冻结。保留单阶段模型为有任务能力的独立候选，不追加 seed、不调
学习率/增益/温度、不更换激活或读出。若未来另行授权，应先区分 omitted
替代效果的不确定性与两种流程共有的 retained materiality 缺口，再注册新的
判别问题；不能把这些不同层次的限制一并归因为“单阶段不可行”。
