# Meta-training plastic RNN to human constructive ranking

本仓库包用于阶段性汇报，整理了从原始 meta-training 代码开始到转移为行为学论文范式的探索与验证过程。

## 0. 仓库结构与当前定位

本文件是仓库唯一的项目总览、机制说明和结果报告。目录按机制谱系而非旧 stage 编号组织：

| 目录 | 内容与状态 |
| --- | --- |
| `research/00_upstream_miconi_kay` | 上游 Miconi & Kay 代码快照，作为参考实现。 |
| `research/01_explicit_score_memory` | 显式 score / edge-memory 基线；已确认存在预设 rank-axis 捷径。 |
| `research/02_implicit_item_state_writer` | dynamic item-state writer；已确认 writer 而非 RNN/plasticity 是主通道。 |
| `research/03_plastic_representation_transformer_v5_v8` | V5-V8 去 writer 的机制检验；尚未稳定形成全局排序。 |
| `research/04_simple_neo_mutations` | 当前主要证据：`simple_neo` v2/v3、评估器及其训练输出。 |
| `archive/legacy_01_08` | 早期 01-08 的源码、结果、文档、diff 与 patch；仅供追溯，不是当前主线。 |
| `references`、`tools` | 参考材料与非实验性维护脚本。 |

资产迁移表位于 `STRUCTURE.md`。当前结论是：v3 已分别复现人类的群体级正确率、symbolic-distance effect、全局自洽性与 pair-level 双峰，但尚未在同一个严格 no-feedback 模型中同时复现全部行为表型。G5-F 的 pair-level 分布最接近人类；G7-F 的距离效应、自洽性和 correct-ranker 比例最接近人类，但表现偏强且主体间趋同过高。

## 1. 早期的显式建模方向：score 范式

### 1.1. 概述

`research/01_explicit_score_memory/constructive_score_ranker.py`（原 `01_04_rnn_integrated_version.py`）采用的是显式 score-coordinate 范式。

模型为每个 item 维护一个 episode 内的标量 score[i]，学习阶段看到 pairwise 关系后，直接更新两个 item 的 score。测试阶段，通过用 score_i - score_j 判断哪一个 item 更高。

在 score 更新之外，还加入了可选的 RNN hidden state、Hebbian fast weights、连接记忆（edge memory）、遗忘干扰（forgetting）、记忆容量（capacity）、范式（schema replay）、测试阶段自增强（self-reconsolidation）等机制。

训练逻辑上，每个 episode 生成一个 ranking 任务，学习阶段暴露少量 pairwise relation，模型根据这些关系更新内部 score / memory / RNN state；测试阶段不给真实反馈，但允许模型根据自身判断进行弱的内部无反馈更新。外层训练仍使用测试 pair 的 ground-truth choice loss 做 meta-learning，因此真实标签不会在测试内部直接反馈给模型，而是只通过 episode 结束后的 loss 更新参数。

在 01_04 score+RNN 范式中，RNN 的作用是作为 episode-level controller。学习阶段，RNN 读取当前 pair、observed relation、当前 score prediction error 等信息，更新内部隐藏状态和可选 Hebbian plastic state；随后内部隐藏状态拼接进 update gate，再来调节两个 item scalar score 的更新幅度。在 edge-memory 模式下，RNN 还会影响 pair relation 写入 memory 的 attention/reliability。测试阶段，RNN 可以在 hidden/self-reconsolidate 模式下继续更新，并在 rnn_choice_gain>0 时向最终 logits 添加 choice bias。

注意：这里 RNN 与 Hebbian 都是可选机制。这是因为，score 范式下，本质上都是维护显式 score 的网络；核心状态不是 hidden state，而是每个 item 的显式 score。开 RNN 与 no-RNN 的区别在于：开 RNN 后，score 更新和 memory 写入会带有历史上下文调制，并可产生额外 choice bias；no-RNN 则完全依赖当前 pair features 与显式 score error。

小规模测试结果：no-RNN、RNN/no-Hebbian、RNN/Hebbian 三组中，score-only prediction 与 full prediction 完全一致，且在 ~30 个 episodes 后都能达到较高正确率，~60 个 episodes 后三组 acc 均接近 1.000。

它证明显式 score-coordinate 更新足以解决任务，但本质上是一个强 independent-value / rank-axis accumulator，无法证明 RNN/Hebbian 的必要性，与 meta-training 论文有本质区别，也难以自然产生人类式自一致的错误排序结果。

它更像“模型知道要维护一个全局坐标轴，并把 pair 误差直接投影到该轴上”。换句话讲：

**已经设计好了解决问题的正确渠道（维护 scalar score / rank-axis），模型只需要学会“解方程组套数字”就能完成任务，而并非让模型主动从 meta-training 中学会“维护 rank-axis 以解决问题”这个底层机制本身。**

05~08 active rank hypothesis sampler 则更加“手写好底层”，因此也被弃用。此处不再细提其范式细节，只当做是初步尝试。

### 1.2. 范式细节

#### 1.2.1. 训练与学习阶段输入

episode 生成逻辑是：

1）每个 batch subject 先随机生成一组 item feature，由 sample_item_vectors 生成的 ±1 二值 item 向量表示；

2）生成 subject latent subject_z；

3）通过 rank_to_item_maps 随机打乱 rank-to-item 对应关系，得到每个 subject 的 true hidden ranking。

真实 ranking 归一化为 true_scores，即 rank 位置对应一个中心化的一维标量。

学习阶段的输入是某个 pair (item_i, item_j) 以及它们的真实 signed difference：observed_diff_j_minus_i = true_score[j] - true_score[i]。

也就是说，模型在 learning phase 直接得到“j 比 i 高多少”的关系量，最多加入 relation noise 或 dropout。

### 1.2.2. 网络结构

核心网络是 ConstructiveRankingNet，是手搓好的专门为任务设计的网络。其中心状态是 episode-local 的 scores[B, N]：

每个 subject：
  item_vecs:      [N, item_dim]
  subject_z:      [subject_dim]
  scores:         [N]        ← 显式 item score
  optional memory:[N, N]     ← 可选 pairwise edge memory
  optional h:     [H]        ← 可选 RNN hidden
  optional P:     [H, H]     ← 可选 Hebbian fast weights

初始 score 由 score_init([item_vec, subject_z]) -> scalar 得到。也就是说，每个 item 一开始就有一个可训练初始化的一维 score。之后所有 pair 判断主要由 score_j - score_i 决定。

对于开 RNN 的情况，PlasticHebbianCore 接收的输入特征包括：两个 item 的向量、subject latent、当前 observed diff、abs diff、当前预测 diff、prediction error。RNN hidden 更新形式为：

h_new = tanh(LN(input_proj(x) + rec_proj(h) + fast_weight_gain * P h))
P_new = plastic_decay * P + eta * h_new h_old^T

其中 eta 由一个 MLP 根据当前输入和 hidden 生成。如果关闭 Hebbian，则 eta=0，P 不更新。

### 1.2.3. 更新机制

在 memory_mode="direct" 下，学习阶段每个 observed pair 直接调用 update_scores。逻辑是：

pred_diff = score_j - score_i
error = observed_diff - pred_diff
gate = sigmoid(update_gate(features, optional h))
step = update_scale * sigmoid(logit_step) * gate
delta = 0.5 * step * error

score_i ← score_i - delta
score_j ← score_j + delta
scores  ← scores - mean(scores)

也就是说，这是一个非常明确的“分数坐标校正器”。它不是让 RNN 自己学会如何表示关系，而是直接把 observed pair 的误差写回两个 item 的 score 坐标。

RNN/Hebbian 只是在这个更新过程中提供 gate 或 choice bias，让 score 更新门控依赖历史上下文，但无法阻止 score 本身成为主通道。

如果换 memory_mode 为 edge_online / edge_block / edge_hybrid 等，则会多一个显式 memory[B,N,N] 和 strength[B,N,N]，把 observed relation 写入 pairwise edge table；随后再从 memory table replay 或 schema reconstruction 到 score。这个设计进一步加强了“显式关系存储”的意味。

### 1.2.4. 测试阶段

测试阶段，模型遍历全部 28 个 pair。选择 logits 主要来自：

logits = beta * [score_i, score_j] + optional RNN choice_bias

也就是基于 scalar scores 的 softmax。

如果 test_update_mode="self_reconsolidate"，测试阶段不使用真实标签，但会根据模型自己的 softmax 概率生成 pseudo relation，再弱更新 score 或 memory。

episode 外层训练 loss 是对全部测试 pair 的 cross-entropy 监督；真实标签不进入 test 内部更新，只作为 episode 结束后的 meta-loss。

### 1.3. 范式合理性与问题

这个范式在工程上是合理的 baseline，模拟了“看到 pairwise relation 后更新 item value”的 independent-value / Q-learning 类机制。

它也与行为学论文中的 classical independent-value account 有对应关系，因为每个 item 都有一个独立标量值，局部 pair 更新会逐步推动这些值接近真值轴。

但它不适合作为主机制模型，原因是：它的核心状态就是显式 score。只要 score 分支能工作，RNN 和 Hebbian 就没有必要成为信息瓶颈。


## 2. 不再显式维护 score / edge memory 的 implicit plastic RNN 范式

### 2.1. 概述

为避免 score 分支吸走全部任务，后续切换到 implicit_plastic_rnn_ranker。

该版本明确取消了强捷径：不再让模型内部维护 scalar scores[i]，不再维护显式 pairwise edge_memory[i,j]，改为维护 dynamic item representation e_i(t)、可选 RNN hidden state h_t、可选 Hebbian fast weights A_t，再通过 antisymmetric comparator 判断 pair。

训练与更新逻辑也相应改变：学习阶段的每个 observed pair 不再直接改 scalar score，而是通过 observe_pair 更新 RNN 隐藏状态（可选 plastic matrix），并由 item-state writer 修改两个 item 的 representation；测试阶段没有真实反馈，但可以用模型自己的 pseudo-relation 做 self-reconsolidation。外层仍使用测试 pair 的 choice loss 做 meta-training。

小规模机制隔离测试结果表明：模型之所以能学，主要不是 RNN/Hebbian，而是 dynamic item-state writer。当关闭 item-state update 后，no-RNN、RNN/no-Hebbian、RNN/Hebbian 都回到接近随机；只要打开 item-state writer，即使 no-RNN/no-Hebbian 也能表现很强。对此的判断是：当前 implicit 成功的核心机制是 item-state writer，而不是 plastic RNN。

因此，该版本证明，去掉 explicit score 后，模型仍可能通过另一个强 writer 捷径解决任务。换句话讲：

**换汤不换药：改了 score 的底层表示形式，但模型底层仍通过 item_state 捷径维护，RNN 仍不必要。**

### 2.2. 范式细节

#### 2.2.1. 训练与学习阶段输入

episode 输入仍然是：

subject_z
rank_to_item permutation
true_scores_by_item
learning_edges: 8 个 observed pair
relation rel = true_score[j] - true_score[i]

所以学习阶段仍然直接给 signed relation magnitude。与 01_04 的区别不在输入，而在内部状态和更新载体。

#### 2.2.2. 网络结构

ImplicitPlasticRanker 的核心状态是：

item_state: [B, N, D]  ← 每个 item 一个动态向量
h:          [B, H]     ← global recurrent hidden
plastic:    [B, H, H]  ← 可选的 Hebbian weights

每个 item 有一个可训练的 base_item[N,D0]。episode 开始时，base_item 与 subject_z 拼接，通过 item_init 得到初始 item_state。如果启用 subject modulation，则每个 subject 的 item 初始状态都会被 subject_z 调制。

每个 pair 的 context 由以下部分组成：
e_i
e_j
e_j - e_i
e_i * e_j
relation features: [signed relation, magnitude, confidence, is_test, is_self, bias]
subject_z

RNN 更新仍然是：

h_new = tanh(LN(rnn_input(x) + rnn_rec(h) + fast_weight_gain * plastic h))
plastic_new = plastic_decay * plastic + eta * h_new h_old^T

write_items 更新：

write_head([context, h]) → Δe_i, Δe_j
gate_head([context, h])  → gate
e_i ← decay * e_i + scale * gate * Δe_i
e_j ← decay * e_j + scale * gate * Δe_j

这就是 dynamic item-state writer。它能直接修改两个 item 的表示。

读出则是 antisymmetric comparator：

logit(i,j) = beta * [phi(e_i, e_j, e_j-e_i, h, z) - phi(e_j, e_i, e_i-e_j, h, z)]

正 logit 表示判断 j 高于 i。

#### 2.2.3. 更新机制

learning phase 中，每个 observed pair 调用 observe_pair：

pair_context(...)
state, eta = rnn_step(context, state)
state, gate = write_items(context, state, item_i, item_j)

所以一个 observed relation 会同时更新 global hidden / plastic matrix 和两个 item 的动态表示。

也就是会所，即使关闭 RNN/Hebbian，只要 use_item_state_updates=True，writer 仍然可以直接把 observed relation 写入 item_state。

测试阶段，模型对全部 28 个 pair 做 binary cross-entropy。若 test_update_mode != frozen，模型会根据自己的 logit 生成 pseudo relation 和 confidence，再把这个 pseudo relation 通过同样的 observe_pair 写回 hidden / plastic / item_state。真实标签只用于 outer-loop loss，不用于 test 内部更新。

#### 2.2.4. 范式合理性与问题

V2/V3/V4 核心价值是完成“去显式 score 化”：模型不再维护一维 score state，而是维护高维 item_state。工程上也修了 batch indexing、AMP 下 entropy NaN、orientation randomization 等小 BUG。从行为学论文的神经表征角度看，item representation 被学习后重组，这一点是有意义的。论文 MEG 部分确实强调 learning 后 item-wise neural similarity 会对齐个体 subjective ranking，而不是 shared ground truth。

但从机制检验角度看，implicit V2/V3/V4 仍不是真正的 RNN/Hebbian 机制模型。原因是它引入了一个更直接、更容易优化的通道：

observed pair → write_head → 直接修改 e_i 和 e_j → comparator 直接读 e_i/e_j

这个机制相当于给模型一个外部可写 item memory。它不是显式 score，但它仍然是一个强记忆通道。跨 trial 信息不必通过 recurrent plastic weights 保存，而可以直接保存在每个 item 的动态向量里。

机制上，这与 Miconi & Kay 的差异更大，因为 Miconi 的关键约束是 trial 间记忆压缩在 recurrent plastic weights 中，而不是给每个 item 一个可直接写入的持久状态。Miconi 论文中 plastic weights 在 episode 内更新且不重置，而每个 trial 的 neural activations 和 eligibility trace 会重置，从而强迫跨 trial 记忆主要进入 plastic weights。

## 3. 从行为学论文范式开始逼近 Miconi & Kay 的 plastic RNN 尝试

### 3.1. 概述

在确认 item-state writer 是新的捷径后，V5 开始，模型去掉 item-state writer 对 item 表征的写入，而是尝试让 recurrent controller、Hebbian plastic matrix 和 active reinstatement 共同改变 item 的可读出表征。也就是说，模型需要通过 episode 内的 plastic weights 来形成关系结构，而不是直接把 pair 信息写到某个显式表里。

一个 episode 可以抽象成下面这个过程：
1. 生成 episode
  生成 subject_z，8 个 item code，true ranking / true_scores
  生成 8 个 learning_edges（由 paper_graph_train_prob 参数调节固定使用行为学论文 8 条 learning pair 与随机 8 条的比例）
  生成 28 个 test_pairs

2. 初始化模型 episode-local state
  h_0 = 0 或 learned init
  trace_0 = 0
  P_0 = 0
  item codes c_1...c_8 固定

3. learning phase
  依次呈现 4 blocks × 8 learning pairs
  每个 learning pair 根据不同版本更新 h_t / trace_t / P_t

4. testing phase
  用 learning 后得到的 final P_t / h_t
  判断全部 28 pairs
  test phase 无真实反馈

5. 计算 episode loss
  test BCE loss
  + learning auxiliary / margin loss
  + entropy / plastic / trace regularization

6. outer-loop 更新网络参数
  Adam 反传更新固定参数
  下一个 episode 重新初始化 h, trace, P

V5 的基本范式是：学习阶段只给 observed pair relation，测试阶段不给 true feedback；模型通过 recurrent plastic state 生成 transformed item representation，再由 score/readout 判断 pair。结果显示，no-RNN/no-Hebbian 和 RNN/no-Hebbian 接近随机，Full 组虽然 plasticity 非零，但 accuracy 长期接近 0.5，score 分布也出现塌缩。

V6 在此基础上加入 observed-pair replay 和 learning auxiliary scaffold。注意 auxiliary loss 和 replay 只作用于学习阶段实际出现过的 pair，不把未观察过的 pair 答案泄露给模型。一方面，这加强模型对 observed relation 的编码，另一方面也是避免信息泄漏。但结果仍然接近随机，Full 组 plasticity 很活跃，P(t) 变大了，但 learning auxiliary accuracy 仍接近随机，说明 plastic update 没有学成稳定的关系方向。

V7 进一步把 learning trial 改成 predict → observed feedback → plastic update：模型先对 observed pair 做预测，再揭示 relation，再用 feedback error 触发 neuromodulated plastic update；测试阶段仍然 frozen / no true feedback。结果仍没有实质起势，overall、learned、nonlearned 和 aux accuracy 都没有明显上升。

V8 进一步向论文驱动的 plastic representation transformer 靠拢，保留行为学论文的 fixed sparse learning graph 和 no-feedback test，但不再依赖显式 score、edge table 或高容量 writer。这个方向在机制上最干净，但 10k episodes 左右训练后仍只得到弱学习：模型能形成一些局部 scaffold，却难以稳定建立全局 rank-axis。

因此，V5–V8 的总体结论是：去掉捷径后，当前架构还没有找到能稳定把 observed pair relation 写入 recurrent plastic memory 的机制。

### 3.2. 范式细节与分版本评估

#### 3.2.1. 网络结构

该版本主体网络均为 PlasticReinstatementRanker，其底层结构为：

static item code
→ plastic representation transformer
→ pair comparator

learning pair
→ recurrent controller
→ Hebbian trace / plastic matrix 更新
→ 改变后续 item representation

V5–V8 不再有 direct item-state writer，不再允许 observed pair → write_head 直接改 e_i / e_j，而强迫模型通过 observed pair → RNN controller → plastic matrix P(t) 改变 item 表征来完成 episode 内学习。这与 Miconi 论文的范式最为接近。

1. episode 开始：生成 item code，但不是 item state

每个 episode 有 8 个 item。前面 implicit V2/V3/V4 里，每个 item 有一个会被直接修改的动态状态 e_i(t)。

在 V5–V8 里，item 本身先被编码成基本固定的 code：

item_id / item_feature + subject_z → item_code_i

可以理解成 codes = [c_1, c_2, ..., c_8]，其中 c_i 是第 i 个 item 的基础表示。

这个 code 在 episode 内不被 writer 直接改写，而是 item 的稳定身份向量。

这样，不再像 V1~V4 中模型能够维护并直接改变内部关于 item 的表示，现在 episode 内只能变全局 recurrent/plastic 状态：

h_t       # RNN hidden
trace_t   # Hebbian eligibility trace
P_t       # plastic fast-weight matrix

整体状态可以写成：

State(t) = {
    h_t:       [B, H],
    trace_t:   [B, H, H],
    plastic_t: [B, H, H]
}

2. plastic representation transformer：P(t) 如何改变 item 表征

每次做 pair 判断时，网络用当前 plastic matrix P_t 重新变换所有 item code：

c_i + P_t → ψ_i(t)

这一步可以理解为：

ψ_i(t) = transform(c_i, P_t, subject_z)

更具体地说，源码逻辑可简化为：

fixed_i = learned_fixed_transform(c_i)
fast_i  = P_t * c_i
ψ_i     = tanh(layer_norm(fixed_i + fast_weight_gain * fast_i + subject_bias))

所以，item 表征不是被单独写入的，而是被同一个 episode-local plastic matrix 统一变换出来的。

对比 V2~V4 明显发现：

implicit V2/V3/V4 中，每个 item 有自己的动态槽位 e_i(t)，可以被直接写。

V5–V8 中，每个 item 只有固定 code c_i；episode 内变化的是 P_t，item 的动态表征 ψ_i(t) 是 c_i 经过 P_t 变换后得到的。

网络因此得名 plastic representation transformer。

注意这里有别于 Miconi 论文本身。Miconi 原模型里，pw 是 recurrent weights 的 episode-local 增量：

h_t = f(W_in x_t + (W_rec + A ⊙ P_t) h_{t-1})
P_{t+1} = P_t + m_t H_t

P_t 只通过改变 recurrent dynamics 来改变后续行为。

这里复用了这一套机制，见 4. learning pair 更新 P(t)，但多出“由 P_t 重塑输入向量表征”这一步。

为什么？

**因为行为学 ranking 任务和 Miconi 原任务有一个额外困难：测试时需要对任意 item pair 做比较。也就是说，模型不只是要根据当前 stimulus 输出动作，还要把之前 sparse pair experience 组织成一套 item-wise relational representation。”**

因此，P_t 被赋予了“最终学习状态”的角色，它作用到所有 item code，得到所有 item 当前表征 ψ_1(t), ..., ψ_8(t)，才可能使模型有希望对任意 pair 都可以比较。

如果只把 P_t 用在 RNN hidden 上，而不重新作用到 item code，那么 test pair (i, j) 无反馈地到来时，模型将难以将 test pair 的 item identity 与 episode 内学到的关系记忆对齐。

**Note:** 在 review 时，发现这个做法也有继续改进的空间。**调制 item code 不一定非要直接拿同一个 P_t 矩阵去乘 item code。**同一个矩阵既负责 recurrent dynamics，又负责 item representation transformation，在机制上可能有些粗暴。可以考虑单独有一个 item-code transformer：

T_t = g(h_t, P_t, subject_z)

item_i 当前表征：
ψ_i(t) = T_t(c_i)

pair 判断：
logit(i,j) = comparator(ψ_i(t), ψ_j(t))

依然注意这个 transformer 必须是 episode-global、共享的、由 RNN/plastic state 调制的 readout operator，不能变成对每个 observed item 直接写入的 item-state writer。

3. pair comparator：如何从 item 表征做判断

得到每个 item 的当前表征 ψ_i(t) 后，网络会给每个 item 读出一个 scalar：

score_i(t) = score_head(ψ_i(t))

如果启用 hidden readout，还会从 RNN 隐藏状态 h_t 生成一个 global axis 辅助：

axis_t = f(h_t, subject_z)
score_i(t) += hidden_readout_gain * dot(ψ_i(t), axis_t)

最后 pair (i, j) 的 logit 是：

logit(i,j) = β * [score_j(t) - score_i(t)]

这里容易误解：V5–V8 仍然有 score_head，但此 score 非彼 score，不是 explicit score memory。

区别在于，01_04 explicit score 中 score_i 是 episode-local 状态变量，会被直接更新，而这里 score_i 是从 ψ_i(t) 即时读出来的结果，episode 内不维护、不更新，真正被更新的是 P_t / h_t / trace_t。

4. recurrent controller：learning pair 如何更新 P(t)

学习阶段，每看到一个 pair (i, j)，模型先用当前 P_t 得到 ψ_i(t), ψ_j(t)，然后拼接构造 pair context：

x_t = [
    ψ_i,
    ψ_j,
    ψ_j - ψ_i,
    ψ_i * ψ_j,
    relation_features,
    subject_z
]

其中 relation_features 包含 observed relation 的方向/大小、phase 标记、confidence、是否 replay/test 等信息。V7/V8 中进一步把反馈信号信息组织进 update 流程。

然后 controller 更新 hidden：

h_{t+1} = tanh(
    LN(
        input_proj(x_t)
        + recurrent_proj(h_t)
        + fast_weight_gain * P_t h_t
    )
)

由 hidden 和 context 产生 neuromodulation：

mod_t = mod_head(x_t, h_{t+1}, subject_z)

Hebbian trace 更新：

trace_{t+1}
= trace_decay * trace_t
  + (1 - trace_decay) * outer(h_{t+1}, h_t)

plastic matrix 更新：

P_{t+1}
= plastic_decay * P_t
  + mod_t * trace_{t+1}

后面的这套 Hebbian learning 形式和 Miconi/Kay 的机制接近。Miconi 模型里 episode 内只有 plastic recurrent weights 更新，普通参数在 episode 内固定；eligibility trace 由前后时刻神经活动外积产生，再由 neuromodulation signal 写入 plastic weights。

#### 3.2.2. V5：passive observe + plastic update

V5 更新方式：

observed relation → pair_context → controller_step → update h / trace / plastic

可选的 reinstatement_steps 会根据当前 h 对 item codes 做 attention，取出某些 item 的 recoded representation，再构造 replay context 推动 controller 更新。

V5 的问题是：虽然 plasticity 是活的，P(t) 会变化，但 outer loss 主要来自最后 all-pair test 的 cross-entropy。对每个 learning pair 的 plastic update 来说，缺少局部 feedback 或局部 prediction loss，导致 neuromodulation 很难知道当前 relation 应该怎样改变 P。结果就是 score/readout 容易塌缩，训练长期接近随机。

~10000 episode 结果：acc 一直在 0.49–0.52 附近，表明几乎随机乱选，choice loss 长期接近 0.693，score_std 从早期约 0.083 持续塌到 0.01 左右；同时 eta 和 |P| 并不是零，说明 plasticity 活着，但没有变成可读出的 rank representation。

#### 3.2.3. V6：observed-pair replay 与 learning auxiliary scaffold

V6 没有回到 writer，而是在 V5 基础上加了两部分：

1. learning-pair auxiliary/margin loss: 针对 learning phase 实际出现过的 pair 作 loss 形成局部监督。也就是：
对于 shown pair(i,j):
    logit(i,j) = comparator(ψ_i, ψ_j)
    target = observed relation direction
    aux_loss = BCE(logit, target)
    weak_margin_loss = max(0, margin - y_ij · logit_ij)

2. observed-pair replay buffer: 维护一个 observed buffer，每看到一个新的 learning pair，就把它放进去。replay 时，从 buffer 里抽已经出现过的 pair。

里面的关键限制是：auxiliary 和 replay 只使用 learning phase 实际出现过的 observed pairs，不使用 unobserved test pair 的标签。因此它不泄露 all-pair test 答案。

相比 V5，这一步让 plastic RNN 不再完全靠 episode 末尾 test loss 反推，而是在 observed relation 上获得局部 scaffold。

但小规模 ~500 episode 测试显示，结果仍不理想，辅助头输出完全没有起势，连 observed-pair aux_acc 都没起来，表明连“能记 observed、不能泛化”的阶段都没到，而是 P(t) 的方向性语义都尚未建立。

#### 3.2.4. V7：predict → observed feedback → plastic update

V7 进一步修正 V6 的问题：V6 虽然加了 auxiliary，但它仍然更像“看到 relation 后顺手加一个局部约束”，而不是形成完整的 trial-level learning loop。

V7 改为对每个 observed learning pair：

1. 先用当前 state 预测该 pair
2. 再揭示 observed relation / feedback
3. 用 prediction error 或 feedback signal 驱动 plastic update

这个改动更接近 Miconi/Kay 的 trial 结构：先 action / prediction，再 reward / feedback，再通过 neuromodulated plasticity 改变未来表现。它比 V5/V6 更适合检验“plastic RNN 是否能形成 episode 内学习算法”。

但它和行为学论文也自然产生了差距：行为学 learning phase 是 passive observation，参与者不需要在 learning trial 上作答；V7 加了 learning-trial prediction/feedback，这更像训练人工网络的 curriculum，而不是严格的人类任务复现。

结果上，小规模 ~500 episode 测试显示，V7 P(t) 很大，trace 很大，但 learning_aux_accuracy 仍然接近随机，表明 P(t) 对 item representation 作全局变换仍然难以让模型学会任务。learning-pair auxiliary accuracy 指标、observed_score_std 指标等均无早期起势效果。

#### 3.2.5. V8：feedback curriculum + observed replay + zero-feedback deployment

V8 是 V7 的进一步工程化：加入 staged feedback curriculum。源码中 run_learning_phase 结构为：
predict → observed feedback → plastic update loop

每个 learning pair 先预测，再计算 feedback signal，再把 relation 与 feedback_signal 输入 controller。feedback 可以是 reward（答错或答对，+1/-1）、soft_reward（target_sign * tanh(logit)，confidence-weighted）、target_sign（真值 +1/-1） 或 signed_error（target_sign - tanh(logit)，directional prediction-error feedback）；并且 feedback probability 可以从 stage1 高反馈逐步退火到 stage3 zero-feedback。V8 也保留 observed-pair replay，但默认关闭 free-form reinstatement，强调 grounded replay。测试阶段默认 test_update_mode="frozen"，eval_feedback_prob=0.0，也就是最终部署仍是 no-feedback test。

V8 的 loss 由几部分组成：

test choice_loss
+ learning_aux_loss
+ observed-pair margin_loss
+ entropy regularization
+ plastic / trace / eta regularization
+ optional score_std_floor

因此 V8 的机制已经很清楚：希望通过 shown pair 的预测和反馈，把 relation 写进 recurrent plastic matrix，使 P(t) 改变全部 item 的 transformed representation，从而在 all-pair test 上产生 rank-axis。

10000 episode 结果显示，从 ~3000 episode 开始，acc 就达到 0.60 并在 0.58–0.60 附近浮动，aux_acc 也在 0.60 左右，表明并非完全随机；但最终 behavioral-paper eval 只有：

overall_accuracy ≈ 0.540
learned_pairs_accuracy ≈ 0.542
nonlearned_pairs_accuracy ≈ 0.539
learning_aux_accuracy ≈ 0.753

对比上，这与 Miconi 的原始 simple_neo.py 结果完全不同：simple_neo 在 ~3000 左右基本高于 0.8 之后 ~5000 episode 整体基本全对；与这里测试出的“ ~3000 开始进入瓶颈期”的结果完全不同。

解释上，auxiliary 已经明显学到 observed-pair scaffold，但 all-pair generalization 仍然很弱。它学会了一定的局部结构，但没有稳定形成全局 rank-axis。10k 后如果 learned 和 nonlearned 都只有 0.54 左右，那么继续堆到 30k 也难以自然跳到真正全局排序。

**Note:** Review 时发现一种可能改进。Miconi 论文中 test trials 仍有真值 feedback，因此，可以基于 V8 考虑加入预学习的 episode stage 退火 V9：

Stage 1：全 pair feedback，学习“游戏规则”

早期 episode 中，learning pair / test pair 都给 feedback。

这相当于给模型建立人类已有的先验性直觉。注意：这里仍需要模型“自己学会”规则本身，不等同于“告诉规则”，Miconi 论文中正是这样做的。

Stage 2：只在 learning pair 给 feedback，test 无反馈

中期退火到：

learning pairs: 有 feedback
test pairs: no feedback

这一步开始接近行为学任务。模型必须学会只根据少量 learning pair 更新内部状态，然后在 test all pairs 时不再依赖 test feedback。


Stage 3：learning 也只观察 relation，不给额外 feedback

最后退火到：

learning pairs: 只观察 relation / outcome
test pairs: no feedback
external feedback_signal = 0

也就是回到行为学论文范式：learning phase 给被试看训练 pair 的关系，testing phase 无反馈判断全部 pair。

注意这里要区分 observed relation ≠ feedback_signal：

即使 feedback_signal = 0，learning pair 的 relation 仍然是 observed 的。否则模型连学习信息都没有。最终行为学范式应该是 learning phase 观察 pair relation，test phase 不给 feedback，而不是整个 episode 完全没有 relation 信息。

V9 前期给 test pair feedback，有可能让模型先经历大量完整 pair space，从而学到不同 pair 不是独立的，它们都来自一个 latent order；反馈应该更新这个 latent order。但学会这个底层机制本身，仍是 meta-training 的任务；预学习阶段 test trials 也给 feedback，不代表就是透露底层结构，这与 Miconi 范式更相符合。

如果尝试该思路，务必注意预学习 stage（Stage 1）中 28 个全测试 pair 要在不同 episodes 中打乱顺序；另外，如果直接学全部 pair 的真值有泄露风险，也可卡绿进一步像 Miconi 范式靠齐，每个 episode 选不同随机 pair 作 test trial。关于 feedback，建议 reward / soft_reward，不要直接 target_sign + magnitude 学精确差值加减。

## 4. 回到 Miconi & Kay 范式本身的 mutations 尝试

### 4.1. 概述

V5–V8 是从行为学任务出发，逐步限制显式 score、edge memory 和 item-state writer，希望最终逼近 Miconi & Kay 式 neuromodulated plastic RNN。结果表明，一旦移除这些高容量写入通道，模型虽然可以产生非零 Hebbian trace 和 plastic weights，却很难把少量 observed pair relation 稳定组织成可供全部 28 个 pair 读取的全局排序结构，准确率上基本可视为失败。

因此，本节采用相反路线：不再继续从复杂的行为学架构向 Miconi 靠拢，而是直接从已经可以稳定学习的 single_neo.py 出发，每次只替换一个任务条件，观察性能在哪一步下降。这样做的目的不是立即得到最终模型，而是把两个问题拆开：

1. **能力问题**：原始 simple_neo plastic RNN 是否有能力学习 8-item sparse graph；
2. **训练信号问题**：active choice、reward、test feedback、outer-loop test loss 中，哪些是必要的；
3. **机制问题**：模型是否在 learning phase 后已经形成 episode-persistent 的排序状态，还是依赖 test-time plastic update 边测试边继续改写；
4. **行为表型问题**：模型能否不仅达到合理的正确率，还同时产生论文所报告的自洽错误、pair-level 双峰分布、稳定错误和个体化排序。

无 distance、更加接近 Miconi 范式的 simple_neo_mutants_v2.py 从原始 Miconi 范式开始，依次考察固定 8 个 item、替换为行为学 8 条 sparse learning pair、取消测试反馈、改成 passive observation、加入 outer-loop test CE、加入 learning auxiliary 等变化。v2 结果：行为学 sparse graph 本身并不是 simple_neo 学不会的；真正敏感的是训练信号和测试阶段的状态读取方式。

随后建立 single_neo_mutants_distance_input_v3.py。v3 保留 simple_neo 的四步 trial、每个 trial 重置 hidden state 和 eligibility trace、episode 内持续保存 plastic weights 的基本结构，只增加一个独立的 signed rank-distance 输入：

distance = (second_item_rank - first_item_rank) / (N - 1)

该距离只在 learning pair 呈现时进入输入。正值表示当前画面中第一个 item 排名更高，负值表示第二个 item 排名更高，绝对值表示两者的 rank distance。默认测试阶段 distance 为 0。

v3 的评估分成两层：

- 标准 full28 eval：在训练式 learning phase 后，冻结测试期 plastic weights，测试全部 28 个 pair，用于判断 learning 后是否已经形成可泛化结构；
- 论文对齐 eval：77 名模型受试者，4 个 learning blocks，每个 block 包含 8 条行为学 learning pair；10 个 test blocks，每个 block 包含全部 28 个 pair；使用 shared cue set 加 subject-specific rank permutation、sample readout、T=1、phase_rescale、严格 tie 处理，并分别比较 test plastic active 与 frozen。

### 4.2. 无 distance 输入的 v2 版本

#### 4.2.1. simple_neo

3000 episode 结果：
test ≈ 0.963, adjacent ≈ 0.929, nonadjacent ≈ 1.000, mean_abs_pw ≈ 3.09

simple_neo / Miconi 式 neuromodulated plastic RNN 正控成立。

#### 4.2.2. n8_fixed

3000 episode 结果：
test ≈ 0.725, adjacent ≈ 0.595, nonadjacent ≈ 0.768, mean_abs_pw ≈ 2.08

固定 8 个 item 后明显变难，邻近对正确率低但后期高于 0.5 表明不是瞎蒙；非邻近对则远高于瞎蒙水平。

V5–V8 不能学出来，不是“行为学任务 8 个 item 太难”所致。

#### 4.2.3. behavior_graph_rewarded

3000 episode 结果：
test ≈ 0.856, adjacent ≈ 0.688, nonadjacent ≈ 0.909, mean_abs_pw ≈ 2.14

learning phase 训练时从邻近对换成行为学论文的 8 条 sparse non-adjacent learning pairs，train 和 test 仍然 choice + reward，结果成功。

表明只要保留 simple_neo 的 active choice/reward loop，behavior graph 是可以被 plastic RNN 学出来的。

#### 4.2.4. behavior_graph_no_test_feedback

训练阶段有 choice + reward，测试阶段无 reward。

普通 full28 non-freeze eval：
overall ≈ 0.725, learned ≈ 0.829, nonlearned ≈ 0.684, nonadjacent ≈ 0.777, mean_circular_triads ≈ 6.18, pw_after_learning ≈ 1.565, pw_after_full_test ≈ 2.967

freeze-test-plastic eval：
overall ≈ 0.801, learned ≈ 0.954, nonlearned ≈ 0.740, nonadjacent ≈ 0.868, mean_circular_triads ≈ 0.249, transitive_triplet_fraction ≈ 0.996, pw_after_learning = pw_after_full_test ≈ 1.565

这组反映两个事实：

第一，test 无反馈不是根本障碍。因为 freeze 后 overall 能到 0.80，nonlearned 能到 0.74，说明 learning phase 后已经形成了可泛化结构。

第二，non-freeze 比 freeze 差，是因为 test 阶段继续 plastic update 造成了 drift。pw_after_full_test 从 1.565 涨到 2.967，说明全 28 pair 测试过程中，plastic weights 被无反馈测试流继续扰动，反而破坏了已形成的结构。

#### 4.2.5. observational_learning

训练阶段 passive observation，测试阶段 rewarded。

3000 episode 结果：
test ≈ 0.794, adjacent ≈ 0.542, nonadjacent ≈ 0.882, mean_abs_pw ≈ 2.24

这组说明 passive observation 也可用。只要 test 仍然有 reward，网络仍然可以通过测试阶段的 RL 信号学出有效策略。

#### 4.2.6. observational_learning_no_test_feedback

训练 passive，测试无反馈，也没有 supervised test loss。这是正常负控，要证明没有泄露路径。

3000 episode 结果：
test ≈ 0.497, nonadjacent ≈ 0.515, mean_abs_pw ≈ 0.010

没有 train reward，没有 test reward，也没有 supervised outer loss，模型没有任何有效学习信号，当然不学。符合预期。

#### 4.2.7. observational_learning_no_test_feedback_with_test_loss

训练 passive，测试无反馈，但 outer-loop 加 test CE loss。

3000 episode 结果：
test ≈ 0.772, adjacent ≈ 0.554, nonadjacent ≈ 0.848, test_ce ≈ 0.375, mean_abs_pw ≈ 2.20

full 28-pair eval，non-freeze：
overall ≈ 0.825, learned ≈ 0.982, nonlearned ≈ 0.763, adjacent ≈ 0.598, nonadjacent ≈ 0.901, mean_circular_triads ≈ 0.166, transitive_triplet_fraction ≈ 0.997
distance accuracy: (d1,d2,...,d7) ≈ (0.598, 0.797, 0.875, 0.956, 0.984, 0.998, 0.999)

freeze-test-plastic：
overall ≈ 0.824, learned ≈ 0.981, nonlearned ≈ 0.761, nonadjacent ≈ 0.901, mean_circular_triads ≈ 0.155, transitive_triplet_fraction ≈ 0.997

这是目前最重要的结果，说明行为学式 passive learning + no-feedback test，在 simple_neo/Miconi-style plastic RNN 架构下是可学的。

而且 freeze 与 non-freeze 结果基本相同，说明不是靠 test-time plastic update 继续“做题时学习”得到的，而是在学习后已经形成了一个稳定的 pw 结构。full test 中继续不继续更新，对结果几乎无影响。

这点和第4组相反。第4组 non-freeze 会 drift，第7组不会。这说明第7组更接近学习阶段形成内部 ranking state，测试时冻结/稳定读取该 state。

这是机制上最重要的正结果。

#### 4.2.8. train_aux-only

训练 passive，测试无反馈，只加 observed-pair train auxiliary，不加 test CE。

3000 episode 结果：
test ≈ 0.491, train_aux ≈ 0.692, mean_abs_pw ≈ 0.078

表明只监督 observed learning pairs 不足以诱导 global rank-axis。

也就是说，如果 loss 只要求模型在 learning phase 的已见 pair 上给出方向判断，它不会自动学会 all-pair transitive inference。它甚至连 observed pair auxiliary 都没有真正学起来。

这与 V8 结果一定程度上呼应：V8 scaffold 很大一部分正是 learning auxiliary / observed-pair margin / replay。但这组结果说明，observed-pair scaffold 可能不是正确方向。

#### 4.2.9. train_aux + test_loss

3000 episode 结果：
test ≈ 0.491, train_aux ≈ 0.691, test_ce ≈ 0.690, mean_abs_pw ≈ 0.070

这是最反直觉但很重要的一组：加了 test CE 本来应该像第7组一样学起来，但一旦再加 train_aux，反而整个系统卡死在 chance 附近。

这说明当前 train_aux 写法不是帮助，而是干扰。原因大概率是：

train_aux 要求模型在 passive learning trial 上做局部 pair 分类，但 test CE 要求 learning 后形成 final global state；
这两个目标在当前时间结构中不是同一个目标，甚至会冲突。

train_aux 可能把优化压力放在“当前 shown pair 的即时分类”上，而不是“把该 pair 写进 episode-persistent pw 以支持后续 all-pair test”。结果就是 pw 没长起来，test CE 也下不去。

所以第9组反向确认了沿着 train_aux / observed-pair auxiliary 方向堆未必可行。

### 4.3. 加入 signed rank-distance 输入的 v3 版本

#### 4.3.1. v3 的输入与任务变化

v3 只对 behavior-graph 和 observational-learning 系列提供 distance input，exact_simple_neo`与 n8_fixed 由于仍为 adjacent learning pair，仍保持原始 reward-only 对照。

主要实现约束如下：

1. distance 使用显示方向编码，范围归一化到 [-1, 1]；
2. distance 只在 learning pair 本身出现的 step 输入，不在 response、reward 或 previous-action step 重复输入；
3. 默认 distance_input_train_only=True，测试阶段 distance 恒为 0；
4. passive observational learning 不要求 action，也不返回 reward；
5. test CE 只作为 outer-loop loss，不作为 episode 内输入；
6. test plastic 是否冻结由 evaluator 控制，用于区分 learning 后形成的状态与 test-time drift。

#### 4.3.2. 四个主要组的 3000-episode 训练结果

当前 v3 已完成第 3、4、5、7 组初步测试，训练参数保持一致：seed=1、batch_size=32、hidden_size=200、cue_size=15、lr=1e-4、lpw=1e-4、3000 episodes。

| 组别 | variant | learning phase | training test phase | Episode 2999 test | adjacent | nonadjacent | mean-abs pw |
|---|---|---|---|---:|---:|---:|---:|
| 第3组 | `behavior_graph_rewarded` | active choice + reward + distance | rewarded | 0.809 | 0.636 | 0.864 | 2.498 |
| 第4组 | `behavior_graph_no_test_feedback` | active choice + reward + distance | no feedback | 0.491 | 0.468 | 0.498 | 1.748 |
| 第5组 | `observational_learning` | passive observation + distance | rewarded | 0.897 | 0.740 | 0.947 | 0.706 |
| 第7组 | `observational_learning_no_test_feedback_with_test_loss` | passive observation + distance | no feedback + outer test CE | 0.906 | 0.740 | 0.959 | 0.585 |

结果如下：

1. 第3组仍能学习，但加入 distance 后并没有超过 v2 第3组。

2. 第4组在 learning phase 的 train performance 接近满分，但 no-feedback 测试退化为了随机瞎蒙。这说明模型可以直接利用当前 trial 的 distance 完成 rewarded learning choice，却不会在缺少 test objective 时自动把关系写成可供 no-feedback test 读取的全局排序状态。

3. 第5组和第7组均显著优于对应 v2 结果。尤其第7组在 passive learning、test no feedback 的条件下仍达到较高表现，说明 signed distance 加上 outer-loop test CE 可以促使 plastic RNN 把 sparse relation 转化为后续可读出的排序结构。

#### 4.3.3. 标准 full28 freeze-test-plastic 结果

下面的 full28 结果使用 512 个 evaluation episodes、greedy readout、随机 pair orientation，并在测试期间冻结 plastic weights。它主要用于回答：**learning phase 结束时，模型的 episode-persistent state 是否已经足以支持全部 28 个 pair。**

| 组别  | overall | learned | nonlearned | adjacent | nonadjacent | circular triads(循环三元组) | Kendall τ to true(衡量与真实排序一致性) |
| 第3组 |  0.780  |  0.844  |    0.755   |   0.609  |    0.837    |           0.553            |                 0.583                 |
| 第4组 |  0.525  |  0.541  |    0.519   |   0.520  |    0.527    |          10.402            |                 0.067                 |
| 第5组 |  0.863  |  0.927  |    0.838   |   0.695  |    0.919    |           0.330            |                 0.743                 |
| 第7组 |  0.909  |  0.963  |    0.888   |   0.767  |    0.957    |           0.061            |                 0.823                 |

第4组仍是明确负控；第3组形成了部分可泛化结构。第5组和第7组的 nonlearned accuracy 均显著高于 chance，且测试期间 plastic weights 已冻结，因此结果不是依靠 test-time feedback 或 test-time plastic update 临时做出来的。

### 4.4. 与人类行为学论文对齐的评估

#### 4.4.1. 评估口径

论文对齐 evaluator 使用以下设置：

eval subjects: 77（注意：第4组因初步结果被排除，使用原始的36评估，没有切到最新版评估的77；群体分析上不会出现大误差）
learning: 4 blocks × 8 learning pairs
test: 10 blocks × 28 pairs
action mode: sample
choice temperature: 1.0
cue set: shared cue set + subject-specific rank permutation
time mode: phase_rescale
tie policy: incorrect
test reward: always zero
distance input at test: zero

首先排除 overall accuracy 低于 chance 的主体；随后在 pair-level、Beta fitting、stable-error 和 inter-subject analyses 中，排除 28 个 pair 的准确率全部严格高于 0.5 的 high-accuracy subjects。

每个模型都评估两个版本：

- A（active）：无反馈测试期仍允许 plastic weights 按模型自身动力学继续更新；
- F（frozen）：learning phase 结束后冻结 plastic weights，测试阶段只读取已形成的内部状态。

#### 4.4.2. 人类受试者的关键结果

论文原始样本为 80 人，排除 3 名整体表现低于 chance 的受试者后保留 77 人。

精确报告的主要行为指标为：

- 正确全局排序：8/77，10.4%；
- 自洽但错误排序：64/77，83.1%；
- 非自洽排序：5/77，6.5%；
- 在 69 名非正确排序者中，自洽错误比例为 64/69，92.8%；
- symbolic-distance slope：0.040；
- 至少存在一个错误率不低于 80% 的 pair：63/69，91.3%；
- 至少存在一个 10 次全部错误的 pair：54/69，78.3%；
- pair-level Beta 分类：13/28 high-accuracy，15/28 bimodal，0/28 ordinary unimodal；
- 论文图中 learned accuracy 约为 0.93–0.95，nonlearned accuracy 约为 0.83–0.85，整体水平约为 0.86–0.88；
- serial-position effect 呈两端高、中间低的 U 形；
- self-consistency coefficient 报告为接近 1.00。

图中读取的 accuracy 数值只作为近似参照，精确比较优先采用论文明确报告的排序类别、distance slope、stable-error 和 Beta 分类。

#### 4.4.3. v3 各组论文对齐的准确率与距离效应

“保留 n”表示排除 overall accuracy 低于 chance 后进入后续分析的主体数。

|    条件   |   overall  |  learned  | nonlearned |  distance 1 | nonadjacent | distance slope |
|    人类   |   ~0.87    |   ~0.94   |   ~0.84    |    ~0.73    |       —     |      0.040     |
|    G3-A   |   0.757    |   0.816   |   0.733   |     0.597    |    0.810    |      0.065     |
|    G3-F   |   0.784    |   0.842   |   0.761   |     0.612    |    0.842    |      0.064     |
|    G4-A   |   0.539    |   0.549   |   0.535   |     0.526    |    0.543    |      0.011     |
|    G4-F   |   0.527    |   0.540   |   0.523   |     0.519    |    0.530    |      0.005     |
|    G5-A   |   0.626    |   0.653   |   0.615   |     0.556    |    0.650    |      0.029     |
|  **G5-F** | **0.866**  | **0.929** | **0.840** |   **0.699**  |  **0.921**  |    **0.047**   |
|    G7-A   |   0.768    |   0.801   |   0.754   |     0.622    |    0.817    |      0.051     |
|  **G7-F** | **0.885**  | **0.934** | **0.865** |   **0.716**  |  **0.941**  |    **0.042**   |

从群体平均行为看，G5-F 与人类最接近；G7-F 的 symbolic-distance slope 为 0.0416，与人类 0.040 最接近，但整体表现略高，尤其 nonlearned 和大距离 pair 更接近天花板。

G3-A/F 的距离效应过陡且总体准确率偏低。G4-A/F 接近 chance，说明其没有形成稳定的全局排序。G5-A 与 G7-A 都明显低于对应 frozen 版本，说明在论文长度的无反馈测试中，持续 plastic update 会改写甚至破坏 learning 后形成的结构。

#### 4.4.4. 全局排序、自洽性与个体差异
|   条件   | correct | self-consistent incorrect | self-inconsistent | self-consistency | circular triads | τ to true | inter-subject τ |
|   人类   |  8/77   |           64/77           |       5/77        |      ~1.000      |        ~0       |     —     |        —        |
|   G3-A   |    0    |            69             |         8         |      0.992       |      0.156      |   0.539   |     0.398       |
|   G3-F   |    0    |            65             |        12         |      0.989       |      0.221      |   0.596   |     0.496       |
|   G4-A   |    0    |             1             |        35         |      0.576       |      8.472      |   0.224   |     0.083       |
|   G4-F   |    0    |             0             |        36         |      0.417       |     11.667      |   0.214   |     0.111       |
|   G5-A   |    0    |            23             |        43         |      0.848       |      3.030      |   0.352   |     0.153       |
| **G5-F** |  **3**  |          **62**           |      **12**       |    **0.989**     |    **0.221**    | **0.789** |   **0.670**     |
|   G7-A   |    1    |            34             |        42         |      0.924       |      1.519      |   0.753   |     0.595       |
| **G7-F** | **11**  |          **55**           |      **11**       |    **0.992**     |    **0.169**    | **0.902** |   **0.807**     |

G7-F 的 correct-ranker 比例为 11/77，最接近人类的 8/77；其 distance slope 和 self-consistency 也高度接近人类。但 inter-subject τ=0.807，说明不同模型主体之间仍过度趋同，形成的排序比人类更相似。

G5-F 在准确率和自洽性之间达到较好平衡：62 名 self-consistent incorrect、12 名 self-inconsistent，整体行为比 G7-F 更具个体差异，但 correct ranker 偏少。

#### 4.4.5. 稳定错误与 Beta 分类

|   条件   |   pair-level analysis n   | ≥80% stable error | 100% stable error | high-accuracy pairs | bimodal pairs | unimodal pairs |
|   人类   |            69             |      91.3%        |      78.3%        |         13          |      15       |       0        |
|   G3-A   |            77            |       98.7%       |       96.1%       |          1          |       27      |       0        |
|   G3-F   |            77            |       97.4%       |       87.0%       |          4          |       24      |       0        |
|   G4-A   |            36            |       88.9%       |       22.2%       |          0          |       0       |       28       |
|   G4-F   |            36            |       91.7%       |       13.9%       |          0          |       0       |       28       |
|   G5-A   |            66            |       89.4%       |       24.2%       |         19          |       2       |       7        |
| **G5-F** |          **74**          |     **79.7%**     |     **43.2%**     |       **15**        |     **13**    |     **0**      |
|   G7-A   |            76            |       52.6%       |        9.2%       |         20          |       0       |       8        |
|   G7-F   |            66            |       31.8%       |        6.1%       |         28          |       0       |       0        |

这一层最重要的发现是：

Human: 13 high-accuracy + 15 bimodal
G5-F:  15 high-accuracy + 13 bimodal

G5-F 的 pair-level 分布结构是目前所有 v3 条件中最接近人类的结果。它已经复现了“部分 pair 几乎所有人都做对，另一些 pair 在主体间呈明显两极分化”的群体结构。

但 G5-F 的 100% stable-error subject 比例仍只有 43.2%，低于人类的 78.3%。这说明模型虽然能形成 pair-level 双峰，但主体内部对错误 pair 的重复选择仍没有人类那么稳定。

G3-A/F 的稳定错误比例接近甚至高于人类，但 24–27 个 pair 都呈双峰，说明任务整体过难、极化过度。G7-F 则相反：28 个 pair 全部被归为 high-accuracy，没有任何 bimodal pair，说明其更像一个高度一致、接近真实排序的强 solver。

#### 4.4.6. serial-position effect

G5-F 的 rank-position accuracy 为：0.945, 0.869, 0.833, 0.824, 0.844, 0.848, 0.857, 0.903

G7-F 为：0.930, 0.893, 0.864, 0.859, 0.865, 0.862, 0.879, 0.926

两组都表现出两端高、中间低的 U 形。G5-F 的中部低谷更明显，整体幅度更接近论文图示；G7-F 的形状正确，但中间 item 仍然偏容易，因此曲线较平。

### 4.5. 当前结论

第一，distance information 是必要的任务输入之一。
与 v2 相比，第5组和第7组在 adjacent、nonlearned 和 paper-aligned performance 上均明显提高，说明只提供关系方向不能充分复现行为学任务；signed distance 有助于网络构造 rank geometry。

第二，distance 本身并不足以诱导全局排序。
第4组 learning performance 接近满分，但 paper-aligned test 接近 chance，说明模型可以利用当前 trial 的 distance 做局部选择，却不会在没有 test objective 时自动形成可泛化的 episode-persistent ranking state。

第三，G5-F 在 overall、learned/nonlearned accuracy、serial-position effect 和 high/bimodal pair 等行为表型上均很好拟合人类，但机制上 test phase 有 reward；G7-F 机制上最对齐行为学论文范式，但 human-like bimodal pair 和 stable errors 过少。

因此，当前最准确的总结是：

> v3 已经分别在不同条件下复现了人类的群体级正确率、symbolic-distance effect、全局自洽性和 pair-level 双峰结构，但尚未在同一个严格 no-feedback 模型中同时复现这些现象。

### 4.6. 下一步方向

1. **在第5组上做“G5→G7”的test-reward退火**
   第5组最适合做退火。自然形成G5（test rewarded） → 退火 → G7（test no feedback + test CE），有望同时保留人类式行为表型和无反馈测试机制的方向。

2. **subject-specific、跨重复稳定的 relation distortion**
   对 distance gain、pair reliability、item salience 或部分 relation encoding 引入 episode 内固定偏差，使同一主体对某些 pair 持续产生相似失真，而不是每次独立随机出错。

3. **episode / test CE sweep，并做多 seed 稳定性评估** (正在进行)
   对第3组尝试加大 episode 数扫描，第5/7组尝试 3000 附近 episode sweep，尝试 test CE = 1.0、0.5、0.25、0.1 等。对于第7组，目标是保持 self-consistency 和 distance effect 的同时，降低 correct-ranker 比例和 inter-subject similarity，增加 bimodal pair 与 stable subject-specific errors。第5组类似。
