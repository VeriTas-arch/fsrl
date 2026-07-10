# Proposal

## 1. 题目

基于 meta-trained neuromodulated plastic RNN 的 few-shot constructive ranking 机制模拟

## 2. 研究背景

Miconi 与 Kay 表明，带有神经调制 Hebbian 可塑性的 RNN 可以通过 meta-training 学会 episode 内快速关系学习，并呈现 transitive inference、symbolic-distance 和 serial-position 等行为。其关键约束是：跨 trial 的学习必须进入 episode-local recurrent plastic state，而不是依赖预先设计的外部记忆表。

目标行为学任务则要求解释另一类现象：被试从相同的少量已知 distance 的局部 pair 关系出发，会形成稳定、自洽、但个体化的全局排序。模型既要匹配 learned/nonlearned accuracy、symbolic-distance 和 serial-position effect，也要解释 pair-level bimodality、稳定错误、自洽错误排序和跨主体差异。

## 3. 研究问题

能否让 plastic RNN 在 learning phase 的稀疏关系观察后，形成可在 no-feedback test 中读取的 episode-persistent ranking state？该状态能否同时解释人类的群体级正确率与个体级、稳定且自洽的错误结构，而不是依赖显式 score、edge table 或直接可写的 item memory？

## 4. 方法

- 任务：8 个 item 构成隐藏排序；learning phase 呈现行为论文的 8 条 sparse non-adjacent pairs，test 覆盖全部 28 条 pair。
- 训练：外层以 test choice loss meta-train；test 内不输入标签。v3 的 signed rank-distance 只在 learning pair 中输入，test 时为零。
- 评估：overall、learned/nonlearned、distance slope、serial-position、循环三元组、Kendall tau、stable error、pair-level Beta 分类与 inter-subject similarity。

## 5. 路线与尝试

1. **显式 score / edge-memory 基线**：直接维护 `score[i]` 或关系表，能够解题但将 rank-axis 预先写入模型；RNN/Hebbian 不构成必要瓶颈。
2. **Implicit item-state writer**：移除标量 score，改为维护 item 的动态向量；但机制隔离显示 writer 而非 RNN/plasticity 是主要通道。
3. **V5-V8 plastic representation transformer**：移除 direct writer，固定 item code，只让 global hidden、trace 和 plastic matrix 改变共享表征。该路线机制最干净，但目前仅形成局部 scaffold，未稳定学出全局排序。
4. **`simple_neo` mutations（v2/v3）**：回到 Miconi-style plastic RNN 的有效内核，逐步改写为行为图、passive observation、no-feedback test 和 outer test loss；v3 再引入仅限 learning phase 的 signed distance。

早期组织的 score、active-rank 和 rank-attractor 尝试已归档为历史证据，不再作为当前机制主线。

## 6. 当前进展

- `simple_neo` v2 证明：固定 8-item 行为图并非根本障碍；passive learning + no-feedback test + outer test loss 可以形成可泛化结构，且 test-plastic freeze 后表现仍保持。
- v3 表明 signed distance 是有效的 learning input，但 distance 本身不足以在没有 test objective 时诱导全局排序。
- 在 paper-aligned frozen evaluation 中，**G5-F** 最接近人类的 pair-level 结构（15 个 high-accuracy、13 个 bimodal pairs），并具有较合理的准确率、自洽性与 serial-position 效应。
- **G7-F** 的 distance slope（约 0.042）、全局自洽性和 correct-ranker 比例最接近人类，但整体表现偏强、主体间排序过度趋同，缺少人类式 bimodal pair 和稳定错误。

因此，当前已分别复现人类的群体级正确率、distance effect、全局自洽性和 pair-level 双峰结构，但与行为学论文完全对齐的严格 no-feedback 条件下目前只能复现部分表型。

## 7. 下一步

1. 在 G5 到 G7 之间进行 **test-reward 退火**：逐步移除 test reward，同时保留 G5 的人类式 pair-level 双峰和稳定错误。
2. 引入 episode 内固定、跨重复稳定的 subject-specific relation distortion，例如 distance gain、pair reliability、item salience 或 relation encoding 的系统偏差，以产生可重复的个体化错误。
3. 对 G5/G7 的 episode 数、test-loss 权重、随机种子进行系统 sweep，报告 frozen 与 active 评估的差异，避免把 test-time drift 误判为学习。

完整机制说明、实验口径和数值结果见 `README_v3_results_updated.md`；当前目录与资产位置见 `README.md` 和 `STRUCTURE.md`。
