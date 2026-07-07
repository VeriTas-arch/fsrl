# Proposal

## 1. 题目

基于 meta-training 可塑 RNN 的 few-shot 关系学习机制模拟尝试

## 2. 研究背景

研究来自两个相关问题的交叉。

Miconi 与 Kay 的关系学习工作说明，带有神经调制 Hebbian 可塑性的 RNN 可通过 meta-training 学会在 episode 内快速获得新序列，并表现出传递推理、symbolic distance effect、end-anchor / serial position effect 以及 list linking 等关系学习行为。论文强调，成功模型通过外层 meta-training 发现 inner-loop 学习过程，关键机制包括可塑性权重、自生成神经调制，以及对既往 item 的 recoded reinstatement。

目标行为学论文提出了下述现象：在 few-shot 条件下，所有被试接受相同的 8 条局部 pair 关系输入，但最终并不简单收敛到 ground-truth 排序，而是形成**稳定、自洽、但个体化的主观全局排序**。这要求构建出的模型不仅能解释 group-level 的 learned/non-learned accuracy、serial position effect 和 symbolic distance effect，还要解释 individual-level 的 bimodal pair errors、consistent errors、自洽错误结构、低 inter-subject similarity 等结果。

## 3. 研究问题

本 Proposal 的核心问题是：

能否从 meta-training plastic RNN 的神经机制出发，建立通过 episode 内学习自然产生个体化全局排序的模拟模型，用来解释行为学论文中的 constructive ranking findings？另外，如果能成功解释，这些行为结果主要来自哪一类底层机制？

## 4. 研究方法

### 4.1 任务与评估数据

本研究沿用论文固定行为任务做 simulation。任务结构为：

- 8 个 item 构成隐藏 ground-truth ranking；
- learning phase 给出 8 条固定 non-adjacent local pairs，每条 pair 在 4 个 learning blocks 中出现一次；
- testing phase 测试全部 28 个 possible pairs，每个 pair 重复若干次；
- testing 无 feedback；
- virtual subjects 默认对齐论文规模，常用 N=77，pair repetition=10。

评估指标包括：

- overall / learned / non-learned accuracy；
- serial position effect；
- symbolic distance effect；
- pair-level beta fitting 与 bimodal pair count；
- consistent error subjects，尤其是 >=80% 和 100% threshold；
- circular triads / self-consistency coefficient；
- HodgeRank-style reconstructed subjective rankings；
- inter-subject Kendall tau；
- ablation drop，例如 no-RNN drop、no-plasticity drop、posterior-mean readout drop。

### 4.2 模型路线

已经尝试的路线如下：

1. **原始 meta-training plastic RNN 阅读与整理**：复现和梳理 Miconi & Kay 源码结构，包括 RNN、Hebbian plasticity、neuromodulation、active/passive solution、transitive inference 和 list-linking 评估。
2. **score-coordinate constructive baseline**：将行为任务改写为统一 latent score schema，而不是 28 个 pair 的独立分类器。
3. **edge memory + forgetting/interference**：引入局部边记忆、可靠性、遗忘、容量竞争，模拟 sparse learning 下的编码损坏和检索不稳定。
4. **subject-specific attention/reliability**：让不同 virtual subjects 对 item、pair、distance 的编码可靠性不同，把个体差异前移到学习阶段。
5. **modular memory mechanisms**：加入 schema-biased reconsolidation、schema-consistent encoding、internal replay/rehearsal，并进行模块化对照。
6. **active-rank hypothesis sampler**：把学习后的状态从连续 score 改为离散 global ranking attractor，测试 subject-level commitment 对稳定、自洽错误的作用。
7. **meta-plastic RNN + differentiable rank attractor**：将 active-rank posterior 接入 observation-only meta-trained plastic RNN，由 RNN 和 fast weights 处理学习输入。
8. **modular ablations / raw observation input**：将 no-RNN、no-plasticity、constant reliability、posterior-mean readout、raw bars 等做成独立开关，测试 RNN 是否真正必要。
9. **RNN hidden auxiliary targets**：给 hidden activity 增加 signed relation 和 reliability 辅助目标，尝试让 relation/reliability 更明确地从 RNN 活动产生。

## 5. 已完成尝试与当前结果

| 阶段 | 代表尝试 | 结果摘要 | 阶段性判断 |
|---|---|---|---|
| 原始 meta-training 代码阅读 | `fsrl.zip` / original TransitiveInference code | 已整理 RNN + plasticity + neuromodulation 结构，确认其原任务是 adjacent-trial learning + reward feedback + list linking，而目标行为任务是 observation-only few-shot pairs。 | 需要改成行为论文的 fixed 8-pair observation-only 任务。 |
| score-coordinate baseline | D_best_train, G_strong_prior_beta16 | D: overall 0.613, c80 0.675, self 0.855, tau 0.232；G: overall 0.642, self 0.908。 | 统一 score schema 可产生稳定主观排序，但个体化主要来自 prior/noise，神经机制不足。 |
| edge memory + forgetting | M4 edge_hybrid, M2 strong_forgetting_beta30 | M4: overall 0.683, learned 0.739, self 0.918，但 tau 0.800；M2_beta30: c80 0.610，但 overall 0.554。 | memory 能提高 accuracy/self-consistency；强 forgetting 可产生 stable errors，但像记忆损坏。 |
| subject reliability | R4 online reliability, R5 low-temp reliability, R2 strong reliability | R4: overall 0.699, self 0.922；R5: c80 0.649, tau 0.384；R2: c80 0.623, tau 0.162。 | 个体化可以前移到学习阶段，但强可靠性差异会牺牲 accuracy 和自洽性。 |
| modular memory mechanisms | T1-T7 | T1 reconsolidation: overall 0.686, self 0.913；T2 strong reconsolidation: c80 0.636；T3 schema encoding: c80 0.571；T4 replay: c80 0.416。 | reconsolidation/schema encoding 是潜在有用方向；replay 更像稳定学习，不是个体化主来源。 |
| active-rank sampler | active_rank_hypothesis_sampler | overall 0.854, learned 0.916, nonlearned 0.829, c80 0.948, self 1.000, bimodal pairs 16。 | 行为形态非常好，说明 global ranking commitment 很关键；但该版本手工成分较强，不能作为最终神经机制。 |
| meta-plastic RNN + rank attractor | 120-episode meta-plastic RNN | overall 0.831, learned 0.904, nonlearned 0.802, c80 0.883, self 1.000, tau 0.548, bimodal pairs 22。 | 把 active-rank 接入 RNN 后仍表现较好；但需要 ablation 验证 RNN 是否真是必要来源。 |
| modular ablations + raw input | smoke / large3600 | smoke 中 no-RNN drop 明显；但 large3600 中 distance_rnn_300 overall 0.562，no-RNN 0.526，constant reliability 0.647，raw bars 仍接近 chance。 | RNN 必要性不稳定；active-rank/commitment 仍可能绕过 RNN。raw input 需要 sensory encoder。 |
| RNN hidden auxiliary targets | aux general / large1800 / raw | general full 0.632, no-RNN 0.679；large default full 0.548, no-RNN 0.536；strong aux full 0.587, no-RNN 0.649；raw full 0.531。 | 辅助目标没有解决 RNN 必要性；reliability 仍应从 write strength 改成 posterior evidence precision / likelihood temperature。 |

## 6. 预期结果

理想结果不是单纯提高 accuracy，而是同时满足以下条件：

1. group-level 上出现 learned/non-learned accuracy、serial position effect 和 symbolic distance effect；
2. individual-level 上出现稳定错误、pair-level bimodality、自洽但错误的 subjective ranking、低 inter-subject similarity；
3. no-RNN / no-plasticity / no-reliability / no-commitment ablation 产生可解释的 signature loss；
4. raw observation 或接近 raw observation 的输入条件下，模型仍能通过内部学习生成 relation evidence，而不是依赖手工 distance salience；
5. RNN hidden activity 中可以解码或解释 signed relation、confidence/reliability、以及后续 global ranking commitment 的形成过程。

## 7. 意义

初步研究建立了一套**可被 ablation 证伪的机制模拟框架**。它可以帮助区分：

- 传统 independent-value / Q-learning 类模型能解释的 group-level pattern；
- human-like constructive ranking 需要的 individual-level global schema；
- 哪些行为来自一般记忆噪声，哪些来自主动构造和主观排序 commitment；
- meta-trained plastic RNN 中哪些神经活动确实承担了 relation encoding / reliability estimation / knowledge reassembly 的作用。

如果后续能拿到 RNN 必要、且可解释行为学 findings 的机制，就可以把行为论文的 constructive ranking account 与 plastic neural network 的可解释机制连接起来。

## 8. 当前进度与下一步计划

当前进度可以概括为三点：

1. **工程上**：已经把原始 meta-training 代码、行为任务评估、多个机制版本、active-rank、modular ablation、auxiliary hidden targets 全部整理为可复现源码和结果。
2. **结果上**：active-rank / rank commitment 对稳定、自洽、个体化错误非常关键；但 RNN-derived reliability 目前还不是稳定必要瓶颈。
3. **方向上**：需要考虑使 relation evidence 和 evidence precision 必须通过 RNN hidden activity 产生的办法，同时避免 active-rank posterior 直接吃到可绕过 RNN 的显式证据。