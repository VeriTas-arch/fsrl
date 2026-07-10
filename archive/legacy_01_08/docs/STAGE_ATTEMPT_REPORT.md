# 阶段尝试汇总与未汇报结果

本文件把从“阅读 meta-training 原始代码”到“active-rank 与 auxiliary targets”的全部未汇报尝试串起来。active-rank 是最后一类尝试之一，但不是全部工作。

## 总线索

项目起点是 Miconi & Kay 的 plastic RNN meta-training 源码。该源码的原目标是让带有 Hebbian plasticity 与 self-generated neuromodulation 的 RNN 在 episode 内学习序列，并在 transitive inference / list linking 上表现出人类和动物常见行为。目标项目则来自另一篇行为学论文：相同 few-shot local pair 输入后，人类会形成稳定、自洽但个体化的 global ranking。两者之间的差异决定了后续必须把 reward-based adjacent-pair 任务改成 observation-only fixed-pair ranking 任务。

## 1. 原始 meta-training 代码整理

- 输入包：`fsrl.zip`。
- 主要文件：`main.py`, `simple.py`, `simple_neo.py`, `eval_figures.py`。
- 结论：原代码适合理解 plastic RNN、neuromodulated Hebbian learning、active/passive solutions；但任务结构与目标行为论文不同，不能直接当作复现机制。

## 2. 行为任务改写与 score-coordinate constructive baseline

- 主要文件：`sources/01_behavior_task_score_coordinate/simple_neo.py`。
- 代表结果：见 `results/01_constructive_score_coordinate/constructive_sweep_summary.json`。
- 有用发现：统一 latent score schema 比 pair classifier 更合理，能让错误保持一定自洽。
- 局限：个体化主要来自 prior/noise/temperature，仍偏手工。

## 3. Edge memory + forgetting/interference

- 结果：`results/02_memory_forgetting/memory_forgetting_experiment_summary.csv`。
- 代表阳性：M4 edge_hybrid：overall 0.683, self-consistency 0.918。
- 代表 trade-off：M2 strong forgetting 可提高 stable errors，但 overall/self-consistency 下降。
- 判断：edge memory 是必要方向，但单靠 forgetting 很难得到“人类式”constructive ranking。

## 4. Subject-specific attention/reliability

- 结果：`results/03_subject_attention_reliability/subject_attention_experiment_summary.csv`。
- 代表阳性：R4 online reliability overall 0.699, self 0.922。
- 个体化方向：R5/R2 可降低 tau、提高 c80。
- 判断：个体差异前移到学习阶段是合理方向，但强 reliability 仍会牺牲 accuracy。

## 5. Modular memory mechanisms

- 结果：`results/04_modular_memory_mechanisms/module_experiment_summary.csv`。
- T1 moderate reconsolidation 同时提高 accuracy 与 self-consistency。
- T2/T3 更容易产生 stable errors。
- replay 更像稳定学习，而不是产生个体化错误的主因。
- 判断：schema-guided encoding + reconsolidation 是后续最值得保留的心理/神经机制方向。

## 6. Active-rank hypothesis sampler

- 代码：`sources/05_active_rank_hypothesis_sampler/active_rank_hypothesis_sampler.py`。
- 结果：overall 0.854, c80 0.948, self 1.000, bimodal pairs 16。
- 价值：证明 testing 前的 global ranking commitment 可以自然产生稳定、自洽错误。
- 局限：使用离散 ranking hypothesis 枚举和手工 salience，容易被质疑为“直接捏机制”。

## 7. Meta-plastic RNN + differentiable rank-attractor

- 代码：`sources/06_meta_plastic_rnn_rank_attractor/meta_plastic_rank_rnn.py`。
- 结果：overall 0.831, c80 0.883, self 1.000, bimodal pairs 22。
- 价值：把 active-rank 思路接到 observation-only plastic RNN 上，是从手工 sampler 向神经机制过渡的重要步骤。
- 局限：ablation 之前不能说明 RNN 必要。

## 8. Modular ablation / raw observation input

- 代码：`sources/07_modular_ablation_active_rank/`。
- smoke 一度显示 no-RNN/no-plasticity 明显下降；但 3600s 大测试中 RNN 必要性不稳定。
- raw bars 输入没有自然学起来，说明原始观测到关系证据之间还缺 sensory encoder。
- 判断：active-rank 模块太强，底层仍存在 bypass RNN 的 shortcut。

## 9. RNN hidden auxiliary targets

- 代码：`sources/08_rnn_hidden_aux_targets/`。
- 新增 signed relation auxiliary head、reliability auxiliary head、write calibration loss。
- 结果：没有明显改善；no-RNN 有时仍更好。
- 结论：reliability 不应只作为 edge write strength，而应进入 posterior evidence precision / likelihood temperature。

## 当前结论

1. `commitment/readout` 是目前最稳定的阳性机制：去掉 commitment 后，自洽性和 bimodality 明显下降。
2. `RNN-derived reliability` 目前还不是稳定必要瓶颈；不能在汇报中声称已经证明 RNN 必要。
3. `schema encoding / reconsolidation` 虽然没有 active-rank 指标漂亮，但潜在解释性更自然，值得保留为后续方向。
4. 下一阶段应强调“消除 shortcut、建立机制瓶颈”，而不是“为了让 RNN 重要而调参”。
