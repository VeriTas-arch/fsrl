# Repository Structure

`README_v3_results_updated.md` 是机制和结果的权威说明。本文件记录本次目录重组后的资产位置，避免旧路径与当前研究主线混淆。

## Mechanism-oriented research tree

| 新路径 | 原路径 / 来源 | 机制判断 |
| --- | --- | --- |
| `research/00_upstream_miconi_kay/source` | `sources/00_original_meta_training_code` | 上游参考快照，不是本项目的最终任务范式。 |
| `research/01_explicit_score_memory` | `sources/01_04_rnn_integrated_version/01_04_rnn_integrated_version.py` 及其 outputs | 显式 `score[i]`、可选 edge memory；RNN/Hebbian 不是信息瓶颈。 |
| `research/02_implicit_item_state_writer` | `implicit_plastic_rnn_ranker.py`、implicit outputs 与 ablation log | 移除了标量 score，但 direct item-state writer 仍是强捷径。 |
| `research/03_plastic_representation_transformer_v5_v8` | `behavioral_plastic_reinstatement_ranker_V5..V8*.py` 和 outputs | 只通过 episode-local recurrent/plastic state 改变共享 item representation；当前未稳定学成全局 rank-axis。 |
| `research/04_simple_neo_mutations` | `simple_neo*`、mutation/evaluator 脚本和全部 v2/v3 outputs | 当前主要路线；保留 v2 对照、v3 distance-input 实验和 paper-aligned 评估。 |

`research/04_simple_neo_mutations/results/runs_distance_v3` 保存 G3/G4/G5/G7 的配置、训练日志、freeze full-28 输出和 paper-aligned 输出。其他 `outputs_mutants_*` 目录保留各 mutation / episode / seed sweep 的原始训练结果。

## Legacy archive

`archive/legacy_01_08` 完整保存之前按 stage 编号组织的资产：

| 归档路径 | 内容 |
| --- | --- |
| `archive/legacy_01_08/sources/01_behavior_task_score_coordinate` 至 `archive/legacy_01_08/sources/04_modular_memory_mechanisms` | 显式 score、edge memory、reliability 与 modular-memory 变体。 |
| `archive/legacy_01_08/sources/05_active_rank_hypothesis_sampler` 至 `archive/legacy_01_08/sources/08_rnn_hidden_aux_targets` | 手写 active-rank、rank-attractor、ablation 和 auxiliary-target 路线。 |
| `archive/legacy_01_08/results` | 对应的 01-08 原始结果与旧主表。 |
| `docs` | 当时的阶段性汇总；历史解释可能被 v3 报告取代。 |
| `diffs`、`git_patches`、`scripts` | 旧阶段的变更记录及其 smoke 脚本。 |

早期 01-08 不应被用于定义当前模型的版本顺序或机制结论。它们保留在仓库中仅为可追溯性和复现实验记录。

## Result interpretation

所有实现和输出文件均保留原始内容；本次变化只调整目录、文件名和导航文档。读取结果时，优先使用 `README_v3_results_updated.md` 中的统一评估口径，尤其要区分 active 与 frozen test-plastic evaluation。
