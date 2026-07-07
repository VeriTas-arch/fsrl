# Git-ready package: meta-training plastic RNN to human constructive ranking

本仓库包用于阶段性汇报，整理了从原始 meta-training 代码阅读到多轮机制尝试、active-rank、ablation、auxiliary hidden targets 的过程。

## 快速入口

- `proposal.md`：汇报用 proposal 主文档。
- `docs/STAGE_ATTEMPT_REPORT.md`：所有未汇报尝试和阶段结论。
- `results/master_results_table.csv`：跨阶段总结果表。
- `sources/`：各阶段源码完整覆盖版本。
- `results/`：各阶段小批次/大批次结果摘要、报告和 CSV/JSON。
- `diffs/`：阶段间源码 diff。
- `scripts/git_push_template.sh`：远端推送模板。

## 目录结构

```text
sources/
  00_original_meta_training_code/
  01_behavior_task_score_coordinate/
  02_memory_forgetting_edge_memory/
  03_subject_attention_reliability/
  04_modular_memory_mechanisms/
  05_active_rank_hypothesis_sampler/
  06_meta_plastic_rnn_rank_attractor/
  07_modular_ablation_active_rank/
  08_rnn_hidden_aux_targets/
results/
  01_constructive_score_coordinate/
  02_memory_forgetting/
  03_subject_attention_reliability/
  04_modular_memory_mechanisms/
  05_active_rank_hypothesis_sampler/
  06_meta_plastic_rnn_rank_attractor/
  07_modular_active_rank_ablations/
  08_rnn_hidden_aux_targets/
```

## 为什么没有直接放入所有二进制文件

为了让仓库可以直接推送远端，本包默认不纳入大体积或不适合版本管理的文件，例如 `.pt`, `.npy`, `.dat`, `.pdf`, `.ipynb`, `.png`, `.pyc`。结果判断所需的 `summary.csv`, `summary.json`, `report.md`, `train_log.csv` 均已保留。
