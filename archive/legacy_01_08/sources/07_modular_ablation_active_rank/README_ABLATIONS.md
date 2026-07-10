# Active-rank meta-plastic RNN: modular ablations and new bottom settings

本版本围绕两个目标改了底层：

1. 把几类 ablation 做成互相独立的开关，便于后续每改一次底层就快速复测。
2. 去掉旧版 `raw_strength = head + 0.9 * abs(distance) + ...` 这一类手工 distance salience，让 reliability 默认从 RNN hidden activity 产生。
3. 增加 raw observation input：学习阶段可输入两根原始 bar height，而不是直接给 signed distance channel。

## 关键代码

- `meta_plastic_rank_rnn.py`：主模型、训练、评估、CLI。
- `ablation_suite.py`：自动训练一个 baseline，并对同一 checkpoint 跑独立 ablations，再训练 raw-bar variant。
- `ablation_runs_smoke/ablation_expectation_report.md`：本轮 smoke 结果和“预期 vs 实际”。

## 新增独立模块

### Observation module

```bash
--observation-mode distance      # signed relation observation，较容易
--observation-mode raw_bars      # 两根原始 bar height，较难
```

### Reliability module

默认：

```bash
--reliability-mode rnn
```

这时写入强度只从 `reliability_head(h)` 产生，不再加入手工 `abs(distance)` salience。

可选 ablation / 对照：

```bash
--reliability-mode constant
--reliability-mode manual_distance
--reliability-mode oracle_distance
--reliability-mode feature_rnn
```

### Relation encoding bottleneck

默认：

```bash
--relation-encoding-mode rnn
```

这会让 relation memory 的 signed relation 由 RNN hidden-dependent head 产生，避免 active-rank posterior 直接拿到精确 signed distance 后在 no-RNN 条件下仍然表现很好。

保留旧式对照：

```bash
--relation-encoding-mode residual_observation
```

### Independent ablations

```bash
--ablate-rnn
--ablate-plasticity
--ablate-subject-latent
--ablate-item-vectors
--ablate-resource
```

### Readout / commitment ablation

```bash
--rank-readout commit           # 每个 virtual subject commit 到一个 rank attractor
--rank-readout top1_commit      # 取 posterior top-1
--rank-readout posterior_mean   # 不 commit，用 posterior marginal 直接答题
```

## 快速 smoke 测试

```bash
python ablation_suite.py --smoke --timeout 600 --big-timeout 1200 --output-dir ablation_runs_smoke_new
```

## 单独运行示例

训练 RNN-derived reliability + RNN-only encoding baseline：

```bash
python meta_plastic_rank_rnn.py \
  --nbiter 30 --batch-size 4 --hidden-size 16 --item-dim 8 --subject-dim 6 \
  --eval-subjects 24 --eval-repetitions 5 \
  --reliability-mode rnn --relation-encoding-mode rnn --observation-mode distance \
  --output-dir smoke_rnn_distance
```

在同一 checkpoint 上跑 no-RNN ablation：

```bash
python meta_plastic_rank_rnn.py \
  --load-checkpoint smoke_rnn_distance/meta_plastic_rank_rnn.pt --eval-only \
  --batch-size 4 --hidden-size 16 --item-dim 8 --subject-dim 6 \
  --eval-subjects 24 --eval-repetitions 5 \
  --reliability-mode rnn --relation-encoding-mode rnn --observation-mode distance \
  --ablate-rnn \
  --output-dir smoke_rnn_distance_nornn
```

训练 raw observation input：

```bash
python meta_plastic_rank_rnn.py \
  --nbiter 30 --batch-size 4 --hidden-size 16 --item-dim 8 --subject-dim 6 \
  --eval-subjects 24 --eval-repetitions 5 \
  --reliability-mode rnn --relation-encoding-mode rnn --observation-mode raw_bars \
  --output-dir smoke_raw_rnn
```

## 本轮结果摘要

详见：

```text
ablation_runs_smoke/ablation_expectation_report.md
ablation_runs_smoke/ablation_summary.csv
```

核心结论：

- RNN-only relation encoding 后，no-RNN 不再维持旧版 active-rank 的高表现：distance baseline overall=0.638，no-RNN overall=0.506。
- no-plasticity 和 constant-reliability 也明显下降，说明当前 bottleneck 已经让 RNN / fast weights / reliability 更接近“必要模块”。
- posterior-mean readout 明显降低 self-consistency，支持 subject-level commitment 是稳定自洽错误的关键读出机制。
- raw bar input 的短训练结果还不理想，overall≈0.49；后续需要更长训练或增加可学习 sensory-difference encoder。

## timeout 与单位提醒

- `ablation_suite.py` 中 timeout 单位是秒，默认 smoke 用 600s，大测试用 1200s。
- 若后续接入实验呈现代码或仿真环境的 timing 参数，很多库默认单位可能是 ms；需要把秒级预期乘以 1000 后再传入。
