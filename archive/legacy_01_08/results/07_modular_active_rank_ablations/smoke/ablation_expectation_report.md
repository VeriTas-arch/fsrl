# Smoke ablation expectation report

## 预期（测试前写入）

- RNN-derived reliability + RNN-only relation encoding 应消除旧版 active-rank 在去掉 RNN 后仍可依靠精确距离输入解题的 shortcut。
- no-RNN、no-plasticity、constant-reliability 应降低 overall accuracy，并暴露 RNN/fast-weight/reliability 对当前底层的必要性。
- posterior-mean readout 应破坏 subject-level commitment，从而降低自洽性、增加 circular triads。
- raw bar observation 比 signed distance observation 更难，短训练下可能还达不到行为学论文的整体 accuracy，需要后续更长训练或调底层。

## 结果对照

- distance/RNN baseline overall=0.638, learned/nonlearned=0.728/0.602, self-consistency=1.000。形状上 learned > nonlearned、自洽性高；但整体 accuracy 仍低于人类锚点，说明还没有拿到理想结果。
- no-RNN ablation overall=0.506，相对 baseline drop=0.132；符合 RNN/底层模块必要性的预期。
- no-plasticity ablation overall=0.505，相对 baseline drop=0.133；符合 RNN/底层模块必要性的预期。
- constant reliability ablation overall=0.475，相对 baseline drop=0.162；符合 RNN/底层模块必要性的预期。
- posterior-mean readout self-consistency=0.835, circular triads=3.292；符合 commitment 去除后自洽性下降的预期。
- raw observation baseline overall=0.492，raw no-RNN overall=0.532；短训练下 raw 输入没有形成理想机制，后续应优先做更长训练或增加可学习 sensory-difference encoder。

## 汇总表

| variant | overall | learned | nonlearned | c80 | self | tau | circular | edge_recon |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| distance_rnn_reliability | 0.638 | 0.728 | 0.602 | 1.000 | 1.000 | 0.296 | 0.000 | 12.027 |
| distance_no_rnn_ablation | 0.506 | 0.484 | 0.515 | 1.000 | 1.000 | 0.237 | 0.000 | 15.697 |
| distance_no_plasticity_ablation | 0.505 | 0.465 | 0.521 | 1.000 | 0.992 | 0.204 | 0.167 | 16.121 |
| distance_constant_reliability_ablation | 0.475 | 0.400 | 0.505 | 1.000 | 1.000 | 0.260 | 0.000 | 16.375 |
| distance_posterior_mean_readout | 0.472 | 0.377 | 0.510 | 1.000 | 0.835 | 0.170 | 3.292 | 16.457 |
| raw_observation_rnn_reliability | 0.492 | 0.545 | 0.470 | 1.000 | 0.987 | 0.487 | 0.250 | 14.714 |
| raw_observation_no_rnn_ablation | 0.532 | 0.558 | 0.521 | 1.000 | 1.000 | 0.282 | 0.000 | 14.818 |

## 大测试记录

尝试过 120-iteration / batch 8 / hidden 32 的 RNN-only relation encoding 大测试，按 1200s timeout 运行，训练到约第 90 iteration 时超时。因此本包保留 smoke 级可复现实验结果；后续大测试建议开启 `--save-every 30` 并分块续训。
