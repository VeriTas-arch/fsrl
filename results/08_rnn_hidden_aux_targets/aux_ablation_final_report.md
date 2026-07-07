# RNN hidden auxiliary-target ablation report

## 代码改动

在 `meta_plastic_rank_rnn.py` 中新增了三个 training-only auxiliary losses：

1. `hidden_relation_aux_head(h)`：从 RNN hidden activity 预测 signed relation。
2. `hidden_reliability_aux_head(h)`：从 RNN hidden activity 预测 observation reliability/confidence。
3. `aux_write_calibration_loss`：约束由 hidden-derived reliability head 产生的 effective write strength，避免上一版 edge strength 直接塌缩到接近 0。

这些 target 只用于训练；评估阶段仍然使用 RNN 产生的 relation encoding 和 reliability，没有重新加入 manual distance salience。

## 运行设置

- 一般测试：基于 40-iteration、batch 8 的 distance-input checkpoint，按 800s 预算评估 full / no-RNN / constant reliability / posterior-mean readout。
- 大批次测试：基于 batch 16 的 distance-input checkpoint。由于 sandbox 内长单进程会在约 20-40 iter 后随机卡住，改成 chunked continuation；默认 auxiliary 版本最终保留 175 effective iterations。
- 额外调参测试：尝试 stronger auxiliary weights、100 effective iterations，用来判断是否只是 auxiliary loss 权重偏弱。
- Raw bars：完成 20 effective iterations 的 raw-observation sanity check。

## 汇总结果

| group | variant | overall | learned | nonlearned | c80 | c100 | self | tau | bimodal | entropy | edge |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| general800 | general_aux_constant | 0.627 | 0.784 | 0.565 | 1.000 | 1.000 | 1.000 | 0.189 | 27 | 4.257 | 0.500 |
| general800 | general_aux_no_rnn | 0.679 | 0.870 | 0.603 | 1.000 | 1.000 | 1.000 | 0.427 | 24 | 5.409 | 0.329 |
| general800 | general_aux_posterior_mean | 0.622 | 0.769 | 0.563 | 0.948 | 0.610 | 0.880 | 0.173 | 14 | 5.668 | 0.307 |
| general800 | general_aux_rnn | 0.632 | 0.768 | 0.578 | 1.000 | 1.000 | 1.000 | 0.173 | 28 | 5.668 | 0.307 |
| large1800_default | distance_aux_constant_reliability | 0.583 | 0.719 | 0.528 | 1.000 | 1.000 | 1.000 | 0.267 | 26 | 4.963 | 0.500 |
| large1800_default | distance_aux_no_plasticity | 0.538 | 0.592 | 0.517 | 1.000 | 1.000 | 1.000 | 0.104 | 28 | 7.228 | 0.221 |
| large1800_default | distance_aux_no_rnn | 0.536 | 0.608 | 0.506 | 1.000 | 1.000 | 1.000 | 0.163 | 28 | 5.685 | 0.306 |
| large1800_default | distance_aux_rnn | 0.548 | 0.625 | 0.516 | 1.000 | 1.000 | 1.000 | 0.061 | 28 | 7.622 | 0.200 |
| large1800_strong | distance_aux_strong_constant | 0.594 | 0.583 | 0.598 | 1.000 | 1.000 | 1.000 | 0.593 | 22 | 3.789 | 0.500 |
| large1800_strong | distance_aux_strong_no_rnn | 0.649 | 0.712 | 0.624 | 1.000 | 1.000 | 1.000 | 0.425 | 21 | 4.947 | 0.316 |
| large1800_strong | distance_aux_strong_posterior_mean | 0.591 | 0.579 | 0.596 | 0.974 | 0.688 | 0.918 | 0.335 | 1 | 6.217 | 0.236 |
| large1800_strong | distance_aux_strong_rnn | 0.587 | 0.572 | 0.593 | 1.000 | 1.000 | 1.000 | 0.335 | 26 | 6.217 | 0.236 |
| raw_general | raw_aux_no_rnn | 0.527 | 0.525 | 0.528 | 1.000 | 1.000 | 1.000 | 0.352 | 27 | 4.723 | 0.380 |
| raw_general | raw_aux_rnn | 0.531 | 0.532 | 0.531 | 1.000 | 1.000 | 1.000 | 0.134 | 28 | 4.650 | 0.393 |

## 预期 vs 结果

- 一般测试：full overall=0.632，no-RNN=0.679，RNN ablation drop=-0.047；constant reliability 相对 full 的差值=-0.005。
- 大批次默认 auxiliary：full overall=0.548，no-RNN=0.536，drop=0.012；no-plasticity drop=0.009；constant reliability 比 full 高 0.035。
- Strong auxiliary 调参：full overall=0.587，no-RNN=0.649，RNN ablation 反而高 0.063；constant reliability 差值=0.007。
- Raw bars sanity check：raw RNN overall=0.531，raw no-RNN=0.527，drop=0.004。

## 结论

辅助目标让早期 edge write strength 从上一版极低值回到约 0.3-0.4，短程/general 测试的 learned accuracy 和 symbolic-distance trend 有所改善；但随着训练继续，模型仍会把 RNN reliability 压低，posterior entropy 升高，导致大批次版本整体 accuracy 下降。

因此这一步没有拿到理想结果：RNN 必要性仍不稳定，no-RNN ablation 的下降很小，有时甚至 no-RNN 更高；constant reliability 仍能达到或超过 RNN-derived reliability。说明仅靠 hidden auxiliary prediction 还不足以让 reliability 成为真正的神经机制瓶颈。

最有价值的阳性结果仍是 commitment/readout：posterior-mean readout 会明显降低 bimodal pair 和自洽性，说明 subject-level commitment 是稳定、自洽、个体化错误的必要模块；但 RNN-derived reliability 不是当前版本的必要瓶颈。

## 下一步建议

1. 把 reliability 从“写入强度”改成 posterior likelihood temperature / evidence precision，由 RNN hidden activity 直接调节 active-rank posterior 的 sharpness，而不是只调 edge memory 写入。
2. 增加 no-RNN 不可绕过的 bottleneck：relation evidence 不再直接进入 active-rank memory，而是必须通过 RNN hidden 的 signed-relation head 输出。
3. 对 raw bars 加 sensory-difference encoder，并用 auxiliary signed-relation loss 预训练或联合训练，否则 raw bars 仍难以稳定转成 relational evidence。
