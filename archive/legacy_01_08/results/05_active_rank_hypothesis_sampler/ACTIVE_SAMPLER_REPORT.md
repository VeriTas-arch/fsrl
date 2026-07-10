# Active rank-hypothesis attractor sampler report

## 核心机制

该模型把学习后的内部状态从连续分数平均值改成一个离散的全局排序吸引子。每个虚拟被试先以个体化可靠性编码 8 条局部边，随后在 8! 个可能全局排序中进行主动重激活/重放式后验采样，并在无反馈测试中从这个已承诺的主观排序读出选择。

关键结构差异是：模型不再把不确定性保留为每个 item 的独立噪声，而是在测试前 collapse 到一个整体 ranking hypothesis。因此，同一个被试的错误天然是稳定且传递自洽的；不同被试因为可靠性和资源不同，会落入不同的吸引子。

## 单次运行结果

- seed = 42; virtual subjects = 77; repetitions = 10
- overall accuracy = 0.854
- learned / non-learned accuracy = 0.916 / 0.829
- subjects with >=80% stable error on at least one pair = 73 / 77 (0.948)
- subjects with 100% stable error on at least one pair = 53 / 77 (0.688)
- majority-choice self-consistency = 1.000; circular triads = 0.000
- inter-subject Kendall tau = 0.625
- correct / self-consistent incorrect / self-inconsistent subjects = 2 / 75 / 0
- beta pair categories = {'bimodal': 16, 'high_accuracy': 12, 'low_accuracy': 0, 'unimodal': 0, 'scipy_unavailable': 0}
- distance accuracy = 1:0.688, 2:0.812, 3:0.923, 4:0.944, 5:0.964, 6:0.977, 7:0.990

## 推荐参数

```text
sigma_distance=2.0
order_bonus=1.5
resource_log_sd=0.6
choice_beta=2.2
lapse=0.03
edge_weight_concentration=8.0
distance_salience=1.2
```

## Sweep top 5

1. score=0.211, sigma=2.0, order_bonus=2.0, resource_sd=0.6, beta=2.2, lapse=0.03, overall=0.854, learned=0.911, nonlearned=0.831, c80=0.948, c100=0.701, self=1.000, tau=0.627, bimodal_pairs=9
2. score=0.222, sigma=2.0, order_bonus=1.5, resource_sd=0.6, beta=2.2, lapse=0.03, overall=0.854, learned=0.916, nonlearned=0.829, c80=0.948, c100=0.688, self=1.000, tau=0.625, bimodal_pairs=10
3. score=0.226, sigma=2.0, order_bonus=1.5, resource_sd=0.0, beta=2.2, lapse=0.05, overall=0.851, learned=0.915, nonlearned=0.826, c80=0.935, c100=0.714, self=1.000, tau=0.644, bimodal_pairs=10
4. score=0.229, sigma=2.0, order_bonus=2.0, resource_sd=0.6, beta=2.2, lapse=0.05, overall=0.847, learned=0.902, nonlearned=0.825, c80=0.948, c100=0.675, self=1.000, tau=0.627, bimodal_pairs=9
5. score=0.243, sigma=2.0, order_bonus=1.0, resource_sd=0.6, beta=2.2, lapse=0.05, overall=0.840, learned=0.895, nonlearned=0.819, c80=0.974, c100=0.727, self=1.000, tau=0.608, bimodal_pairs=9
