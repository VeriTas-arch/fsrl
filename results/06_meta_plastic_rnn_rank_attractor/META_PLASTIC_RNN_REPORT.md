# Meta-trained plastic RNN with differentiable rank-attractor module

## 机制

该版本把之前离散枚举采样器改造成可微模块，并接到完整的 observation-only meta-trained plastic RNN：学习阶段由带 Hebbian fast weights 的 RNN 逐个处理 pair observation，写入局部 edge memory；学习结束后，可微 active-rank posterior 对全部 8! 个全局排序假设计算 soft posterior；训练时从 soft posterior 反传，评估时每个 virtual subject 抽样/承诺到一个 ranking attractor，再进行重复无反馈选择。

## 行为学固定任务结果

- seed = 1301; meta-training episodes = 120; virtual subjects = 77; repetitions = 10
- overall accuracy = 0.831
- learned / non-learned accuracy = 0.904 / 0.802
- >=80% stable-error subjects = 68 / 77 (0.883)
- 100% stable-error subjects = 55 / 77 (0.714)
- majority-choice self-consistency = 1.000; circular triads = 0.000
- correct / self-consistent incorrect / self-inconsistent subjects = 9 / 68 / 0
- inter-subject Kendall tau = 0.548
- beta pair categories = {'bimodal': 22, 'high_accuracy': 6, 'low_accuracy': 0, 'unimodal': 0, 'scipy_unavailable': 0}
- distance accuracy = 1:0.673, 2:0.791, 3:0.897, 4:0.908, 5:0.951, 6:0.947, 7:0.947

## 学到的神经模块参数

- likelihood sigma = 2.004
- order bonus = 1.432
- global posterior precision = 0.986
- train-time choice beta = 1.967
- mean Hebbian gate = 0.491
- mean edge write strength = 0.345

## 解释

这个模型与旧版连续 score/update 模型的核心差异不是多加噪声，而是把学习后的内部表征组织成一个 subject-level global-ranking attractor。RNN 和 fast weights 负责把相同局部输入变成个体化 edge reliability；active-rank posterior 负责把局部证据压缩为全局排序假设；测试读出来自同一个已承诺排序，因此稳定错误和传递自洽可以同时出现。

## Post-training commitment/readout sweep top 5

1. score=0.068, beta=2.2, lapse=0.03, resource_sd=1.15, temp=1.6, overall=0.836, learned=0.910, nonlearned=0.806, c80=0.883, c100=0.701, self=1.000, tau=0.572, correct=8, bimodal=9
2. score=0.079, beta=2.2, lapse=0.03, resource_sd=0.85, temp=1.3, overall=0.846, learned=0.925, nonlearned=0.815, c80=0.896, c100=0.701, self=1.000, tau=0.597, correct=8, bimodal=10
3. score=0.090, beta=2.2, lapse=0.03, resource_sd=0.85, temp=1.6, overall=0.844, learned=0.922, nonlearned=0.812, c80=0.883, c100=0.701, self=1.000, tau=0.594, correct=7, bimodal=10
4. score=0.090, beta=2.2, lapse=0.05, resource_sd=1.15, temp=1.6, overall=0.826, learned=0.899, nonlearned=0.797, c80=0.883, c100=0.701, self=1.000, tau=0.553, correct=9, bimodal=10
5. score=0.100, beta=1.9, lapse=0.03, resource_sd=1.15, temp=1.9, overall=0.820, learned=0.892, nonlearned=0.791, c80=0.909, c100=0.701, self=1.000, tau=0.529, correct=7, bimodal=11
