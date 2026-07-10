# Implicit Item-State Writer

该路线去除了显式 scalar score 与 edge table，改为维护每个 item 的动态向量 `item_state`。`implicit_plastic_rnn_ranker.py` 中的 writer 可直接修改被观察 pair 的两个 item representation。

机制隔离结果显示，只要 item-state writer 开启，即使没有 RNN/Hebbian 也可以表现很好；关闭 writer 后各组回到接近随机。因此它替换了 score 的表示，却没有移除跨 trial 记忆捷径。

`results/` 保存 mechanism ablation log 和原始 implicit outputs。
