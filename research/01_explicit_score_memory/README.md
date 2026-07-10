# Explicit Score / Memory Baseline

`constructive_score_ranker.py` 为原 `01_04_rnn_integrated_version.py`。其 episode-local 核心状态是显式 `score[i]`，并可选用 edge memory、RNN hidden state 与 Hebbian fast weights。

结果表明该范式可以高精度解题，但显式 rank-axis 是直接捷径：RNN/Hebbian 只调节 score 更新或 choice bias，不构成必要的信息瓶颈。因此本目录是可解释的基线和历史证据，不是当前机制主线。

`results/` 保留该版本的原始输出。
