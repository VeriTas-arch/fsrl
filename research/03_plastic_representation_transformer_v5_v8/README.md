# Plastic Representation Transformer: V5-V8

V5-V8 移除了 direct item-state writer。item code 在 episode 内固定，只有 global recurrent hidden、Hebbian trace 和 plastic matrix 更新；每个 item 的可读出表征由共享的 plastic transformation 生成。

这是最接近“跨 trial 关系应压缩到 recurrent plastic state”约束的系列。V5 的 test loss 反传不足，V6 的 observed-pair replay / auxiliary 没有形成方向性语义，V7 的 predict-feedback loop 仍未起势，V8 只学到局部 scaffold，all-pair generalization 仍弱。

因此此目录是关键的负向机制检验。`results/v5` 至 `results/v8` 以版本保存原始输出。
