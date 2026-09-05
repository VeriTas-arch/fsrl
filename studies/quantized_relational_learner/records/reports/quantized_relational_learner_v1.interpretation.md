# 四值关系编码：持续复用不是本轮胜出的解释

## 结论与证据等级

全部三组配对训练流（2114、2115、2116）及其 Exact、Persistent、Resampled
九个最终拟合均完成。三种配方的注册 outcome 都是
`partial_behavioral_reproduction`，均不具备本合同规定的 unchanged replication
资格，也没有主模型晋升。所有拟合通过 generic 能力、证据绑定与原九项定性规则；
原定量分类命中分别为每组 2/9、4/9、8/9。

这保留一个有价值的正结果：在不增加慢参数、局部读出或复杂递归电路的情况下，
有限精度输入能够实质改变个体化内部排序；重抽编码的配方在本次共同评价样本上
命中八项原定量规则。与此同时，第一次编码后永久复用并未得到本轮充分性支持。
八项命中不是完整复现，不能为了保留候选而去掉第九项。

事实权威为[冻结合同](../benchmarks/quantized_relational_learner_v1.json)、
[模型中立准入](../../../main_model_admission/records/benchmarks/main_model_admission_v1.json)、
[完整结果](../results/quantized_relational_learner_v1.json)和
[逐拟合报告](quantized_relational_learner_v1.md)。本文只解释已有估计量，不新增
评价轴、修改区间或改变注册 outcome。

## 执行与独立性

合同在 `d45183f6323eb15e3a537258e6fe4678c06e41a0` 冻结；当前数值实现见证为
`ea260edb1fcd92edc995ad07f6b64022b49456d5`，资格检查包含 817 个必要检查。
更新后的实现与输入锁在 `0ed7472` 提交，非 Liu recovery 结果在 `d13cffd`
提交。全部九个训练档案在 `0f0a281` 联合锁定并推送后，才执行任何拟合模型评价。

每个拟合均从同样的两枚慢参数初值出发，在 48,000 个 generic episodes 上进行
1,500 次联合更新；两枚参数的实际 Adam 计数均为 1,500。每组配方的基础训练
任务流和完整编码 uniform 流的哈希链相同，三个条件按预定轮换顺序执行。
没有追加训练、checkpoint 选择、Liu 校准或事后符号保护。

三枚 seed 是独立的 generic 训练流，不是三个随机初始化的大型 RNN backbone。
九个拟合共享训练前锁定的 256 个 generic episodes 与同一组 77 名模拟 Liu
被试，包括 cue、支持顺序、接入状态和配对随机数。结果因此支持训练流稳健性，
不构成三个独立人类队列或新模拟队列的复制。participant bootstrap 每个拟合
独立进行 10,000 次，不跨模型合并被试。

## 完整行为结果

下表为原采样行为估计量；范围是三组各自点估计的范围，不是跨模型置信区间。
距离效应使用原 28 个无向 pairs、采样选择和七个 distance-level 均值的等权
OLS，未换成 20 个 nonlearned pairs 的期望概率斜率。

| 原行为量 | Exact | Persistent | Resampled | 冻结人类参考 |
| --- | ---: | ---: | ---: | --- |
| learned accuracy | 0.944805 | 0.889286–0.889610 | 0.904708–0.905357 | [0.893182, 0.933766] |
| nonlearned accuracy | 0.897208 | 0.808312–0.808831 | 0.840779–0.841299 | [0.803310, 0.850196] |
| distance slope | 0.028463 | 0.053016–0.053173 | 0.045145–0.045500 | [0.034754, 0.044899] |
| serial endpoint contrast | 0.042486 | 0.080025 | 0.077984–0.078231 | [0.053738, 0.110700] |
| sampled correct-ranker proportion | 0.259740 | 0 | 0.077922 | [0.038961, 0.168831] |
| self-consistent incorrect proportion | 0.740260 | 0.987013 | 0.909091 | [0.740260, 0.909091] |
| self-inconsistent proportion | 0 | 0.012987 | 0.012987 | [0.012987, 0.129870] |
| stable-error proportion, analysis subjects | 0.842105 | 0.974026 | 0.929577 | [0.840580, 0.971429] |
| inter-subject Kendall tau | 0.637934 | 0.529343–0.530173 | 0.567864–0.570911 | [0.507184, 0.612410] |
| bimodal pair count | 18 | 21 | 17–18 | at least 15; ordinary-unimodal and low-accuracy counts both zero |

自洽/不自洽两行共同构成一个原分类器，所以表中列出十个量不等于有十项规则。
全部 77 人通过原 accuracy eligibility；排除 sampled correct rankers 后的
analysis 人数分别为 57、77、71。稳定错误和 tau 的分母使用这些 analysis
人数，其余分类不以此替换全体分母。完整 pair 分类含 not-fit 项，均保留在结果中。

Exact 此次命中双峰与稳定错误两项；旧 score-only 的 3/9 来自不同的已冻结
评价实例，不能把两次计数变化直接归因于算法退步。Persistent 命中 nonlearned、
serial position、双峰和 tau，但 learned accuracy 略低、distance slope 偏高，
correct rankers 太少，自洽错误及稳定错误偏多。

Resampled 的唯一失败行为行是 distance slope。2114、2115、2116 的精确点值
依次为 0.045144889124481、0.045238095238095、0.045499602438378；相对于上界
0.044898798480431，超出约 0.000246、0.000339、0.000601。偏差很小，但原合同
判定仍是 FAIL。不能以模型 CI 与人类区间有重叠、显示位数取整或“只差一点”
替代原来的点分类规则。另一方面，也不能由这种边界失败直接断言一个很大的
总体机制缺陷。它目前是固定共同样本上的描述性失配。

Resampled 的自洽错误比例恰好位于参考上界，不自洽比例恰好位于下界；加上
距离边界失配，更不能由 8/9 推出已经通过新队列稳定性。没有运行准入合同中
只对合格候选开放的 400 队列与三个 witness cohorts，也不以追加队列寻找 PASS。

## 内部结构不是单纯把选择变软

严格正确的确定性内部排序人数，Exact 每组是 23/77，Persistent 是 0/77，
Resampled 是 7/77；三种配方的 Liu 分数都没有 tie。采样 Hodge 正确排序人数
则分别为 20/77、0/77、6/77。两套数字具有不同估计量，不能混用，也不能把
确定性内部排序比例直接拿去命中 sampled human interval。

Persistent 的精确条件枚举覆盖每名被试最多 128 个有效代码组合。固定 cue、
接入、顺序及参数时，条件平均的严格正确内部排序概率约为 0.003524–0.004337；
所以观察到 0/77 不代表这个模型在所有编码实现上都不可能得到正确排序。
这些枚举没有新增被试，也没有用平均 policy 重新生成原行为分类。

改变慢参数前的固定参数控制同样保留。使用同一 seed 新训 Exact 的 eta/gain，
Liu overall 正确概率约为 Exact 0.91048–0.91051、Persistent 0.83592–0.83594、
Resampled 0.85536–0.85537；使用各配方自身参数后，后两者分别约为
0.83086–0.83122、0.85983–0.86024。这说明编码本身已改变输出，配方适配又产生
不同影响；它不证明 generic 训练最优，也不能把整套差异都归因于增益变化。
这里是正确概率均值，不是表中的 sampled accuracy 或 exact decision。

## 能力、绑定与可辨别性

所有拟合的 generic learned/nonlearned exact-accuracy bootstrap 下界均高于
0.5；最弱的 Persistent nonlearned 下界仍约为 0.7405。打乱证据与关系的绑定
后，Liu overall 正确概率降至 Exact 约 0.5184、Persistent 约 0.5242–0.5247、
Resampled 约 0.5284–0.5286。所有拟合的 intact-minus-shuffled 配对区间下界
均为正，generic 也全部通过。不能把 shuffled 稍高于 0.5 改写为严格 chance；
真正严格为 0.5 的是没有 admitted update 的 z-off 控制。查询无写入身份成立。

训练前 non-Liu recovery screen 的混淆矩阵为
Exact 9/9、Persistent 9/9、Resampled 8/9；512 与 1024 个 nuisance draws 下
27 个生成设置的胜出 family 全部一致。三类 diagonal recovery 的 Wilson
95% 下界分别约为 0.7009、0.7009、0.5650，超过冻结的 1/3。
解码器未读取生成时的 z、代码、w 或真实 rank；独立 likelihood 重算误差为零。
这只是有限合成网格上的可辨别性，不是人类数据支持持续编码，更不是对学到的
所有连续参数与所有任务的可辨别性保证。

Persistent 和 Resampled 单次编码分布相同，但四次呈现的联合信息预算不同。
此次结果不能被描述为“等总信息量，只改变时间相关性”。Resampled 仍可在支持
结束后通过固定 w 产生稳定错误，所以测试时错误稳定并不要求输入代码永久缓存。

## 生物合理性与下一步边界

本轮继续支持低容量误差学习器作为有竞争力的算法骨架，但没有选出最终主模型。
两种量化模型都仍只有两枚共同训练的慢参数和 15 维全局 score 状态；Persistent
另外保存已接入关系的代码和 cue 地址。Liu 最多八条关系，代码内容最多 16 bit；
generic 训练最多十条，日志最大代码内容 20 bit、地址 payload 300 bit。
这些都不包括稳定接入变量、浮点 w、容器与执行缓冲，不能冒充总记忆成本。
Resampled 不保留代码缓存，是这次状态更简洁且描述性更接近目标的一方。

已有 score-circuit 正结果只覆盖旧参数/输入。当前归一化误差规则未变，因而
保留一个条件性电路实现路线；但本次 eta/gain、有限精度教学输入、稳定接入、
固定 cue 和随机编码器仍需各自核验，不能自动继承生物实现 PASS。没有声称
随机量化、两 bit 容量、精确地址或外环 Adam 已在人类神经系统中得到确认。

按本合同 stop rule，完整矩阵到此停止。保留 Resampled 的正结果作为下一问题
的出发点，但关闭本次完整复现/晋升主张；不改变码本、精度、符号、刷新混合、
学习率、增益、温度、训练时长或添加局部 trace 来补本轮 FAIL。
失败约束的是这一固定配方，不是所有有限容量模型。

若另行授权后续问题，应先区分固定样本的边界失配与稳健的距离曲线偏差，再
决定是否需要新机制。可前瞻冻结一个不调参的独立队列不确定性审计，完整保留
原九行与 morphology，明确其是失败配方的诊断而不是合格配方的确认；不能把
事后找到的通过队列用于反转本研究 outcome。诊断前不预设需要第三种编码或
返回大规模 RNN，更不以“只差一个指标”作为任意调参理由。当前没有执行这个
后续审计，也没有授权凭本报告启动它。

## 数值与归档完整性

独立 scalar quantizer 和 float64 recurrence 重建全部九拟合及控制，最大
recurrence 误差为 1.900384e-6，低于冻结容忍值；原采样选择、分类、统计与
注册判定均重建一致。Persistent 均值/协方差恒等式最大误差为 3.331e-15。
纯数值 arrays 均以 allow_pickle=False 保存。

早先 recovery 输出触及单文件存储审查门槛，按事前的追加存储修复记录分成
两个无损 shard 后重放；原尝试保留，全部 961 个数组逐值相同，科学设置没有
变化。当前最大注册文件为完整结果 JSON，4,098,536 bytes，低于存储审查门槛。

训练使用 CUDA float32、fullgraph/default、禁用 TF32/autocast；Torch、BLAS
与编译线程各限一条。九个拟合的训练阶段约为 20.5–27.0 秒/拟合，最大记录
warmup 约 4.06 秒，当前已有编译缓存，因此不声称这是首次冷编译成本。
监测 CPU 约一个核；小模型的低 GPU 利用率不构成修改冻结执行方式的理由。

无需 ignored checkpoints 的结果复核命令：

```bash
direnv exec . python -m fsrl.infra.formal_runtime quantized-relational-learner verify-record
```

这只重建注册结果；不要重新运行 train、evaluate 或 publish 来覆盖本研究。
