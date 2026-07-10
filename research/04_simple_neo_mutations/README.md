# simple_neo Mutations

这是当前主要证据目录，包含从原始 `simple_neo` 出发的 v2/v3 task mutations、训练器、full-28 evaluator 和 paper-aligned evaluator。

`simple_neo_mutants_v2.py` 覆盖无 distance 输入的对照；`simple_neo_mutants_distance_input_v3.py` 引入 signed rank-distance，只在 learning pair 输入、测试时置零。`*_eval_*.py` 提供 freeze / active test-plastic 与 paper-aligned 行为评估。

结果重点：G5-F 最接近人类的 pair-level high-accuracy / bimodal 构成；G7-F 最接近人类的 distance slope 和整体自洽性，但过度趋同且偏强 solver。详细结果见仓库根目录的 `README_v3_results_updated.md`。

`results/runs_distance_v3` 保存已整理的 G3/G4/G5/G7 配置、日志和评估；其余 `outputs_mutants_*` 目录保留 v2/v3 sweep 的原始输出。
