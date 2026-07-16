# FSRL 模型评估工具使用说明

本目录包含三个脚本，用于评估训练好的 FSRL（Flexible Stimulus-Response Learning）模型在 8 项传递推理任务（Transitive Inference）上的表现。所有脚本遵循相同的实验协议，但评估的维度和输出形式有所不同。

*如有任何问题，请参考脚本源代码中的详细注释或联系nwy24@mails.tsinghua.edu.cn。*

## 实验协议

- **刺激**：8 个刺激项，按线性顺序索引为 A..H（对应数字 0..7）。
- **训练阶段**：使用 8 个固定配对（见下），重复 4 次，配对顺序在每次重复内随机打乱。
- **测试阶段**：所有 28 个无序配对（C(8,2)），重复 10 次，配对顺序在每次重复内随机打乱。
- **训练配对**（按顺序 A-H）：
  ```python
  [(0,5), (1,2), (1,4), (2,6), (3,5), (3,6), (4,7), (0,7)]
  ```
- **输入信号**：
  - 训练试次中，奖励槽（input 的特定维度）会被填充一个**带符号的配对距离**：`(b - a) / 7 * (random + 1)`，其中 `a,b` 为呈现的刺激索引，`b > a`。
  - 测试试次中，该槽置为 0。
- **响应**：每个试次持续 `triallen` 个时间步，在 `NUMRESPONSESTEP` 时刻收集动作（0 或 1，分别代表选择左边或右边的刺激）。正确动作定义为与刺激对的自然顺序一致（即较小的索引对应左边，较大对应右边）。
- **随机化**：每个试次中，配对的两个刺激以 50% 概率左右交换位置，动作标签相应调整。

所有脚本均兼容 `fsrl` 包中的 `RetroModulRNN` 模型，并使用 `TrainConfig` 配置。

---

## 脚本说明

### 1. `test.py` —— 单模型单次评估

**用途**：对单个训练好的模型文件，在一个固定的随机种子下评估所有测试配对的准确率，并输出到控制台和 CSV 文件。

**主要流程**：
- 使用给定种子构建训练/测试试次序列。
- 加载模型，进行推理，统计每个配对的正确次数。
- 打印每对准确率及总体准确率。
- 将结果保存至 `outputs/<模型名>_test_results.csv`。

**命令行参数**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model-path` | 路径 | `net.dat`（与脚本同目录） | 模型状态字典文件（.dat） |
| `--seed` | int | 40 | 随机种子（用于试次打乱） |
| `--cs` | int | 15 | 刺激编码长度（应与训练时一致） |
| `--hs` | int | 200 | RNN 隐藏单元数（应与训练时一致） |
| `--triallen` | int | 4 | 每个试次的时间步数 |
| `--stochastic` | flag | False | 是否从策略中采样动作（否则使用 argmax） |

**示例**：
```bash
python test.py --model-path models/my_model.dat --seed 42
```

---

### 2. `testDS.py` —— 单模型多随机种子分布

**用途**：对同一个模型文件，在多个不同的随机种子下分别评估，收集各配对的准确率分布，并绘制 28 个配对的密度直方图。

**主要流程**：
- 对指定的 `num_seeds` 个种子（从 `seed_offset` 开始递增），每个种子独立生成试次序列并评估。
- 收集每个配对在所有种子下的准确率。
- 绘制 4×7 子图的密度直方图（使用 `matplotlib`）。
- 保存详细结果 CSV 和图片。

**命令行参数**（继承 `test.py` 的参数，并增加以下）：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--num-seeds` | int | 100 | 评估的随机种子数量 |
| `--seed-offset` | int | 123 | 种子起始偏移量（实际种子 = `--seed` + `--seed-offset` + 索引） |
| `--output-dir` | 路径 | `figures/Seed Distributions` | 输出目录，存放 CSV 和 PNG 图片 |

**示例**：
```bash
python testDS.py --model-path models/my_model.dat --num-seeds 50 --output-dir ./results
```

---

### 3. `testModels.py` —— 批量模型对比

**用途**：对目录下所有 `.dat` 模型文件进行统一评估，生成一个横向表格（行为配对，列为模型文件），并绘制所有配对的准确率直方图及 Beta 分布拟合图。

**主要流程**：
- 扫描 `--models-dir` 下所有 `.dat` 文件。
- 为每个模型分配一个独立随机种子（由主种子生成）。
- 对每个模型执行一次评估（种子固定），得到各配对的准确率。
- 构建 CSV 表格，列名为模型文件名，行为配对名称。
- 绘制两张图：
  - `pair_accuracy_densities.png`：每个配对的历史直方图（横跨所有模型）。
  - `pair_accuracy_beta_fits.png`：在每个配对直方图上叠加 Beta 分布拟合曲线，并标注拟合参数 α, β。

**命令行参数**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--models-dir` | 路径 | `models` | 存放模型文件的目录 |
| `--output-csv` | 路径 | `models/results/batch_test_results_seed_<seed>.csv` | 输出 CSV 路径 |
| `--figures-dir` | 路径 | `models/results/figures_seed_<seed>` | 图片保存目录 |
| `--seed` | int | 40 | 主种子，用于生成各模型的评估种子 |
| `--cs`, `--hs`, `--triallen`, `--stochastic` | 同 `test.py` | 同上 | 模型超参数，必须与训练时一致 |

**示例**：
```bash
python testModels.py --models-dir ./trained_models --seed 2024
```

---

## 输出说明

### CSV 文件
- `test.py` 输出：每一行包含 `seed`, `pair`, `total`, `correct`, `accuracy`，最后额外一行 `pair="overall"` 表示总体准确率。
- `testDS.py` 输出：每一行包含 `seed`, `pair`, `total`, `correct`, `accuracy`，所有种子所有配对按行存储。
- `testModels.py` 输出：宽表格，首列为 `pair`，后续每列对应一个模型文件，单元格为准确率。

### 图片
- `testDS.py` 生成 `*_seed_distributions.png`：4×7 子图，每个子图为一个配对的准确率密度直方图。
- `testModels.py` 生成：
  - `pair_accuracy_densities.png`：与 `testDS.py` 类似但基于模型间分布（横跨模型）。
  - `pair_accuracy_beta_fits.png`：每个配对直方图叠加 Beta 分布拟合，并显示 α、β 参数。

---

## 依赖与环境

- Python 3.8+
- 核心库：`numpy`, `torch`, `matplotlib`, `scipy`
- 项目专用包：`fsrl`（需安装或将其所在目录加入 `PYTHONPATH`）

`fsrl` 包中至少需要以下模块：
- `fsrl.config`：`TrainConfig`, `ADDINPUT`, `DEVICE`, `NUMRESPONSESTEP`
- `fsrl.model`：`RetroModulRNN`
- `fsrl.task`：`generate_cue_data`

建议使用虚拟环境安装依赖：
```bash
pip install numpy torch matplotlib scipy
# 并将 fsrl 包所在路径添加到 PYTHONPATH
export PYTHONPATH=/path/to/fsrl:$PYTHONPATH
```

---

## 注意事项

1. **模型兼容性**：脚本中使用的 `cs`, `hs`, `triallen` 参数必须与训练时完全一致，否则加载模型会出错。
2. **随机种子控制**：每个脚本对随机种子的使用方式不同，请仔细阅读参数说明，确保可重复性。
3. **设备**：所有脚本均使用 `fsrl.config.DEVICE` 决定计算设备（默认自动检测 CUDA）。
4. **资源消耗**：`testDS.py` 和 `testModels.py` 可能运行较长时间（尤其种子数或模型数较多时），建议在后台运行。

---

## 快速开始

假设已在 `models/` 目录下存放了若干 `.dat` 模型文件，并已安装好环境：

```bash
# 评估单个模型
python test.py --model-path models/net_01.dat

# 评估单个模型在 100 个种子下的分布
python testDS.py --model-path models/net_01.dat --num-seeds 100

# 批量评估所有模型
python testModels.py --models-dir models/
```

结果将保存在默认输出目录中，可修改 `--output-csv` 和 `--output-dir` 自定义路径。

---

