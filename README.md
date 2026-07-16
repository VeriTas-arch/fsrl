# FSRL Transitive Inference Evaluation

This project trains and evaluates **Few-Shot Reinforcement Learning (FSRL)** models on a classic **transitive inference (TI)** task with 8 stimuli (A–H).  
The goal is to assess whether models learn ordered relations from a small set of training pairs and generalize correctly to all 28 unordered test pairs.

---

## Project Structure

| File | Purpose |
|------|---------|
| `simple_neo.py` | Trains 20 FSRL models (seeds 1–20) with the `fsrl` framework. |
| `test.py` | Evaluates a single trained model on the 28 test pairs and saves a CSV. |
| `testDS.py` | Evaluates a model across many random seeds and produces per‑pair accuracy distributions. |
| `testModels.py` | Batch‑evaluates all `.dat` model files in a directory, produces a summary CSV and distribution plots. |
| `1E1G.py` | Analyzes batch results: plots grand‑averaged accuracy for learned vs. non‑learned pairs, and the symbolic distance effect. |
| `KT.py` | Computes Kendall’s τ similarity between networks using HodgeRank‑reconstructed ranks, and visualises a similarity heatmap. |

---

## Dependencies

Install required packages:

```bash
pip install pandas numpy matplotlib seaborn scipy torch
```

Additionally, the code relies on the **`fsrl`** module (not included here).  
Make sure the `fsrl` package is installed or placed in your `PYTHONPATH`.

---

## How to Use

### 1. Training
Run training for 20 seeds (outputs are saved under `output/seed_<N>/`):

```bash
python simple_neo.py
```

### 2. Single Model Evaluation
Evaluate a specific model (e.g., `net.dat`) and save results to `outputs/`:

```bash
python test.py --model-path path/to/net.dat --seed 40
```

### 3. Multi‑Seed Evaluation (Distribution)
Evaluate a model over 100 random seeds to obtain per‑pair accuracy distributions:

```bash
python testDS.py --model-path path/to/net.dat --num-seeds 100 --output-dir figures/
```

### 4. Batch Evaluation of All Models
Evaluate every `.dat` file inside a directory (e.g., `models/`) and produce a CSV + density plots:

```bash
python testModels.py --models-dir models/ --output-csv results.csv --figures-dir figures/
```

### 5. Visualisation & Analysis
After generating the batch CSV (e.g., `batch_test_results_seed_42.csv`), run:

- **`1E1G.py`** – plots two panels: learned vs non‑learned accuracy, and symbolic distance effect.
- **`KT.py`** – computes network similarity (Kendall’s τ) from preference matrices and shows a heatmap.

---

## Key Parameters

- `--cs` : stimulus code length (default 15)
- `--hs` : hidden size of the RNN (default 200)
- `--triallen` : number of time steps per trial (default 4)
- `--stochastic` : sample actions instead of using argmax (default off)

---

## Outputs

- **Trained models**: `.dat` files (state dicts) in `output/seed_*/`.
- **Evaluation CSVs**: contain per‑pair accuracy for each model/seed.
- **Figures**: accuracy histograms, density plots, β‑fits, and similarity heatmaps.

---

## Notes

- The 8 training pairs are fixed: `A‑F, B‑C, B‑E, C‑G, D‑F, D‑G, E‑H, A‑H`.
- Test phase repeats each of the 28 pairs 10 times.
- All trial orderings are randomised per seed.

For more details, refer to the inline comments inside each script.