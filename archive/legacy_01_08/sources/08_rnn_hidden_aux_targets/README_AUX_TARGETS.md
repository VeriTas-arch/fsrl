# RNN hidden auxiliary-target version

This package modifies `meta_plastic_rank_rnn.py` so that RNN hidden activity is trained with auxiliary targets:

- `hidden_relation_aux_head(h)`: predicts the signed relation.
- `hidden_reliability_aux_head(h)`: predicts observation reliability/confidence.
- `aux_write_calibration_loss`: calibrates the hidden-derived effective write strength.

The auxiliary targets are used during training only. Evaluation still uses RNN-derived relation and reliability and does not restore manual distance salience.

Key outputs:

- `aux_ablation_final_report.md`: expectation-vs-result report.
- `aux_ablation_final_summary.csv`: consolidated metrics table.
- `results/aux_general800_final/`: general 800s-budget evaluation outputs.
- `results/aux_runs_1800/`: default auxiliary large batch/chunked outputs.
- `results/aux_runs_strong/`: stronger auxiliary-weight diagnostic outputs.
- `results/aux_raw40/`: raw-bar sanity-check outputs.

Useful commands:

```bash
python train_chunk_noeval.py --output-dir my_run --extra-iters 20 --batch-size 16 --hidden-size 32 --item-dim 16 --subject-dim 10 --reliability-mode rnn --relation-encoding-mode rnn --observation-mode distance
python meta_plastic_rank_rnn.py --eval-only --load-checkpoint my_run/meta_plastic_rank_rnn.pt --output-dir my_eval --reliability-mode rnn --relation-encoding-mode rnn --observation-mode distance
python meta_plastic_rank_rnn.py --eval-only --load-checkpoint my_run/meta_plastic_rank_rnn.pt --ablate-rnn --output-dir my_eval_no_rnn --reliability-mode rnn --relation-encoding-mode rnn --observation-mode distance
```

Note: in this sandbox, long single-process training sometimes stalled after 20-40 iterations. Results were therefore produced with chunked continuation and checkpoint reloads.
