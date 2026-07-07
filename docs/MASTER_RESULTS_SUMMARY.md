# Master results summary

下表只列代表性结果；完整结果见 `results/master_results_table.csv`。

| stage | attempt | overall | learned | nonlearned | c80 | self | tau | bimodal | note |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 01_score_coordinate | D_best_train | 0.613 | 0.637 | 0.603 | 0.675 | 0.855 | 0.232 |  | score-coordinate / parameter sweep |
| 01_score_coordinate | sweep_G_strong_prior_beta16 | 0.642 | 0.669 | 0.631 | 0.545 | 0.908 | 0.362 |  | score-coordinate / parameter sweep |
| 02_memory_forgetting | M4_edge_hybrid | 0.683 | 0.739 | 0.661 | 0.182 | 0.918 | 0.800 |  | edge memory/forgetting/interference |
| 02_memory_forgetting | M2_strong_forgetting_beta30 | 0.554 | 0.573 | 0.547 | 0.610 | 0.800 | 0.574 |  | edge memory/forgetting/interference |
| 03_subject_reliability | exp_R4_attn_online_short | 0.699 | 0.732 | 0.685 | 0.156 | 0.922 | 0.719 |  | subject-specific attention/reliability |
| 03_subject_reliability | exp_R5_attn_lowtemp_commit_short | 0.597 | 0.581 | 0.603 | 0.649 | 0.848 | 0.384 |  | subject-specific attention/reliability |
| 04_modular_memory | exp_T1_recon_moderate | 0.686 | 0.714 | 0.675 | 0.221 | 0.913 | 0.625 |  | reconsolidation/schema encoding/replay |
| 04_modular_memory | exp_T2_recon_strong | 0.592 | 0.636 | 0.575 | 0.636 | 0.838 | 0.418 |  | reconsolidation/schema encoding/replay |
| 04_modular_memory | exp_T3_schema_encoding | 0.635 | 0.709 | 0.606 | 0.571 | 0.882 | 0.553 |  | reconsolidation/schema encoding/replay |
| 05_active_rank_sampler | active_rank_hypothesis_sampler | 0.854 | 0.916 | 0.829 | 0.948 | 1.000 | 0.625 | 16.000 | strong behavior match, but hand-designed attractor and manual salience |
| 06_meta_plastic_rnn_attractor | meta_trained_plastic_RNN_rank_attractor_120ep | 0.831 | 0.904 | 0.802 | 0.883 | 1.000 | 0.548 | 22.000 | differentiable attractor attached to plastic RNN |
| 07_modular_ablation_large3600 | distance_rnn_300 | 0.562 | 0.584 | 0.553 | 1.000 | 1.000 | 0.014 | 28.000 | larger ablation test; RNN necessity unstable |
| 07_modular_ablation_large3600 | distance_no_rnn_300 | 0.526 | 0.564 | 0.510 | 1.000 | 1.000 | 0.077 | 28.000 | larger ablation test; RNN necessity unstable |
| 07_modular_ablation_large3600 | distance_constant_reliability_300 | 0.647 | 0.826 | 0.575 | 1.000 | 1.000 | 0.369 | 25.000 | larger ablation test; RNN necessity unstable |
| 07_modular_ablation_large3600 | distance_posterior_mean_300 | 0.541 | 0.581 | 0.526 | 0.636 | 0.517 | 0.014 | 0.000 | larger ablation test; RNN necessity unstable |
| 07_modular_ablation_large3600 | raw_bars_rnn_100 | 0.480 | 0.516 | 0.466 | 1.000 | 1.000 | 0.094 | 28.000 | larger ablation test; RNN necessity unstable |
| 08_hidden_aux_targets | general_aux_rnn | 0.632 | 0.768 | 0.578 | 1.000 | 1.000 | 0.173 | 28.000 | auxiliary hidden targets; did not restore RNN necessity |
| 08_hidden_aux_targets | general_aux_no_rnn | 0.679 | 0.870 | 0.603 | 1.000 | 1.000 | 0.427 | 24.000 | auxiliary hidden targets; did not restore RNN necessity |
| 08_hidden_aux_targets | distance_aux_rnn | 0.548 | 0.625 | 0.516 | 1.000 | 1.000 | 0.061 | 28.000 | auxiliary hidden targets; did not restore RNN necessity |
| 08_hidden_aux_targets | distance_aux_no_rnn | 0.536 | 0.608 | 0.506 | 1.000 | 1.000 | 0.163 | 28.000 | auxiliary hidden targets; did not restore RNN necessity |
| 08_hidden_aux_targets | distance_aux_strong_rnn | 0.587 | 0.572 | 0.593 | 1.000 | 1.000 | 0.335 | 26.000 | auxiliary hidden targets; did not restore RNN necessity |
| 08_hidden_aux_targets | distance_aux_strong_no_rnn | 0.649 | 0.712 | 0.624 | 1.000 | 1.000 | 0.425 | 21.000 | auxiliary hidden targets; did not restore RNN necessity |
| 08_hidden_aux_targets | raw_aux_rnn | 0.531 | 0.532 | 0.531 | 1.000 | 1.000 | 0.134 | 28.000 | auxiliary hidden targets; did not restore RNN necessity |

## Reading this table

- `overall/learned/nonlearned` 衡量基本 few-shot ranking performance。
- `c80` 是至少一个 pair 上出现 >=80% stable error 的 virtual-subject 比例。
- `self` 越高，说明 majority-choice 排序越接近传递自洽。
- `tau` 越低，说明不同 virtual subjects 的主观排序越个体化；但过低且 accuracy 低时可能只是噪声。
- `bimodal` 是 pair-level beta fit 中 bimodal pair 数，active-rank 后明显升高。