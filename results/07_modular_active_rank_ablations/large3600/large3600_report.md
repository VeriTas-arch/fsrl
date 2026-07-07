# 3600s large-batch / larger-iteration ablation report

## Test plan and expectations

- Increase scale from smoke settings (30 iterations, batch 4, hidden 16) to larger settings. The stable chunked run used batch 16, hidden 32, item dim 16, subject dim 10; distance-input model reached 300 effective meta-training iterations, and raw-bar model reached 100 iterations.
- Expected before running: distance-input RNN reliability should improve paper-task accuracy, preserve learned >= non-learned accuracy, preserve high self-consistency and bimodal pair distributions, and show a positive symbolic-distance trend.
- Expected before running: no-RNN, no-plasticity, and constant-reliability ablations should drop clearly if the mechanism is truly RNN-dependent and reliability is coming from RNN activity.
- Expected before running: posterior-mean readout should reduce self-consistency, because stable idiosyncratic rankings require subject-level commitment.
- Expected before running: raw-bar input should remain harder than signed-distance input, but longer training should move it above chance; no-RNN should not outperform the trained raw-bar RNN.

## Execution notes

- A direct full-suite call was launched with `--timeout 3600 --big-timeout 3600`. The 120-iteration distance baseline completed, but the long shell call was killed by the environment while progressing into later commands; its completed baseline and no-RNN eval are kept under `ablation_runs_full_3600/`.
- I then used safe chunked continuation under the same code path to avoid losing checkpoints when the environment killed long foreground jobs. The attempted single 1200-iteration background run reached about 452/1200 before being killed and did not produce a final checkpoint because it had `save_every=0`, so it is treated only as a failed long-run attempt, not as a valid result.

## Main results

| variant | overall | learned | nonlearned | c80 | c100 | self-consistency | tau | bimodal pairs | entropy | edge strength |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| distance_rnn_300 | 0.562 | 0.584 | 0.553 | 1.000 | 1.000 | 1.000 | 0.014 | 28 | 10.274 | 0.051 |
| distance_no_rnn_300 | 0.526 | 0.564 | 0.510 | 1.000 | 1.000 | 1.000 | 0.077 | 28 | 6.560 | 0.252 |
| distance_no_plasticity_300 | 0.555 | 0.604 | 0.535 | 1.000 | 1.000 | 1.000 | 0.031 | 28 | 10.027 | 0.070 |
| distance_constant_reliability_300 | 0.647 | 0.826 | 0.575 | 1.000 | 1.000 | 1.000 | 0.369 | 25 | 5.873 | 0.500 |
| distance_posterior_mean_300 | 0.541 | 0.581 | 0.526 | 0.636 | 0.026 | 0.517 | 0.014 | 0 | 10.274 | 0.051 |
| raw_bars_rnn_100 | 0.480 | 0.516 | 0.466 | 1.000 | 1.000 | 1.000 | 0.094 | 28 | 7.707 | 0.175 |
| raw_bars_no_rnn_100 | 0.517 | 0.598 | 0.484 | 1.000 | 1.000 | 1.000 | 0.103 | 28 | 5.753 | 0.298 |
| full_suite_distance_rnn_120 | 0.522 | 0.486 | 0.536 | 1.000 | 1.000 | 1.000 | 0.053 | 28 | 7.160 | 0.232 |
| full_suite_distance_no_rnn_120 | 0.430 | 0.367 | 0.455 | 1.000 | 1.000 | 1.000 | 0.510 | 23 | 4.882 | 0.337 |

## Expectation vs observed

- Distance RNN baseline at 300 iterations: overall=0.562, learned=0.584, nonlearned=0.553. This is better than the completed 120-iteration full-suite baseline (0.522) but worse than the earlier smoke run reported before (~0.638).
- RNN necessity did not hold at this checkpoint: no-RNN overall=0.526, drop=0.036, far below the desired >=0.15 drop.
- Plasticity necessity also did not hold: no-plasticity overall=0.555, drop=0.008.
- Reliability-from-RNN did not win against the constant-reliability control: constant reliability overall=0.647, which is 0.085 above the RNN-reliability baseline.
- Commitment/readout expectation held strongly: posterior-mean readout reduced self-consistency from 1.000 to 0.517, increased circular triads to 9.662, and eliminated bimodal pair fits.
- Raw-bar input did not improve at 100 iterations: raw RNN overall=0.480; raw no-RNN overall=0.517; trained raw RNN is worse by 0.036.

## Interpretation

The larger runs did not produce the ideal mechanism. They show that subject-level rank commitment reliably generates self-consistent idiosyncratic rankings, but RNN-derived reliability is not yet the source of behavioral success. At the 300-iteration distance checkpoint, learned/non-learned accuracy and distance trend are present but weak; RNN ablation barely hurts; no-plasticity barely hurts; and constant reliability outperforms RNN reliability. This means the current bottom layer still lets the active-rank / commitment module explain much of the behavior without using the RNN in the intended way.

The likely failure mode is visible in the diagnostics: edge strength falls very low in the trained RNN-reliability model, while posterior entropy remains high. Constant reliability restores stronger effective edge evidence and improves accuracy. In other words, the network is learning an under-confident reliability/write gate rather than a useful reliability signal.

## Next code-level direction

- Add an explicit auxiliary loss that makes RNN hidden activity predict signed relation and reliability separately: keep relation encoding RNN-dependent, but prevent the reliability head from collapsing toward weak edges.
- Replace the current constant posterior precision/readout with a calibrated evidence-temperature learned from RNN activity, then test whether no-RNN destroys that temperature calibration.
- For raw bars, add a small sensory-difference encoder before the RNN, trained jointly, because the current raw input does not reliably recover relational sign within 100 iterations.
- Keep posterior-mean readout as an important negative control: it correctly shows that commitment, not just noisy trial choices, is required for the paper-like stable/self-consistent errors.
