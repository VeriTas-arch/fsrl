# Scalar-history reduced algorithm v2

## Outcome

The registered outcome is `scalar_history_insufficient`. Every source,
artifact, valid-slice, NaN-padding, gauge, finite-parameter, local-exactness,
GPU-preservation, and bounded-thread gate passes, so the result is
interpretable.

The candidate added exactly one task-faithful state:

\[
c_{t+1}=c_t+\tfrac12\lVert x_t\rVert_2^2,
\]

\[
s_{t+1}=\Pi\left[s_t+A x_t+B(c_t x_t)\right].
\]

It was fit by one closed-form ridge regression on the frozen v1 generic
trajectory artifact. No neural development trajectory was regenerated and no
Liu or human quantity entered the fit.

## Positive result: cumulative history is informative

The scalar history term improves one-step prediction robustly in every
leave-one-backbone-out fold.

| Held-out backbone | Accumulator MSE | v2 MSE | v2 / accumulator | Paired 95% interval |
|---|---:|---:|---:|---:|
| 2101 | 0.07371 | 0.06366 | 0.8637 | [-0.01118, -0.00905] |
| 2102 | 0.04648 | 0.04008 | 0.8621 | [-0.00730, -0.00537] |
| 2103 | 0.06558 | 0.05603 | 0.8543 | [-0.01066, -0.00855] |

All three fits also beat the frozen v1 rank-2 candidate. The fitted `B` norm is
stable (`0.434--0.460`) across folds. This is positive evidence that the
mapping from new evidence to potential change depends on accumulated effective
evidence history, not only on the current potential and observation.

The relation-LOO magnitude result is equally informative. v1 recovered only
`0.251--0.352` of the full disjoint effect; v2 recovers `0.781`, `0.897`, and
`0.861`. Thus the scalar state supplies the missing global gain scale that v1
diagnosed.

On untouched Liu backbones, the strict reduced P/L double-dissociation rule now
passes separately in 2104 and 2105. Global-only nonlearned exact accuracy is
`0.84453`; local-only learned accuracy is `0.70693` and `0.71761`; local-only
nonlearned accuracy remains below chance; and local remote magnitude is only
about five to six percent of intact after the scalar history restores global
remote magnitude. Exact local edge-plus-Gram reconstruction remains below
`8.89e-15`.

## Negative result: scalar history restores amount, not allocation

The improvement is insufficient under every registered global link.

- NRMSE remains `0.563--0.601`, above the `0.50` ceiling, and MSE ratios remain
  `0.854--0.864`, above the `0.80` requirement.
- Median prefix cosine remains `0.886--0.890`; terminal cosine remains
  `0.873--0.889`; terminal centered RMSE ratios remain `0.699--0.998`.
- Although remote magnitude is now in the registered range and all-pair
  influence correlation remains positive (`0.774--0.783`), remote MSE becomes
  worse than both the accumulator and v1 in all three folds.

The scalar therefore changes the correct computational dimension—history-
dependent gain—but applies that gain to the wrong pairwise pattern. It should
not be promoted as a closed confidence state or relabeled as a successful
algorithm.

The nine-row behavioral category vector also remains unpreserved. The same
core subset as v1 survives, while difficult-pair bimodality and inter-subject
diversity fail, serial-position calibration changes category, and seed 2105
self-consistency changes category. The new double-dissociation pass cannot
override this independent failure.

## Revised theory and stop rule

Preserve both sides of the result:

\[
\text{cumulative evidence history controls global update magnitude},
\]

but

\[
\text{one global history scalar cannot allocate that magnitude correctly}.
\]

Close scalar history feature engineering. Do not try cumulative count,
absolute magnitude, time, block, normalization, nonlinear scalar transforms,
or mixtures after seeing v2.

The next minimal higher-dimensional state should test item-allocated evidence
confidence without increasing fitted parameter count:

\[
q_{t+1}=q_t+x_t\odot x_t,
\]

\[
s_{t+1}=\Pi\left[s_t+A x_t+B(q_t\odot x_t)\right].
\]

`q_t` is an eight-item evidence-energy ledger derived only from the normal
effective support stream. Compared with the scalar `c_t`, it asks exactly
whether the missing history signal must retain where evidence accumulated.
The full matrix `B` can map endpoint-specific confidence into third-party
potential changes, while total fitted capacity remains 128 parameters. This
must be separately frozen and compared with the accumulator and v2. If it
fails, close task-derived confidence ledgers and move to a P-derived latent
state audit rather than inventing more coverage features.

## Provenance

- Contract: `benchmarks/dual_state_reduced_algorithm_v2.json`
- Implementation lock: `benchmarks/dual_state_reduced_algorithm_v2.lock.json`
- Frozen v1 trajectory source: `results/dual_state_reduced_algorithm_v1.trajectories.npz`
- Result: `results/dual_state_reduced_algorithm_v2.json`, SHA-256 `4326ec76995114848d46ad59e815d365af1b179fad31e563d5fed586a241fe5a`
