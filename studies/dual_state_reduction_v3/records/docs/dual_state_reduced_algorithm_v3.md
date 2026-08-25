# Item-history reduced algorithm v3

## Outcome

The registered outcome is `item_history_insufficient`. All integrity,
artifact, gauge, finite-value, scoped-rollout-binding, local exactness, GPU,
and bounded-thread gates pass.

The capacity-matched candidate was

\[
q_{t+1}=q_t+x_t\odot x_t,
\]

\[
s_{t+1}=\Pi\left[s_t+A x_t+B(q_t\odot x_t)\right],
\]

with the same 128 fitted parameters as scalar-history v2. It reused the frozen
v1 trajectories and did not re-extract development networks.

## Positive but insufficient links

Item history improves one-step prediction over the accumulator in every held-
out backbone, with all paired intervals below zero. MSE ratios are `0.898`,
`0.897`, and `0.899`, corresponding to a stable approximately ten-percent
improvement. The item allocation is therefore informative; it is not a null
feature.

Compared with scalar v2, item allocation removes the severe remote-field MSE
inflation. Its remote MSE is `0.250`, `0.214`, and `0.316`, close to the
accumulator and much lower than v2's `0.350`, `0.398`, and `0.494`. All-pair
remote influence correlations remain positive at `0.770--0.805`.

These positive links support a real distinction between total accumulated
evidence and where that evidence accumulated.

## Failure of task-derived history closure

No registered global link passes.

- One-step NRMSE is `0.585--0.624`, MSE ratios remain above `0.80`, and v3 is
  worse than scalar v2 in all three held-out networks.
- Median prefix cosine is only `0.888--0.891`; terminal cosine is
  `0.866--0.878`; terminal RMSE ratios are `0.674--0.890`.
- Remote magnitude falls to `0.198--0.247` of the full network, below even the
  registered lower bound and close to the original under-expression problem.

The two frozen history experiments therefore form a useful double constraint:

\[
\text{scalar history: remote amount restored, field allocation wrong},
\]

\[
\text{item history: allocation error reduced, remote amount lost}.
\]

Combining the two after seeing this pattern is forbidden. It would add an
unregistered degree of freedom chosen specifically to join complementary
failures.

Liu preservation is also worse than v1/v2. Reduced/full terminal potential
correlation falls to `0.8541` and `0.8566`; global-only nonlearned exact
accuracy falls to `0.78254`; the strict double dissociation fails because
local remote cross-talk is again more than 25 percent of intact. Only a small
subset of the nine behavior flag pairs remains identical. This cannot be
promoted as the main algorithm.

## Revised theory and next test

Close all task-derived confidence/history ledgers. Do not try a pair ledger,
scalar-plus-item mixture, count, absolute magnitude, elapsed time, block,
normalization, or nonlinear transform.

The supported reduced boundary is now:

\[
a_T\ \text{is an exact edge-addressed local state},
\]

\[
s_t\ \text{is a strong near-additive global observable},
\]

but neither `s_t` alone nor its registered task-derived history augmentations
form a closed global learning state.

The next experiment should be a read-only `P_t` latent sufficiency audit, not
another reduced candidate. Re-extract the exact frozen generic episodes and
require the reconstructed `s_t` trajectories to match the v1 NPZ before using
any `P_t` value. Within each backbone, fit a prospectively fixed low-dimensional
projection of `P_t` on training episodes only, then ask whether it predicts the
held-out residual update beyond `(s_t,x_t)` and transports remote LOO effects.
Because hidden/fast-weight coordinates are not aligned across independently
trained backbones, the first audit must test sufficiency separately within each
network and must not pretend that PCA coordinates are shared algorithmic
variables. Only a replicated dimensional requirement may justify a later
cross-network invariant-state construction.

## Provenance

- Contract: `benchmarks/dual_state_reduced_algorithm_v3.json`
- Implementation lock: `benchmarks/dual_state_reduced_algorithm_v3.lock.json`
- Result: `results/dual_state_reduced_algorithm_v3.json`, SHA-256 `44f125cf4d6357c0b385d7c9dcd116c9e686666d4597dbf17c17a2e34698d365`
