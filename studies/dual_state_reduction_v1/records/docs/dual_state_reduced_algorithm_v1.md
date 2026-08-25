# Dual-state reduced algorithm v1

## Outcome

The registered outcome is `rank_2_potential_transition_insufficient`.
All source, artifact, GPU-runtime, bounded-thread, finite-parameter, gauge, and
local exactness gates pass. The negative result is therefore interpretable.

The experiment used frozen generic held-out-graph trajectories from development
backbones 2101--2103. Each leave-one-backbone-out fold fit on two different
backbones and evaluated 128 different episodes from the third. Neither the fit,
the thresholds, nor any restart used Liu behavior. One final generic-only fit
was then evaluated on untouched v2.4 backbones 2104 and 2105.

## Positive result: the local state compresses exactly

The tensor trace

\[
L_T=\sum_t s_t^L k_t,
\qquad \ell_q=L_T^\mathsf{T}k_q
\]

is exactly equivalent to the edge-state algorithm

\[
a_{t+1}=a_t+s_t^L e_{r_t},
\qquad \ell_q=(K a_T)_q,
\qquad K_{qr}=k_q^\mathsf{T}k_r.
\]

The maximum absolute discrepancy is below `8.44e-15` in every development
fold and `8.89e-15` in both Liu preservation backbones. This promotes
edge-addressed persistent memory plus a fixed Gram kernel from a proposed
compression to an exact algorithmic representation of the frozen local branch.
It does not establish that `K=I`, the tensor-product key, or the two-state
architecture is unique or biological.

## Negative result: the item potential is not closed by the registered transition

The candidate was

\[
s_{t+1}=\Pi\left[s_t+A x_t+
U\left((V^\mathsf{T}s_t)\odot(W^\mathsf{T}x_t)\right)\right]
\]

with rank two. It fails all three global primary links in all three held-out
backbones.

| Held-out backbone | Candidate NRMSE | Candidate / accumulator MSE | Median prefix cosine | Terminal cosine | Terminal order agreement |
|---|---:|---:|---:|---:|---:|
| 2101 | 0.6991 | 1.0051 | 0.8904 | 0.8859 | 0.8552 |
| 2102 | 0.6592 | 1.0099 | 0.8848 | 0.8718 | 0.8449 |
| 2103 | 0.6918 | 1.0007 | 0.8889 | 0.8865 | 0.8666 |

The paired episode contrast is robustly worse than the accumulator in 2101
and 2102 and unresolved in 2103. Teacher initialization from the full neural
`s_0` does not repair terminal alignment (`0.8693--0.8840`), so the failure is
not caused by the exchangeable zero initial state.

Increasing the same feature family is not currently supported. The frozen
576-parameter unconstrained bilinear diagnostic has held-out one-step MSE
`0.07454`, `0.04758`, and `0.06628`, also worse than the corresponding
accumulators. This rules out the response “increase rank until the transition
fits” as the next registered route. It does not prove that `s_t` is non-Markov
under every nonlinear transition.

## Remote reassembly is directionally visible but too weak

The reduced relation-LOO influence field correlates with the full field at
`0.8116`, `0.8176`, and `0.8217`. Thus the item potential and evidence retain a
stable coarse fingerprint of where a removed relation matters. The recovered
mean absolute disjoint effect is only `0.2900`, `0.3515`, and `0.2514` of the
full-network magnitude, and its MSE is worse than the accumulator in every
fold. The missing closure is therefore not arbitrary output geometry; it is
especially a failure to reproduce the history-dependent magnitude of global
reassembly.

This coexists with nearly additive full-network terminal fields: mean Hodge
fractions are `0.9960`, `0.9982`, and `0.9960`. The result establishes the
intended distinction:

\[
\text{low-dimensional output geometry}
\not\Rightarrow
\text{closed low-dimensional learning state}.
\]

## Liu preservation boundary

The final generic-only candidate remains strongly related to the untouched
Liu neural potential (correlation `0.90995` and `0.91002`). Several functional
roles survive:

- global-only nonlearned exact accuracy is `0.84552` in both backbones;
- local-only learned exact accuracy is `0.70693` and `0.71761`;
- local-only nonlearned exact accuracy is `0.44836` and `0.44523`;
- global-only remote influence is positive.

The strict double-dissociation rule nevertheless fails because exact-K local
cross-talk retains `30.51%` and `32.42%` of intact remote magnitude, above the
frozen 25-percent ceiling. This preserves the direct-versus-global division of
labor qualitatively but does not satisfy the registered reduced identity gate.

The nine-row behavior category vector is not preserved. Learned accuracy,
nonlearned accuracy, symbolic-distance direction/mismatch, stable errors, and
Hodge-rank status survive in both seeds. Difficult-pair bimodality and
inter-subject diversity do not. Serial-position calibration changes category
in both seeds, and seed 2105 also changes the self-consistency category. A
numerically more human-like serial contrast is not counted as a repair because
human metrics were not fitting targets and category preservation was frozen.

## Revised theory and next decisive test

Keep the supported chain:

\[
\text{effective evidence}\rightarrow P_T\rightarrow
\text{near-additive global field},
\qquad
\text{broader local evidence}\rightarrow a_T\rightarrow K a_T.
\]

Replace only the unsupported closure claim. The current evidence supports
`s_t` as a strong global observable, not as a sufficient state for the
registered `(s_t,x_t)` bilinear transition. The under-recovered remote
amplitude and failure of both rank two and the full bilinear diagnostic point
to a missing cumulative confidence/history variable rather than another rank
choice.

The next admissible family should therefore be frozen separately as

\[
c_{t+1}=c_t+\tfrac12\lVert x_t\rVert_2^2,
\]

\[
s_{t+1}=\Pi\left[s_t+A x_t+B(c_t x_t)\right].
\]

Here `c_t` is a one-scalar, task-faithful cumulative effective-evidence energy;
it contains no Liu behavior, label, posterior, neural state, or query
information. Removing a relation changes later write gains through `c_t`, so
the model can make a prospective, falsifiable remote-reassembly prediction.
It must reuse the frozen v1 trajectory artifact and compare against both the
accumulator and the failed v1 candidate. If it fails, do not add more history
features post hoc; conclude that this scalar augmentation is insufficient and
move to a separately registered higher-dimensional state test.

## Provenance and execution repair audit

- Contract: `benchmarks/dual_state_reduced_algorithm_v1.json`
- Original implementation lock: `benchmarks/dual_state_reduced_algorithm_v1.lock.json`
- Source-semantic repairs: `benchmarks/dual_state_reduced_algorithm_v1.repair1.lock.json`, `benchmarks/dual_state_reduced_algorithm_v1.repair2.lock.json`, and `benchmarks/dual_state_reduced_algorithm_v1.repair3.lock.json`
- Raw trajectories: `results/dual_state_reduced_algorithm_v1.trajectories.npz`, SHA-256 `3bc5f874b459a490a3e66acbe1b057cee863898eeaef9fc2c8492e76d101ce34`
- Result: `results/dual_state_reduced_algorithm_v1.json`, SHA-256 `573b56ea212849f83d61e6f73d24ed9a5a8e078d6793a18e6a5da73198e871c9`

The three recorded failures occurred before any result write or metric
interpretation. They repaired exact source semantics only: zero-norm local
keys, scalar evaluator margins, and scalar behavior margins. No scientific
equation, seed, episode, fit, threshold, or outcome rule changed.
