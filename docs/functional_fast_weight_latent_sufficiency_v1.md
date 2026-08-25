# Functional fast-weight latent sufficiency v1

## Registered question

After the observable global potential `s_t` and current effective evidence
`x_t` are known, does the functional fast-weight state

\[
\widetilde P_t=\alpha\odot P_t
\]

contain a low-dimensional readout sufficient for both the next global update
and remote relation-LOO reassembly?

This was a model-only, read-only upper-bound audit. It used no Liu participant
data, human target, posterior comparator, behavior calibration, retraining, or
intervention. It did not test autonomous latent dynamics.

The registered outcome is:

> **`functional_P_linear_audit_insufficient`**

## Frozen design

The exact v1 generic episodes were recreated separately in backbones 2101,
2102, and 2103. Within each backbone, episodes 0--95 trained the estimators and
episodes 96--127 were held out. Coordinates and observations were never pooled
across networks.

The sole fast-weight representation was `vec(alpha * P_t)`. A train-only
linear regression first removed the component predictable from `s_t`. The
baseline predicted the centered update from `[s_t,x_t]`; the full functional-P
oracle added a dual-ridge readout of the residual state. Supervised rank
reductions used the frozen grid `k in {1,2,4,7}`. Seven is the maximum effective
linear rank because an eight-item centered potential has seven independent
output directions.

Natural and relation-LOO trajectories used the same fitted residualization,
full readout, and rank projection. Their teacher-forced predicted updates were
summed from the observed `s_0`, then compared with the full neural remote field.
This is a sufficiency readout, not an executable rollout.

## Integrity

All registered integrity gates pass.

- Natural evidence, all natural fields and potentials, selected LOO relations,
  and terminal LOO fields and potentials reproduce the frozen v1 artifact. All
  natural quantities are byte-equivalent; the largest LOO potential difference
  is `1.33e-15`.
- The initial functional state is exactly zero in every branch.
- Rank 7 reconstructs the full-P prediction to maximum absolute errors of
  `4.77e-6`, `5.72e-6`, and `6.68e-6`.
- All coefficients are finite. Execution used an RTX 5090 with PyTorch intra-op
  and inter-op threads fixed to one.

The first execution attempt stopped before held-out evaluation or output
creation because the float32 dual Gram lost numerical positive-definiteness.
Repair 1 retained the registered ridge equation and formed only that linear
system in GPU float64 before casting its solution back. The repair was recorded,
tested, committed, and pushed before the complete rerun.

## Full-P oracle failure

The full residual functional-P readout does not generalize. It is dramatically
worse than the `[s_t,x_t]` baseline in every independent backbone.

| backbone | baseline MSE | full-P MSE | full/base | paired 95% interval, full minus base |
|---|---:|---:|---:|---:|
| 2101 | 0.07406 | 4.20104 | 56.72 | [3.4939, 4.7731] |
| 2102 | 0.04441 | 6.28214 | 141.45 | [5.1077, 7.3808] |
| 2103 | 0.06244 | 4.48636 | 71.85 | [3.6401, 5.2810] |

Rank 1 already increases held-out error robustly in every network, and error
then rises monotonically through ranks 2, 4, and 7. Since `E_full > E_0`, the
registered information fraction

\[
\eta_k=\frac{E_0-E_k}{E_0-E_{\rm full}}
\]

is undefined rather than clipped or reinterpreted. Consequently every
`k_min` is null.

The train-only `s_t` residualization leaves `0.923`, `0.932`, and `0.948` of
raw functional-state squared variation. This only shows that most
`alpha * P_t` variation is not a linear copy of the potential. The held-out
failure shows that the remaining variation cannot be called a transferable
functional state under the registered estimator.

## Remote reassembly failure

The same failure appears more strongly in the causal LOO bridge.

| backbone | model | remote correlation | magnitude ratio | remote MSE |
|---|---|---:|---:|---:|
| 2101 | baseline | 0.146 | 0.972 | 1.042 |
| 2101 | full P | -0.406 | 27.97 | 204.86 |
| 2102 | baseline | 0.280 | 1.128 | 0.701 |
| 2102 | full P | -0.097 | 118.50 | 2496.05 |
| 2103 | baseline | 0.071 | 1.298 | 1.125 |
| 2103 | full P | -0.609 | 35.11 | 245.25 |

The baseline again separates amount from allocation: its remote magnitude is
near the full network while its pair-field correlation is weak. Adding the
full residual functional state does not repair allocation; it reverses the
correlation and grossly inflates magnitude. Low-rank truncation does not rescue
this, and no registered rank is sufficient.

## Scientific conclusion

Reject the registered hypothesis that a fixed cross-episode linear supervised
readout of residual `alpha * P_t` reveals the missing low-dimensional global
state. This is not evidence that `P_t` has no causal information, nor that no
nonlinear or trajectory-dependent descriptor could be constructed. It is
evidence that the apparent low-dimensionality of the output potential does not
transport back to a stable linear state coordinate, even within a backbone,
under a deliberately permissive full-state readout.

Together with v1--v3, the current algorithmic boundary is:

\[
a_{t+1}=a_t+s_t^L e_{r_t},\qquad \ell=Ka_T
\]

for the exactly compressed local branch, while the supported global statement
is

\[
\text{high-dimensional interacting plastic state}
\longrightarrow
\text{near-additive low-dimensional relational output}.
\]

Do not tune the ridge, normalize `P_t`, substitute raw or absolute-alpha
states, add PCA/nonlinear kernels, or choose another rank after this result.
Do not begin autonomous latent closure, because no sufficient latent readout
was established. The appropriate next model-side task is consolidation of
this asymmetric algorithmic theory, not another compression candidate.

## Provenance

- Contract: `benchmarks/functional_fast_weight_latent_sufficiency_v1.json`
- Original implementation lock:
  `benchmarks/functional_fast_weight_latent_sufficiency_v1.lock.json`
- Preoutput numerical repair and repaired lock:
  `benchmarks/functional_fast_weight_latent_sufficiency_v1.repair1.json` and
  `benchmarks/functional_fast_weight_latent_sufficiency_v1.repair1.lock.json`
- Fit artifact: `results/functional_fast_weight_latent_sufficiency_v1.fit.npz`,
  SHA-256 `5f3b7096879e674fb0b93c029c2210e95d3f45d3b2e5520c1aa260e968fc524b`
- Result: `results/functional_fast_weight_latent_sufficiency_v1.json`, SHA-256
  `e0f3bc7e23666557af53defb867616c04d95a035cb41a7e6ca60087ceaa768ff`
