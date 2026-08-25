# Support factor-swap diagnostic v1

## Status and provenance

This registered read-only diagnostic is complete for frozen pilot seeds 1901
and 1902. The specification was committed and pushed as `97e7b7f` before the
implementation was written or run. No training, checkpoint selection,
architecture change, formal-seed access, posterior refit, or post-result
estimand was introduced.

The complete subject-level result is
`results/support_factor_swap_v1.json` (SHA-256
`14344c6561859206c494af831d1b9f68d80121913172abab843267c0228941a9`).
All neural replays used CUDA. Bootstrap summaries used CPU NumPy.

The preceding diagnostic established eligibility norm dominance, a much
smaller DA component, alpha-dependent functional amplification, and
fourth-exposure history dependence. This experiment asks whether those roles
are causally transferable rather than merely correlated.

## Registered factorization and interventions

For matched retained-evidence and zero-magnitude branches at effective support
steps 2 and 3, define

\[
\bar d=\tfrac12(d^++d^0),\quad
\delta d=d^+-d^0,
\]

\[
\bar E=\tfrac12(E^++E^0),\quad
\delta E=E^+-E^0.
\]

A DA factor `D_a=(dbar_a, delta d_a)` and eligibility factor
`E_b=(Ebar_b, delta E_b)` compose as

\[
C(D_a,E_b)=
\sum_{k=2}^{3}
\left[\delta d_{a,k}\bar E_{b,k}
+\bar d_{a,k}\delta E_{b,k}\right].
\]

`C(D_a,E_a)` exactly reconstructs the matched pre-clip write difference. The
four interventions were:

1. Swap norm-matched `delta E` between cyclically paired retained relations in
   the same subject and exposure while keeping recipient DA, baseline, alpha,
   and effective-write norm fixed.
2. Swap the naturally highest versus lowest same-exposure DA factor onto the
   same recipient eligibility without norm matching.
3. At exposure 4, extract natural-history factors `(D_N,E_N)` and
   no-prior-relation factors `(D_H,E_H)`, then replay all four combinations at
   the same natural zero-branch baseline.
4. Compare the actual alpha-weighted write with 32 fixed permuted-alpha
   directions, each norm matched to the actual effective increment.

The primary targets are natural neural potential updates and frozen-policy
sensitivity. Exact posterior innovation is intentionally not a trial-level
target in this diagnostic.

## Results

### Eligibility transfers relation identity

| Seed | Synthetic-to-donor cosine | Synthetic-to-recipient cosine | Donor identity advantage, 95% interval |
| --- | ---: | ---: | ---: |
| 1901 | 0.99885 | 0.14924 | 0.84961 [0.81451, 0.88477] |
| 1902 | 0.99929 | 0.14357 | 0.85572 [0.81695, 0.89440] |

Donor and recipient natural-update targets were clearly distinguishable:
`1-cosine=0.85690/0.86065`. Effective-increment norm mismatch was below
`1.5e-9`. The donor identity advantage remained `0.842--0.857` in seed 1901
and `0.847--0.861` in seed 1902 across all four exposures.

Thus `delta E` is not merely the larger term in a norm decomposition. After
moving it to another relation's state and DA factor, it causally transfers the
downstream relation-specific potential direction almost exactly.

### DA transfers magnitude while preserving direction

| Seed | High-minus-low effective-write norm | High-minus-low policy norm | High/low direction cosine |
| --- | ---: | ---: | ---: |
| 1901 | 0.13224 [0.12346, 0.14145] | 0.25964 [0.24105, 0.27934] | 0.99896 |
| 1902 | 0.10384 [0.09707, 0.11101] | 0.24268 [0.22474, 0.26176] | 0.99895 |

The policy-magnitude contrast was positive at every exposure and grew from
`0.071/0.095` on exposure 1 to `0.423/0.411` on exposure 4. Direction cosine
remained above `0.995` even at exposure 4.

The write-norm contrast partly follows algebraically from multiplying by a
different scalar factor, but the downstream policy result does not. It shows
that natural DA variation is expressed as a transferable gain change while
leaving eligibility-defined relation identity essentially intact.

### Alpha systematically places writes in high-gain directions

| Seed | Actual local gain | Mean 32-null gain | Actual-minus-null, 95% interval | Null directions beaten |
| --- | ---: | ---: | ---: | ---: |
| 1901 | 2.09807 | 0.64422 | 1.45385 [1.43438, 1.47294] | 100% |
| 1902 | 2.37954 | 0.79744 | 1.58210 [1.56255, 1.60187] | 100% |

Every retained actual write beat every registered permuted-alpha null in both
seeds after effective-increment norm matching. The advantage was stable across
exposures. This upgrades the previous one-shuffle observation: alpha
systematically places naturally generated writes in recurrent directions with
high frozen-policy sensitivity.

This remains a gain-placement result. It does not revive the rejected claim
that alpha uniquely maps writes toward exact posterior innovations.

### History changes eligibility, but factor generation alone does not explain the policy restoration

At exposure 4, replacing natural factors with no-prior-relation factors
increased the effective-write norm at the common natural baseline:

| Outcome | Seed 1901 | Seed 1902 |
| --- | ---: | ---: |
| Total write restoration `HH-NN` | 0.01617 [0.00572, 0.02680] | 0.01386 [0.00632, 0.02172] |
| Eligibility main effect | 0.02225 [0.01473, 0.02992] | 0.01363 [0.00868, 0.01874] |
| DA main effect | -0.00608 [-0.01031, -0.00185] | 0.00023 [-0.00332, 0.00396] |
| Interaction | 0.00214 [-0.00037, 0.00475] | -0.00279 [-0.00475, -0.00087] |

The write-level restoration is therefore generated primarily through the
eligibility factor. DA weakly suppresses it in seed 1901 and is unresolved in
seed 1902; the interaction is unresolved or suppressive.

The downstream policy factorial has the same component signs:

| Outcome | Seed 1901 | Seed 1902 |
| --- | ---: | ---: |
| Eligibility main effect | 0.03233 [0.01831, 0.04655] | 0.01805 [0.00537, 0.03117] |
| DA main effect | -0.01427 [-0.02250, -0.00594] | 0.00063 [-0.00875, 0.01069] |
| Interaction | 0.00015 [-0.00502, 0.00538] | -0.01267 [-0.01836, -0.00719] |
| Total common-state restoration `HH-NN` | 0.01806 [-0.00194, 0.03799] | 0.01867 [-0.00136, 0.03996] |

The registered policy-attribution competence gate fails because both total
intervals include zero. This matters. The preceding diagnostic found robust
policy restoration when natural and no-prior factors were each evaluated at
their own accumulated baseline (`0.05718/0.04893` at exposure 4). Replaying
both factor sets at one natural baseline retains a positive eligibility effect
but no longer yields a resolved total policy effect.

The history dependence therefore has at least two loci:

- accumulated relation history changes the eligibility factor that is
  generated by the next observation;
- accumulated `P` also changes how a given effective write is expressed by the
  recurrent query dynamics.

It is not sufficient to describe assimilation as only a smaller scalar DA or
only a changed eligibility write independent of the recipient state.

All implementation gates passed. Factor composition residuals were at most
`4.77e-7`; batched versus unbatched readout error was at most `3.58e-7`; actual,
history-NN, and common replays reproduced natural fields within `1.67e-6`;
final support endpoints were exact.

## Revised mechanism: state-dependent iterative relaxation

The combined mechanism is now

\[
E_t=\mathcal E_\theta(e_t,P_{t-1}),\qquad
d_t=g_\theta(e_t,P_{t-1}),
\]

\[
P_t=P_{t-1}+d_tE_t,\qquad
W_t^{\mathrm{eff}}=W+\alpha\odot P_t,
\]

\[
\Delta s_t=
\mathcal R_\theta(P_t)-\mathcal R_\theta(P_{t-1}).
\]

Eligibility supplies a transferable relational direction. DA scales its
downstream magnitude. Alpha places the write into a high-sensitivity recurrent
subspace. Accumulated fast weights alter both the next eligibility trace and
the local response of the readout map `R_theta`.

This is not sequential approximate Bayes. The learned operator performs
state-dependent, sometimes locally counter-aligned corrections that interact
over the episode. Its terminal fixed point is posterior-like even though its
individual steps are not exact posterior innovations:

\[
s_T\approx\mathcal Q(D_s)
\approx-\mathbb E[\operatorname{rank}\mid D_s].
\]

Stable subject-specific evidence omissions select different constraint sets;
the shared learned relaxation process then turns them into different coherent
global potentials.

## Claim boundary and next gate

- These are two mechanism-discovery seeds, not prospective confirmation.
- Transfer of a modeled factor establishes a computational role, not a
  biological identity.
- The alpha result concerns local policy sensitivity, not posterior direction.
- The history eligibility main effect is positive, but the registered common-
  state total policy gate is unresolved. Full history attribution is not yet
  complete.

The mechanism-pilot stop rule is therefore not fully satisfied. Do not access
formal seeds or alter the architecture yet. The remaining black box is narrow:
cross the recipient baseline (`B_N` versus `B_H`) with the generated total
factor (`C_NN` versus `C_HH`) at exposure 4. A registered 2-by-2 replay can
separate baseline sensitivity, factor generation, and their interaction. If
that final attribution is stable in both pilot seeds, freeze an independent
`mechanism_confirmation_v1` before any access to seeds 2001--2010.

## Reproduction

```bash
direnv exec . python -m fsrl.support_factor_swap
direnv exec . python -m unittest tests.test_support_factor_swap -v
direnv exec . python -m json.tool results/support_factor_swap_v1.json >/dev/null
```
