# Operator-amplitude path v1

## Status and question

This final read-only local-fidelity gate is complete on frozen development
seeds 1901 and 1902. The protocol, including all 21 amplitude points and every
outcome threshold, was committed and pushed as `a156b44` before any path
estimand was executed. The implementation was committed and pushed as
`492f763` before the successful runs.

No network was trained or changed, no adaptive amplitude point or root search
was used, no probe was fit, and no formal seed was accessed. The question was
not whether `H>A` crosses zero—the prior positive derivative and negative exact
response already guarantee an aggregate crossing by continuity—but where the
crossing lies, how stable it is across subjects, whether other relations share
it, and whether the analytic curvature coefficient is output-opposed.

The result is replicated but heterogeneous:

> `H>A` is the only relation whose mean correctness crosses zero in both
> networks. Its mean crossing lies near 0.6 operator amplitude, but subject
> brackets are broad. The quadratic curvature coefficient is strongly
> correctness-opposed and much more negative for `H>A` than for the other
> relations. Because the failure is relation- and subject-conditioned rather
> than a shared global threshold, v2 should use an online endogenous
> expression/amplitude control, not global clipping.

This completes read-only localization on seeds 1901 and 1902. The next stage is
a frozen v2 pilot on one to three new development seeds.

## Frozen amplitude path

For relation-LOO baseline `b` and bound operator action `A`, the exact path is

\[
H_q(\lambda)
=
\tanh(b_q+\lambda A_q)-\tanh(b_q),
\]

on the prospectively fixed grid

\[
\lambda\in\{0,0.05,0.10,\ldots,0.95,1\}.
\]

Every amplitude uses the same trained action-margin covector, orientation
contrast, 28-edge complete-graph Hodge residual, direct correctness sign, and
15-edge remote control as `operator_output_semantics_v1`. There is no adaptive
refinement or interpolated root.

For each relation, the mean sign-change bracket is the first adjacent pair
whose participant mean changes from positive to nonpositive. A robust
transition interval spans the last bootstrap-robust-positive grid point before
the first later bootstrap-robust-negative point.

Each retained subject also contributes its first fixed-grid
positive-to-nonpositive bracket. Subjects are never filtered on their initial
sign or whether they cross.

## Curvature diagnostic

The exact Taylor expansion is

\[
H(\lambda)
=
\lambda J_bA
+
\lambda^2K_2
+
O(\lambda^3),
\]

where

\[
K_2
=
-\tanh(b)\odot[1-\tanh^2(b)]\odot A^2.
\]

`K2` is passed through the same fixed output, orientation contrast, and Hodge
residual. It identifies the first output-projected curvature component; it is
not labeled generic saturation because heterogeneous componentwise curvature
can rotate or reverse a semantic value rather than only attenuate it.

## Integrity and reproducibility

- Both seeds, all 77 subjects, eight relations, 28 query edges, two
  orientations, and all 21 amplitudes are retained.
- Every registered source and pilot artifact matches its frozen SHA-256.
- `lambda=0` fields and residuals are exactly zero.
- Every stable-omitted path field, residual, Jacobian, and curvature field is
  exactly zero.
- `lambda=1` reproduces actual intact-minus-LOO hidden effects within
  `5.588e-7` and logit influence within `2.489e-6`.
- `lambda=1` exact `H` and analytic `J_b A` reproduce every selected summary
  from the prior locked result exactly; maximum discrepancy is zero.
- Neural replay ran on the RTX 5090 with PyTorch intra-op and inter-op threads
  fixed to one.
- Two complete runs under the final implementation produced the same result
  SHA-256:
  `b88d39b062a2bb9d0ada2b0a4299caa5d591a8fab28233d6ecef37bd13e78302`.

## `H>A` aggregate crossing

| Estimand | Seed 1901 | Seed 1902 |
| --- | ---: | ---: |
| Mean sign-change bracket | `0.55–0.60` | `0.60–0.65` |
| Robust positive-to-negative interval | `0.45–0.65` | `0.50–0.75` |
| Peak mean correctness | `0.04587` at `0.30` | `0.04533` at `0.35` |
| Exact correctness at `1.0` | `-0.28899` | `-0.18363` |

Selected path values show a smooth, non-monotonic trajectory:

| Lambda | Seed 1901 | Seed 1902 |
| ---: | ---: | ---: |
| `0.05` | `0.01334 [0.01154, 0.01516]` | `0.01161 [0.00972, 0.01358]` |
| `0.25` | `0.04490 [0.03515, 0.05487]` | `0.04223 [0.03234, 0.05242]` |
| `0.50` | `0.01643 [-0.00514, 0.03840]` | `0.03162 [0.01094, 0.05267]` |
| `0.60` | `-0.01921 [-0.04597, 0.00820]` | `0.00937 [-0.01588, 0.03496]` |
| `0.65` | `-0.04221 [-0.07197, -0.01195]` | `-0.00577 [-0.03348, 0.02228]` |
| `0.75` | `-0.09811 [-0.13396, -0.06173]` | `-0.04402 [-0.07665, -0.01047]` |
| `1.00` | `-0.28899 [-0.34202, -0.23556]` | `-0.18363 [-0.22971, -0.13686]` |

The frozen categorical boundary labels seed 1901 `intermediate` and seed 1902
`late` because the latter bracket begins exactly at 0.60. That categorical
difference should not be exaggerated: the mean brackets are adjacent and the
robust intervals overlap substantially. The continuous result is a replicated
mid-to-late group crossing centered near 0.6.

## Subject heterogeneity

| `H>A` subject result | Seed 1901 | Seed 1902 |
| --- | ---: | ---: |
| Retained subjects | `55` | `55` |
| Positive-to-nonpositive crossing | `49/55 = 0.891 [0.803, 0.965]` | `45/55 = 0.818 [0.709, 0.912]` |
| Median crossing bracket | `0.45–0.50` | `0.55–0.60` |
| Lower-bracket IQR | `0.30–0.65` | `0.30–0.75` |
| Nonpositive at `0.05` | `0/55` | `1/55` |
| Negative at `1.0` | `49/55` | `46/55` |
| Re-entry after first crossing | `0/55` | `0/55` |

Most subjects therefore follow the same positive-then-negative topology, but
their usable amplitude range is not governed by one stable threshold. The
wide bracket distribution is a direct constraint on v2: a fixed scalar chosen
from the group mean would leave early-crossing subjects corrupted or
unnecessarily suppress late/non-crossing subjects.

## Relation specificity

`H>A` is the only mean curve that crosses zero in either seed. The other seven
relations remain mean-positive through `lambda=1`, so a global amplitude
reduction is not the justified final mechanism.

Subject trajectories reveal a secondary heterogeneous constraint. `F>A`
crosses in `48.1%` and `35.2%` of retained subjects even though its relation
mean does not cross; its median crossing is later (`0.85–0.90` and
`0.70–0.75`). Other relations have crossing proportions between zero and about
12%. Thus the failure is not a hardcoded `H>A` label, but it is strongly
concentrated in particular relation-by-subject operating states.

## Curvature result

| Quadratic direct correctness | Seed 1901 | Seed 1902 |
| --- | ---: | ---: |
| Aggregate `K2` | `-0.06375 [-0.07601, -0.05207]` | `-0.04635 [-0.05572, -0.03721]` |
| `H>A K2` | `-0.29445 [-0.34864, -0.24156]` | `-0.22553 [-0.26761, -0.18241]` |
| `H>A` minus other relations | `-0.25103 [-0.30253, -0.20082]` | `-0.19462 [-0.23406, -0.15463]` |

The first nonlinear correction is already output-opposed and is selectively
strong for `H>A`. This concretizes the prior phrase “finite-amplitude semantic
distortion”: the current baseline and bound operator action generate a
relation-conditioned curvature component pointing against the correct local
value.

The quadratic expansion is accurate near zero: aggregate absolute direct
error is only `1.54e-5/1.04e-5` at `lambda=0.05` and
`1.22e-4/8.24e-5` at `0.10`. At `lambda=1`, error grows to
`0.0840/0.0622`, so higher-order curvature also matters for the actual
response. `K2` identifies the onset and direction of distortion, not a complete
large-amplitude surrogate.

## Supported, rejected, and unidentified links

Supported:

- correct infinitesimal local expression occupies a meaningful amplitude
  interval for the group mean;
- `H>A` has a replicated mid-to-late mean sign crossing;
- most retained subjects cross, but their brackets are broad;
- output-opposed quadratic curvature is present and especially strong for
  `H>A`;
- the other seven relation means remain correct at the natural amplitude.

Rejected:

- the distortion is merely uniform positive attenuation;
- all relations share one amplitude threshold;
- global clipping is the final mechanistic solution;
- a relation label oracle, relation LOO operation, new memory store, or fitted
  readout is justified;
- more post hoc localization on seeds 1901 and 1902 is needed before v2.

Still unidentified until a new-seed v2 test:

- which online combination of query state, total fast-weight drive, and
  recurrent operating state best predicts a safe amplitude;
- whether a bounded amplitude gate is sufficient or a near-linear residual
  expression is needed for early-crossing subjects;
- whether the proposed expression mechanism improves human-like local fidelity
  while preserving global assembly.

## v2 decision

The machine outcome is
`heterogeneous_or_nonreplicated_crossings_register_online_relation_conditioned_v2`
because the frozen `0.60` regime boundary places the two seeds in adjacent
categories. Scientifically, both seeds select the same core direction:

> Register an online relation-/state-conditioned expression mechanism, rather
> than a globally fixed amplitude reduction.

A minimal candidate should compute a bounded positive gate from quantities
available in the normal forward pass—for example the reset-query state, total
fast-weight recurrent drive, and baseline activation statistics. It must not
use relation LOO, true labels, correctness, or a relation-identity oracle.

The frozen one-to-three-new-seed v2 pilot must compare at least:

1. the original v1 network;
2. the online conditioned gate;
3. a matched global-scalar gate control;
4. a selective gate-off or shuffled-gate causal control.

It advances only if learned-pair fidelity improves, global expected-rank
assembly/query binding/nonlearned inference remain competent, and selective
intervention removes the local rescue in the manner predicted by the gate.
This is a mechanism test, not temperature tuning to pass the formal behavioral
scalars.

## Reproduction

```bash
direnv exec . python -m fsrl.operator_amplitude_path
direnv exec . python -m pytest tests/test_operator_amplitude_path.py -q
```

The frozen protocol is `benchmarks/operator_amplitude_path_v1.json`; the
machine-readable result is `results/operator_amplitude_path_v1.json`.
