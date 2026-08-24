# Relation trace localization v1.1

## Status and scientific question

This post-formal exploratory diagnostic is complete on the two frozen
development checkpoints, seeds 1901 and 1902. It asked where retained local
evidence preserves relation-conditioned identity along the implemented path

\[
\text{generated effective write}
\rightarrow
\text{terminal effective fast-weight state}
\rightarrow
\text{query-conditioned hidden response}.
\]

No network was trained or changed, no checkpoint, subject, or relation was
selected, and no formal seed was accessed. All eight relations and all 77
virtual subjects were retained under the original stable-omission masks.

The initially frozen v1 execution stopped before computing primary trace
estimands. Its `32 * float32 epsilon` intended-endpoint tolerance was tighter
than the `64 * float32 epsilon` discrepancy already recorded by the frozen
support-write result. The failed v1 contract remains unchanged. The separately
frozen v1.1 contract changed only that floating implementation tolerance to
`64 * float32 epsilon`; all scientific estimands, nulls, bootstrap rules, and
outcome interpretations remained identical. It was committed and pushed as
`45db22a` before the trace estimands were run.

The registered outcome tree initially had an implementation error that mapped
the observed non-monotonic presence pattern to the wrong label. Commit
`017e5c2` corrected only that Boolean mapping, added a regression test, and the
complete analysis was rerun. No scalar result or threshold changed.

The result is a replicated mixed pattern:

> A shared same-relation prototype is not detected in generated or terminal
> effective fast-weight matrices, but relation identity appears after the
> first query-conditioned recurrent transition in both the full hidden
> influence and its direct Hodge residual.

This does not establish that the terminal state lacks relation information. It
rejects a static cross-subject Euclidean-prototype account and points instead
to a query-conditioned functional code.

## Frozen estimands

For each support slot containing relation \(r\), matched plus-evidence and
zero-evidence branches begin from the same natural fast-weight state. Their
realized difference is accumulated over the four exposures:

\[
C_{r,s}^{\mathrm{raw}}
=\sum_{t:e_t=r}(P_{t,+}-P_{t,0}),
\qquad
C_{r,s}^{\mathrm{eff}}
=\alpha\odot C_{r,s}^{\mathrm{raw}}.
\]

The primary generated-write representation is \(C^{\mathrm{eff}}\), because
the implemented recurrent matrix is \(W+\alpha\odot P\).

The terminal relation-LOO trace is

\[
\Delta P_{r,s}=P_{T,s}-P_{T,s}^{(-r)},
\qquad
\Delta W^{\mathrm{eff}}_{r,s}=\alpha\odot\Delta P_{r,s}.
\]

For query relation \(r\) at recurrent step \(k\), the direct hidden influence
is the orientation-antisymmetric intact-minus-LOO difference. Step 0 is an
exact-zero implementation control: query hidden state is reset, so the
fast-weight matrix cannot act until the next recurrence. Step 1 is both the
first fast-weight-sensitive state and the registered behavioral response. Its
full direct influence and direct complete-graph Hodge residual are primary;
steps 2 and 3 are post-response descriptions only.

For every layer and held-out subject fold, each of the eight relation
prototypes is the unit-normalized mean trace for that same relation among
retained training subjects. A held-out trace is compared with all eight
prototypes. The two registered identity estimands are:

- matching-prototype cosine minus the mean cosine to the other seven;
- eight-way matching-prototype accuracy, with fixed chance `0.125`.

A layer passes within a seed only when the participant-bootstrap lower bounds
are above zero and `0.125`, respectively, and stable-omitted traces are zero.
Replication requires an independent pass in both seeds. This fits no probe and
does not use correctness labels.

## Integrity and execution

- Every registered source, checkpoint, configuration, and behavioral artifact
  matches its frozen SHA-256.
- Natural incremental and final support endpoints reproduce the evaluator
  exactly in both seeds.
- Intended-versus-realized matched endpoint discrepancies are `6.139e-6` and
  `3.561e-6`, below the frozen v1.1 tolerance `7.629e-6`.
- Hidden-to-logit reconstruction errors are `1.143e-6` and `1.196e-6`, also
  below that tolerance.
- Step-0 query influence and every stable-omitted trace are exactly zero.
- Neural replay ran on the RTX 5090 with PyTorch intra-op and inter-op thread
  counts fixed to one.
- Two consecutive complete runs produced the same result SHA-256:
  `8248e72452e77f6d488a27935a04c39505fe2d32cd2c6c45a81ca1428d7a9ae0`.

## Primary results

Intervals are the frozen 95% participant bootstrap after averaging retained
relations within subject. Identity presence requires both registered metrics,
not either one in isolation.

| Layer and estimand | Seed 1901 | Seed 1902 | Replicated presence |
| --- | ---: | ---: | ---: |
| Generated effective write: selectivity | `-0.01727 [-0.04603, 0.01167]` | `-0.00930 [-0.03954, 0.02191]` | No |
| Generated effective write: 8-way accuracy | `0.12523 [0.09552, 0.15648]` | `0.11707 [0.08813, 0.14734]` | No |
| Terminal effective state: selectivity | `0.00773 [-0.00971, 0.02553]` | `0.01299 [-0.00398, 0.03000]` | No |
| Terminal effective state: 8-way accuracy | `0.11619 [0.08901, 0.14371]` | `0.12424 [0.09261, 0.15727]` | No |
| Step-1 full hidden: selectivity | `0.001414 [0.001241, 0.001588]` | `0.000944 [0.000811, 0.001077]` | Yes |
| Step-1 full hidden: 8-way accuracy | `0.30093 [0.25807, 0.34372]` | `0.23108 [0.19470, 0.26801]` | Yes |
| Step-1 Hodge residual: selectivity | `0.44707 [0.39825, 0.49482]` | `0.35771 [0.31837, 0.39735]` | Yes |
| Step-1 Hodge residual: 8-way accuracy | `0.28431 [0.24191, 0.32780]` | `0.29247 [0.25779, 0.32910]` | Yes |

### Negative result: no shared static matrix prototype

Neither the generated nor terminal effective matrix passes either seed. The
raw generated and intended-write controls also fail, and terminal raw state
fails the conjunction in both seeds. Norms are substantial and omitted traces
are exactly zero, so this is not a zero-signal or masking failure. It says that
retained relations do not share the registered cross-subject, same-relation
matrix geometry at these two layers.

This result must not be expanded into “the fast-weight state stores no local
information.” Prototype classification is not an information-complete test,
and a matrix can encode a reproducible operation without having a directly
shared flattened-vector identity.

### Positive result: query-conditioned relation identity emerges at response

Both step-1 hidden representations pass independently in both seeds. The full
hidden selectivity is numerically small because it sits within a large shared
global component, but its eight-way accuracy is well above chance. Removing
the additive graph-gradient component makes the matching-relation selectivity
large while preserving above-chance identification.

Every relation has positive mean Hodge-residual selectivity in both seeds, but
eight-way accuracy is heterogeneous. `H>A` remains the strongest identity
trace: its residual selectivity is `1.331/1.174` and its accuracy is
`0.855/0.855`. The earlier hidden-residual audit showed that this same strong
trace is correctness-opposed. Relation identity at response therefore cannot
be relabeled as human learned-pair fidelity.

### Secondary trajectory and geometry

Step 0 is exactly zero, as required. Relation identity persists after the
registered choice at steps 2 and 3 in both seeds, but those steps cannot explain
the choice causally.

The prototype RDM of generated effective writes correlates only moderately
with the terminal effective RDM (`0.402/0.509`). Response full-hidden and Hodge
RDMs are negatively correlated with the terminal effective RDM
(`-0.308/-0.301` and `-0.354/-0.390`). Because terminal prototypes themselves
do not pass the primary identity test, these secondary correlations are not
evidence for a literal geometry inversion. They do show that the response code
is not a simple preservation of the registered terminal prototype geometry.

## Supported, rejected, and unidentified links

Supported:

\[
\text{retained evidence}
\xrightarrow{\text{causal relation LOO}}
\text{query-conditioned relation identity at step 1},
\]

including a strong non-additive direct component. This adds a positive link to
the existing global expected-rank assembly backbone.

Rejected under the exact registered estimand:

\[
\text{relation identity}
=
\text{one shared flattened-matrix prototype in generated or terminal state}.
\]

Still unidentified:

- whether terminal state contains relation identity as a functional operator
  rather than a static vector geometry;
- whether the step-1 identity is carried by the LOO state relation, supplied by
  the query basis, or produced by their matched interaction;
- the transformation that would make this relation-specific computation
  correctness-aligned and human-like.

The registered hierarchy therefore returns
`mixed_pattern_requires_new_registered_hierarchy` in both seeds. It neither
justifies a second fixed response readout nor licenses a new persistent memory
module.

## Revised theory and next decisive test

At step 0, intact and relation-LOO runs have the same query-driven hidden state
\(h_e^{(0)}\). The first fast-weight-dependent preactivation contrast is

\[
\Delta a_{r\to e}^{(1)}
=
(\alpha\odot\Delta P_r)h_e^{(0)}.
\]

The observed non-monotonic pattern is therefore compatible with a
query-conditioned operator code: relation identity may be absent from the
flattened \(\Delta W_r^{\mathrm{eff}}\) geometry yet emerge when that operator
acts on a relation-specific query state.

Before v2 training, freeze a two-seed read-only `state_query_operator_binding`
protocol. For every state-trace relation \(q\) and query relation \(e\), form
the full `8 x 8` factorial action

\[
A_{q,e}=(\alpha\odot\Delta P_q)h_e^{(0)}.
\]

It should separately register:

1. state-relation identification while query identity is held or crossed out;
2. query-relation identification while state relation is held or crossed out;
3. the matched `q=e` advantage over norm-matched mismatched pairs;
4. exact reconstruction of the observed step-1 preactivation and hidden
   contrast, plus stable-omitted zero controls.

If state identity or a matched interaction is present, v1 already contains a
co-adapted functional storage/access mechanism and the missing link is the
fidelity transformation. If only query identity is present, the current
response result is query-basis structure acting on a non-specific state
perturbation, and a local storage mechanism becomes more plausible. If state
identity is present in operator action but lost after the recurrent
nonlinearity, the missing operation is routing/expression.

Keep this human local-fidelity question separate from seed 2009's complete-write
implementation heterogeneity. Do not train dual-channel v2 or fit another
response readout before this operator-binding test is frozen and run on seeds
1901 and 1902.

## Reproduction

```bash
direnv exec . python -m fsrl.relation_trace_localization
direnv exec . python -m pytest tests/test_relation_trace_localization.py -q
```

The frozen contracts are
`benchmarks/relation_trace_localization_v1.json` and
`benchmarks/relation_trace_localization_v1_1.json`. The machine-readable result
is `results/relation_trace_localization_v1_1.json`.
