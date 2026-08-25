# State-query operator binding v1

## Status and question

This registered post-formal diagnostic is complete on frozen development seeds
1901 and 1902. The protocol was committed and pushed as `249b86c` before any
new factorial estimand was executed. The implementation was source-locked as
`4368e52`; a subsequent `35a0db4` change replaced an expected empty-slice
warning for omitted states with an explicit masked mean, without changing the
estimand, and the complete run was repeated.

No network was trained or changed, no checkpoint, subject, or relation was
selected, and no formal seed was accessed. The question was whether the
terminal relation-LOO fast-weight trace stores relation identity as a
query-keyed operator rather than a shared flattened-matrix embedding, and
whether the recurrent operating point creates or merely expresses that
identity.

The registered outcome is positive and replicated:

> Terminal fast weights implement a basis-dependent, query-keyed relation
> operator. Relation identity is already present in its preactivation action,
> is preferentially bound to the matching query, and is preserved with a
> smaller matched gain advantage by the recurrent nonlinearity.

This establishes functional storage and access for relation-specific
computation under the frozen estimands. It does not establish episodic
retrieval, correctness-aligned fidelity, a rank-one key-value memory, or a
human-like local channel.

## Frozen factorial

For state relation \(q\), the terminal effective relation-LOO operator is

\[
M_{q,s}
=
\alpha\odot
\left(P_{T,s}-P_{T,s}^{(-q)}\right).
\]

For query relation \(e\) and orientation \(o\), hidden state is reset before
the query. The step-0 query state \(h_{e,o}^{(0)}\) is therefore identical
under intact and relation-LOO fast weights. The registered `8 x 8` factorial
separates the direct operator action

\[
A_{q,e,o}=M_qh_{e,o}^{(0)}
\]

from the exact nonlinear hidden effect

\[
H_{q,e,o}
=
\tanh\left(b_{q,e,o}+A_{q,e,o}\right)
-
\tanh\left(b_{q,e,o}\right),
\]

where \(b_{q,e,o}\) is the complete step-1 preactivation under the
relation-LOO baseline. Identity tests use the orientation-antisymmetric
vectors \(\frac12(A_+-A_-)\) and \(\frac12(H_+-H_-)\).

Two state-identity tests were frozen:

1. **Fixed-query:** build held-out-subject same-\(q\) prototypes separately
   within each query basis \(e\).
2. **Cross-query:** additionally leave out \(e\) and build each \(q\) prototype
   from the other seven query bases.

Both use matching-prototype selectivity and eight-way accuracy with fixed
chance `0.125`. Fixed-query success with cross-query failure means that state
identity is functional but basis-dependent.

Binding uses an operator- and key-norm-normalized gain:

\[
g^A_{q,e,o}
=
\frac{\|M_qh_{e,o}^{(0)}\|}
{\|M_q\|_F\|h_{e,o}^{(0)}\|}.
\]

The matched \(q=e\) cell is compared prospectively with exactly two
shared-endpoint mismatches and five disjoint mismatches per state relation.
Binding requires both participant-bootstrap matched advantages to be above
zero.

## Integrity and competence

- Both declared seeds, all eight state relations, all eight query relations,
  both orientations, and all 77 subjects are retained.
- Every registered source and pilot artifact matches its frozen SHA-256.
- Manual step-0 hidden states and intact-versus-LOO step-0 invariance reproduce
  exactly.
- `b+A` reproduces intact step-1 preactivation to `1.431e-6` in both seeds.
- The nonlinear expression reproduces actual intact-minus-LOO step-1 hidden
  state to `4.768e-7` and `3.949e-7`.
- Every stable-omitted effective operator, action, and hidden effect is exactly
  zero.
- Neural replay ran on the RTX 5090 with PyTorch intra-op and inter-op thread
  counts fixed to one.
- Two complete executions under the final implementation produced the same
  result SHA-256:
  `7d334170296b7f5e55509089f4dc0a2757531cdc314f6b9c8ae4dd6c9997487c`.

## Primary results

Intervals are the frozen 95% participant bootstrap after averaging retained
state-query cells within subject.

| Layer and state-ID test | Seed 1901 | Seed 1902 | Replicated presence |
| --- | ---: | ---: | ---: |
| `A`, fixed-query selectivity | `0.68780 [0.67438, 0.70114]` | `0.67184 [0.65807, 0.68567]` | Yes |
| `A`, fixed-query 8-way accuracy | `0.43607 [0.42049, 0.45286]` | `0.43885 [0.42335, 0.45452]` | Yes |
| `A`, cross-query selectivity | `-0.02789 [-0.03114, -0.02473]` | `-0.01999 [-0.02209, -0.01799]` | No |
| `A`, cross-query 8-way accuracy | `0.06541 [0.05710, 0.07401]` | `0.06072 [0.05339, 0.06816]` | No |
| `H`, fixed-query selectivity | `0.69016 [0.67644, 0.70365]` | `0.68434 [0.67052, 0.69805]` | Yes |
| `H`, fixed-query 8-way accuracy | `0.44098 [0.42620, 0.45673]` | `0.44459 [0.42993, 0.45920]` | Yes |
| `H`, cross-query selectivity | `-0.03392 [-0.03782, -0.03002]` | `-0.02461 [-0.02738, -0.02197]` | No |
| `H`, cross-query 8-way accuracy | `0.06209 [0.05340, 0.07083]` | `0.06437 [0.05628, 0.07273]` | No |

### Positive result: functional state identity is present before tanh

At fixed query basis, \(A_{q,e}\) identifies the state relation far above
chance in both seeds. The effect is not driven by one relation: mean
per-relation selectivity ranges from `0.652` to `0.729` in seed 1901 and from
`0.646` to `0.713` in seed 1902. Stable-omitted actions are exactly zero, so
the identity is causally inherited from retained support evidence.

This resolves the previous static-matrix negative. A flattened \(M_q\) does
not have one shared same-relation prototype, but its action on a fixed query
basis has strong, cross-subject state identity. Persistent information is
therefore stored functionally as an operator, not as the previously tested
static embedding.

### Positive result: matching query binding is strong and structured

| Normalized operator gain | Seed 1901 | Seed 1902 |
| --- | ---: | ---: |
| Matched `q=e` | `0.38644 [0.38193, 0.39095]` | `0.37875 [0.37428, 0.38321]` |
| Shared-endpoint mismatch | `0.11737 [0.11514, 0.11951]` | `0.11694 [0.11480, 0.11901]` |
| Disjoint mismatch | `0.06432 [0.06357, 0.06509]` | `0.06485 [0.06400, 0.06569]` |
| Matched minus shared endpoint | `0.26907 [0.26423, 0.27387]` | `0.26180 [0.25679, 0.26681]` |
| Matched minus disjoint | `0.32212 [0.31766, 0.32660]` | `0.31390 [0.30952, 0.31834]` |

Every relation independently has positive bootstrap lower bounds for both
matched contrasts in both seeds. Shared endpoints produce an intermediate
gain, but exact relation matching adds a much larger advantage. This is
state-query binding rather than a generic response to overlapping item cues.

### Negative result: the state code is not query invariant

Cross-query prototypes are not merely unresolved; selectivity is negative and
accuracy is below chance in both seeds and both layers. A state relation does
not map to one common action direction that transports across query bases.
Instead, its functional identity is organized in the joint \((q,e)\)
coordinate system.

The symmetric query controls support the same conclusion. Fixed-state query
identity is strong after operator action, but cross-state query identity fails.
The step-0 query key alone also fails the cross-subject identity rule:
accuracy is `0.136 [0.115, 0.159]` and `0.141 [0.120, 0.164]`, with selectivity
intervals crossing zero. The abstract relation identity is therefore not
supplied by the query cue alone; it emerges through the state-by-query
interaction.

## Nonlinear attribution

The nonlinearity is not the source of state identity. `A` already passes both
fixed-query gates, and `H` retains nearly the same accuracy. The paired
`H-minus-A` accuracy changes are unresolved (`0.00491 [-0.00594, 0.01517]` and
`0.00574 [-0.00378, 0.01528]`). Selectivity increases slightly in both seeds:
`0.00236 [0.00019, 0.00464]` and `0.01249 [0.00995, 0.01514]`.

The recurrent operating point nevertheless provides a smaller matched gain
advantage. The hidden-to-action norm ratio is `0.889/0.907` for matched cells,
`0.844/0.856` for shared-endpoint mismatches, and `0.815/0.831` for disjoint
mismatches. Thus the primary relation binding is already present in the linear
operator action, while state-dependent tanh sensitivity selectively reinforces
its expression.

## Supported, rejected, and unidentified links

Supported:

\[
\Delta P_q
\xrightarrow{\alpha}
M_q
\xrightarrow{h_e^{(0)}}
A_{q,e}
\xrightarrow{\tanh\text{ at }b_{q,e}}
H_{q,e},
\]

with relation-specific state identity at fixed query basis and preferential
matched access. This is positive evidence for a query-addressable synaptic
operator already present in v1.

Rejected under the frozen tests:

- persistent relation identity is a shared flattened-matrix embedding;
- operator output has a query-invariant state direction;
- step-0 query cues alone explain the abstract relation identity;
- tanh creates the identity de novo.

Still unidentified:

- what relation-specific quantity the matched operator output encodes;
- where its direction becomes correctness-opposed for relations such as
  `H>A`;
- how to transform the confirmed local operator output into the human
  learned-pair residual without damaging global expected-rank assembly.

The registered outcome is
`query_keyed_operator_missing_fidelity_transformation` in both seeds. The
word “operator” is warranted by the factorial action and causal LOO controls;
“retrieval,” “rank-one key-value memory,” and “correct local answer” are not.

## Revised theory and next decisive test

The v1 mixed code is now more precisely bounded:

\[
D_s
\rightarrow
\begin{cases}
P_T^{(s)}\rightarrow\text{global expected-rank policy},\\
M_qh_e^{(0)}\rightarrow\text{basis-dependent local computation}.
\end{cases}
\]

Storage and query access no longer motivate adding a separate edge-memory
module. The missing scientific link is the semantics of the operator output:

\[
\text{matched local operator response}
\rightarrow
\text{correctness-aligned fidelity residual}.
\]

Before training v2, freeze a two-seed read-only operator-output semantics
audit over all 28 query edges. With the current fixed output direction and no
new probe, compare the direct Hodge-residual correctness at three nested maps:

\[
W_{out}^{\top}A,
\qquad
W_{out}^{\top}J_bA,
\qquad
W_{out}^{\top}\left[\tanh(b+A)-\tanh(b)\right],
\]

where \(J_b=\operatorname{diag}(1-\tanh^2 b)\). This distinguishes an
incorrect operator value direction from corruption by the local operating
point or finite-amplitude nonlinearity. It must retain the direct-versus-remote
and stable-omission controls and report `H>A` prospectively rather than fitting
a relation-specific readout.

If correctness is present in \(W_{out}^{\top}A\) but lost in \(J_bA\) or the
exact hidden effect, v2 needs a fidelity-preserving expression rule. If all
three stages are correctness-opposed or absent, v2 needs a new
relation-specific value-generation transformation while preserving the
confirmed operator storage and global channel. Do not fit a second response
readout, add a new memory module, or use formal seeds for this decision.

Keep seed 2009's complete-write implementation heterogeneity separate.

## Reproduction

```bash
direnv exec . python -m fsrl.state_query_operator_binding
direnv exec . python -m pytest tests/test_state_query_operator_binding.py -q
```

The frozen protocol is `benchmarks/state_query_operator_binding_v1.json`; the
machine-readable result is `results/state_query_operator_binding_v1.json`.
