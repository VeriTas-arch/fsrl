# Operator-output semantics v1

## Status and question

This registered post-formal diagnostic is complete on frozen development seeds
1901 and 1902. The protocol was committed and pushed as `d97fb7e` before any
new semantic estimand was executed. The implementation was committed as
`6c3653f`; the first execution stopped before writing a result because NumPy
advanced indexing transposed the subject and remote-edge axes. Commit
`6258773` fixed only that axis selection, added a 77-by-15 regression test, and
was pushed before the successful executions.

No network was trained or changed, no probe or readout was fit, and no formal
seed was accessed. The question was whether the already-confirmed
query-addressed operator produces a correct learned-pair value, or whether that
value is corrupted by its recurrent operating point or the finite-amplitude
nonlinearity.

The answer is replicated and mechanistically asymmetric:

> The bound operator value is correctness-aligned, and the local Jacobian makes
> it more strongly aligned. The exact finite-amplitude response then degrades
> correctness substantially. Aggregate correctness remains positive, but the
> prospectively registered `H>A` relation reverses from strongly correct at
> both `A` and `J_b A` to strongly incorrect after the exact tanh response.

Thus v1 does not lack learned-pair value generation. Its missing fidelity link
is relation-conditioned nonlinear expression. A single scalar local gain
cannot repair a relation that is already expressed with the wrong sign.

## Frozen nested maps

For retained support relation `q`, terminal relation-LOO operator

\[
M_q=\alpha\odot(P_T-P_T^{(-q)})
\]

acts on the reset-query step-0 state `h_(e,o)^0`. The registered maps are

\[
A_{q,e,o}=M_qh_{e,o}^{(0)},
\]

\[
J_{q,e,o}
=
J_{b_{q,e,o}}A_{q,e,o},
\qquad
J_b=\operatorname{diag}[1-\tanh^2(b)],
\]

and

\[
H_{q,e,o}
=
\tanh(b_{q,e,o}+A_{q,e,o})-\tanh(b_{q,e,o}),
\]

where `b` is the complete response-step preactivation under the relation-LOO
baseline. The trained action-margin covector

\[
w_{out}=W_{out}[1]-W_{out}[0]
\]

is held fixed at its trained scale. No normalization or target refit is used.
Each nested map forms a 28-edge orientation-antisymmetric scalar field

\[
F_q^X(e)=\frac12
\left[w_{out}^{\top}X_{q,e,+}-w_{out}^{\top}X_{q,e,-}\right],
\qquad X\in\{A,J,H\}.
\]

The registered complete-graph Hodge residual is

\[
C_q^X=(I-Q)F_q^X.
\]

For the direct edge of relation `q`, raw correctness is

\[
d_X(q,s)=y_q C_q^X(q),
\]

with `y_q` fixed by the true order. The locality control subtracts mean
correctness on the 15 disjoint remote edges. To distinguish semantic direction
from amplitude, each subject also has

\[
\rho_X(s)=
\frac{\sum_{q\in R_s}y_qC_q^X(q)}
{\sqrt{\sum_{q\in R_s}[C_q^X(q)]^2}\sqrt{|R_s|}},
\]

where `R_s` contains all retained relations. A common positive rescaling can
change `d_X`, but not ρ.

## Integrity and competence

- Both seeds, all 77 subjects, eight state relations, 28 query edges, and both
  query orientations are retained.
- Every registered source and pilot artifact matches its frozen SHA-256.
- Manual and intact-versus-LOO step-0 hidden states reproduce exactly.
- `b+A` reproduces intact response preactivation within `1.907e-6`.
- Exact `H` reproduces intact-minus-LOO hidden state within `5.588e-7`.
- `w_out^T H` reproduces the actual logit-margin influence within `2.489e-6`.
- The bilinear identity
  `w_out^T M_q h0 = (M_q^T w_out)^T h0` reproduces within `1.431e-6`.
- Every stable-omitted oriented scalar, Hodge field, and residual is exactly
  zero at all three stages.
- Neural replay ran on the RTX 5090 with PyTorch intra-op and inter-op threads
  fixed to one.
- Two complete executions under the final implementation produced the same
  result SHA-256:
  `b38f577ad83ffe64866724252e2ff03279185beb748972a7298099855c85837a`.

## Aggregate semantics

Intervals are frozen 95% participant bootstraps. Raw direct correctness first
averages all retained relations within each subject.

| Estimand | Seed 1901 | Seed 1902 |
| --- | ---: | ---: |
| `A` direct correctness | `0.04853 [0.04460, 0.05274]` | `0.05216 [0.04720, 0.05729]` |
| `A` normalized correctness ρ | `0.79318 [0.77005, 0.81547]` | `0.68584 [0.64613, 0.72260]` |
| `J` direct correctness | `0.18527 [0.17315, 0.19720]` | `0.15727 [0.14614, 0.16882]` |
| `J` normalized correctness ρ | `0.89581 [0.88319, 0.90809]` | `0.87265 [0.85902, 0.88636]` |
| `H` direct correctness | `0.03750 [0.02681, 0.04790]` | `0.04875 [0.04049, 0.05737]` |
| `H` normalized correctness ρ | `0.35846 [0.27797, 0.43815]` | `0.46857 [0.39509, 0.54367]` |

All three stages pass the prospectively frozen aggregate alignment rule in both
seeds. Storage, access, and relation-specific value generation therefore form
a positive chain; the actual policy still retains a positive local residual on
average.

### The operating point improves semantics

| Paired transition | Seed 1901 | Seed 1902 |
| --- | ---: | ---: |
| `J-A` direct correctness | `0.13675 [0.12667, 0.14673]` | `0.10511 [0.09741, 0.11279]` |
| `J-A` ρ | `0.10263 [0.08274, 0.12413]` | `0.18681 [0.15442, 0.22059]` |

Both raw correctness and normalized semantic direction improve. The recurrent
operating point is not merely attenuating a correct value and is not the source
of the fidelity failure. State-dependent local sensitivity rotates or
reweights the fixed output covector in a beneficial direction at first order.

### Finite-amplitude expression degrades semantics

| Paired transition | Seed 1901 | Seed 1902 |
| --- | ---: | ---: |
| `H-J` direct correctness | `-0.14778 [-0.16286, -0.13334]` | `-0.10852 [-0.12010, -0.09706]` |
| `H-J` ρ | `-0.53735 [-0.62139, -0.45410]` | `-0.40408 [-0.48092, -0.32563]` |

This is not pure attenuation: normalized direction degrades strongly in both
seeds. Exact tanh curvature discards most of the first-order correctness gain.
Aggregate `H` remains positive because seven relation trajectories are largely
positive, but averaging conceals a decisive heterogeneous failure.

## Prospective `H>A` discriminator

`H>A` was registered before execution because it previously combined the
strongest relation identity with a correctness-opposed response residual.

| `H>A` direct correctness | Seed 1901 | Seed 1902 |
| --- | ---: | ---: |
| `A` operator value | `0.10596 [0.09040, 0.12138]` | `0.12219 [0.09693, 0.14892]` |
| `J_b A` linearized expression | `0.28270 [0.24753, 0.31840]` | `0.24433 [0.20669, 0.28323]` |
| `H` exact expression | `-0.28899 [-0.34202, -0.23556]` | `-0.18363 [-0.22971, -0.13686]` |

The trajectory is unambiguous: value generation is correct, first-order
operating-point expression strengthens it, and finite-amplitude expression
reverses its sign. `H>A` is the only relation that is robustly opposed at `H`
in both seeds; `F>A` is unresolved at `H` in seed 1901 and positive in seed
1902, while the other six relations remain positive in both.

This replicated relation-specific reversal rules out a global scalar local
gain as the missing mechanism. Increasing the existing exact response would
strengthen the wrong `H>A` value. It also rules out replacing storage, fitting a
new response readout, or blaming the relation-LOO operating point.

## Locality and Hodge boundary

Direct-minus-remote correctness and direct-minus-remote absolute residual
magnitude are positive at every stage in both seeds. For example, at exact `H`
the correctness specificity is `0.02380 [0.01245, 0.03502]` and
`0.03810 [0.02941, 0.04715]`. The semantic effect is therefore selectively
direct rather than a relabeled global remote field.

The full scalar fields nevertheless remain mostly additive: the mean residual
energy fraction ranges from about `0.00077` at `A` to `0.00778` at `H` in seed
1901 and from about `0.00106` to `0.00526` in seed 1902. This audit concerns the
small direct non-additive component, not a replacement for the confirmed
global expected-rank computation.

## Supported, rejected, and unidentified links

Supported:

\[
\Delta P_q
\rightarrow M_q
\rightarrow A_{q,e}
\rightarrow J_bA_{q,e}
\]

now includes relation-specific correctness, binding, and local specificity.
The exact response retains aggregate correctness but introduces a replicated
relation-conditioned sign failure.

Rejected under the frozen tests:

- learned-pair value generation is absent from the operator output;
- the relation-LOO operating point corrupts an otherwise correct value at
  first order;
- finite-amplitude tanh is only a positive scalar attenuation;
- one global scalar local gain can restore human-like fidelity;
- a second memory store or fitted readout is the next justified modification.

Still unidentified:

- which hidden-unit saturation or curvature terms cause the `H>A` sign
  crossing;
- whether the reversal occurs smoothly as operator amplitude grows or through
  a narrow interaction with the LOO baseline;
- the smallest online-computable recurrent modification that preserves the
  correct linearized local value without damaging global assembly.

The registered outcome is
`aggregate_aligned_but_H_greater_A_opposed` in both seeds.

## Revised theory and next decisive test

The current explanatory chain is

\[
D_s
\rightarrow P_T^{(s)}
\rightarrow
\begin{cases}
\text{global expected-rank assembly},\\
\text{query-bound local operator value}
\rightarrow\text{beneficial first-order expression}
\rightarrow\text{relation-conditioned nonlinear corruption}.
\end{cases}
\]

The v2 target is therefore fidelity-preserving nonlinear expression, not new
storage, generic gain, or another readout. Before choosing an architecture,
freeze one final read-only amplitude-path discriminator on the same two
development seeds:

\[
H_q(\lambda)
=
\tanh(b_q+\lambda A_q)-\tanh(b_q),
\qquad
\lambda\in[0,1].
\]

Use a fixed prospective grid and report all relations, with `H>A` primary and
the same Hodge, remote, and stable-omission controls. If only large lambda
causes the sign crossing, a magnitude-bounded or residualized expression rule
is the minimal v2 candidate. If the sign changes immediately away from zero or
depends discontinuously on the baseline, v2 needs a relation-conditioned
routing transformation rather than simple gain control. Any v2 candidate must
be online-computable without a relation-LOO oracle, preserve the original
global path, and begin with one to three new development seeds under selective
causal ablations.

## Reproduction

```bash
direnv exec . python -m fsrl.operator_output_semantics
direnv exec . python -m pytest tests/test_operator_output_semantics.py -q
```

The frozen protocol is `benchmarks/operator_output_semantics_v1.json`; the
machine-readable result is `results/operator_output_semantics_v1.json`.
