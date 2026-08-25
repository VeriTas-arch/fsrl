# Liu item-count transport v1

## Registered conclusion

The registered outcome is **`LIU_ITEM_COUNT_MECHANISM_TRANSPORTED`**. All eight
primary links pass independently in all nine size-by-backbone cells (N=6, 8,
10; seeds 2101, 2102, 2103). This supports transport of the frozen
`P_T`/`a_T` functional organization across the registered six-to-ten-item
cycle family.

This is not a claim of scale-invariant performance. N=6 and N=10 are strict
cardinality OOD tests for backbones meta-trained only at N=8, and item count
co-varies prospectively with support count, support duration, query count, and
direct-query fraction.

## Frozen design and provenance

- Registration commit: `9577d381c205516a9eac2a82a935d4ddcaafca19`.
- Implementation-lock commit:
  `33ea99a094f835ca56a59b3e60b61f28b41a9c5b`.
- Frozen sizes: N=6, 8, 10, with one connected degree-two cycle (`E=N`) and
  four passive presentations per relation.
- The N=6 and N=10 cycles were selected before model evaluation by exact
  Wasserstein matching of normalized rank-distance distributions to the N=8
  Liu cycle, with lexicographic tie-breaking. The respective optima were
  `8/105` (7 minimizers) and `19/420` (110 minimizers).
- Checkpoints, local gains, evidence-admission equation, cue width, recurrent
  update, local address/write/read, activation, readout, temperature, seeds,
  and bootstrap rules were unchanged.
- Evaluation used the NVIDIA GeForce RTX 5090 through the bounded runtime,
  with one PyTorch intra-op and one inter-op CPU thread.
- Two complete GPU executions were byte-identical. The result SHA-256 is
  `2429ef71c849d1a39b2b8f4d60748f054c90342d93e2130fc8cae6dd87b3eee5`.
- Participants were bootstrapped within each size and backbone. No participants
  or networks were pooled.

The variable-item evaluator is isolated from the frozen N=8 evaluator. In all
three backbones, its N=8 cue codes, support schedules, subject states,
admissions, legacy metrics, intervals, and schedule hashes replay exactly.

## Primary result

Ranges below are across the three backbones at each size. Accuracy entries are
participant means; CI-bound entries report the most conservative registered
bound across the three cells.

| N | learned exact accuracy | nonlearned exact accuracy | intact Hodge fraction | Hodge-order tau | all cells |
|---:|---:|---:|---:|---:|:---:|
| 6 | 0.905--0.922 | 0.825--0.833 | 0.989--0.991 | 0.713--0.733 | PASS |
| 8 | 0.945 | 0.809--0.824 | 0.989--0.990 | 0.706--0.723 | PASS |
| 10 | 0.930--0.936 | 0.749--0.759 | 0.988--0.989 | 0.585--0.608 | PASS |

All learned and nonlearned exact-accuracy lower bounds remain above 0.5.
Intact and `a`-off Hodge-fraction lower bounds exceed 0.95, transitive-triplet
lower bounds exceed 0.95, and Hodge-order tau lower bounds are positive in all
nine cells.

### Global `P_T` role transports

- Under `P`-off/`a`-on, the largest nonlearned correct-probability upper bound
  is 0.489 at N=6, 0.473 at N=8, and 0.476 at N=10.
- The upper bound of local remote influence minus one quarter of intact global
  remote influence remains negative at every size (most conservative values:
  -0.063, -0.053, and -0.044).
- Disjoint relation-LOO remote-effect lower bounds remain positive (at least
  0.331, 0.294, and 0.245), as do third-party relational lower bounds (at least
  0.160, 0.193, and 0.200).

Thus `P_T` continues to support nonlearned and remote relational assembly; it
is not an eight-item lookup table under the registered transport test.

### Local `a_T` role transports

- The lower bound of intact-minus-`a`-off learned correct probability is at
  least 0.0096 at N=6, 0.0105 at N=8, and 0.0126 at N=10.
- With `P` off and `a` intact, learned correct-probability lower bounds remain
  at least 0.615, 0.636, and 0.652, while learned-minus-nonlearned lower bounds
  remain at least 0.148, 0.183, and 0.201.
- The exact edge-ledger reconstruction has maximum tensor-state error
  `6.66e-16` and maximum all-query raw-read error `8.88e-15`, both below the
  frozen `1e-12` threshold.

The local trace therefore continues to preserve direct evidence without
becoming a transitive global solver.

### Individualized coherent structure transports

The upper bound on mean inter-subject order tau is at most 0.609, 0.600, and
0.506 for N=6, 8, and 10. The lower bound on the size-normalized 80%-stable
error-pair density is at least 0.088, 0.088, and 0.131. The historical binary
prevalence was reported but not gated because its opportunity count changes
with N.

## Quantitative boundary and revised theory

The mechanism survives cardinality transport, but global performance changes
systematically with size:

- nonlearned exact accuracy falls from 0.825--0.833 at N=6 to 0.749--0.759 at
  N=10;
- mean Hodge-order tau falls to 0.585--0.608 at N=10;
- global relation-LOO remote magnitude decreases with N;
- normalized symbolic-distance slope rises from 0.268--0.304 at N=6, through
  0.348--0.356 at N=8, to 0.383--0.392 at N=10;
- sampled overall accuracy falls to 0.781--0.791 at N=10.

These are not failures of the registered functional-asymmetry links. They show
that a stable algorithmic division can coexist with a cardinality-sensitive
global confidence/allocation regime. The supported theory is therefore:

```text
broader weak direct evidence -> a_T -> exact edge-addressed fidelity
selectively admitted evidence -> P_T -> coherent nonlocal assembly
                                      -> quantitatively size-sensitive policy
```

The result upgrades the mechanism from one fixed eight-item benchmark instance
to the registered Liu-style sparse-cycle task family. It does not establish
arbitrary-size scaling, a unique cycle family, human mechanism, biological
storage, or network-population prevalence.

## Stop/go decision

The preregistered Liu structural-generalization sequence—topology,
presentation order, sparsity, and item count—is now complete. Do not tune the
N=6/N=10 outputs, retrain on those sizes, or add post-hoc cardinalities. The
next useful Liu-mainline step is a read-only provenance and claim synthesis
across the four frozen transport experiments. Any later intervention on the
size-dependent global-policy fingerprint requires a separate prospective
contract and must preserve the confirmed local/global decomposition.
