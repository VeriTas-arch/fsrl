# Assembly slope and Hodge diagnostics v1

## Status and scientific question

This read-only diagnostic is complete for both frozen pilot checkpoints. The
analysis was registered in `benchmarks/assembly_diagnostics_v1.json` and pushed
as commit `4423df4` before execution. It used seeds 1901 and 1902 without
retraining, checkpoint selection, temperature fitting, or seed filtering.

The question was whether the excessive neural symbolic-distance slope arises
from:

1. the stable-omission evidence model itself;
2. neural over-sharpening relative to exact inference; or
3. a human local/conjunctive component missing from the network.

The positive evidence backbone remains held-out graph transfer, fast-weight
necessity, human-scale individuality under stable omission, and antisymmetric
subjective-rank geometry. This diagnostic tests how those results may be
assembled; it does not requalify them.

## Registered estimands

For every canonical pair, positive field values mean that the first item is
preferred as higher. Human fields use observed choice proportions, exact fields
use posterior pair probabilities from the realized support evidence, and neural
choice fields use the frozen orientation-antisymmetric margin at the fixed
temperature 0.25. Neural margins are also analyzed directly.

Each cohort-mean complete-graph field is decomposed as

\[
F_{ij}=(s_i-s_j)+c_{ij},
\]

where the first term is the orthogonal Hodge gradient projection and the second
is the residual. Participant bootstrap resampling occurs before computing each
cohort mean. This reduces the risk of treating the human ten-trial-per-pair
sampling floor as individual conjunctive code.

The projected gradient field is not clipped. Its linear accuracy component can
leave the interval \([0,1]\); only the recombined total is a choice probability.
Gradient and residual slopes are additive decomposition estimands, not separate
psychometric models.

## Results

| Source | Total slope | Gradient energy fraction | Learned correctness-aligned residual effect |
| --- | ---: | ---: | ---: |
| Human, 77 participants | 0.03983 | 0.89720 | 0.05585 |
| Exact posterior | 0.00612 | 0.78190 | 0.01132 |
| Neural seed 1901 | 0.04703 | 0.93680 | 0.01311 |
| Neural seed 1902 | 0.04770 | 0.93780 | 0.01825 |

The last column is the learned-pair coefficient after adjustment for symbolic
distance. Positive values indicate a residual contribution aligned with the
correct response beyond the global Hodge component.

### The evidence model does not produce the excessive slope

The exact-posterior minus human slope difference is

\[
-0.03371,\qquad 95\%\ \mathrm{bootstrap\ interval}
=[-0.04037,-0.02747].
\]

Thus stable omission plus the registered veridical-magnitude likelihood does
not explain the steep neural slope. It predicts a much flatter, already highly
accurate distance profile. The registered evidence-model-contribution direction
is negative.

### The neural transformation produces the slope

Relative to the exact posterior, the neural slope increments are:

| Seed | Neural minus exact slope | 95% bootstrap interval |
| --- | ---: | ---: |
| 1901 | 0.04091 | [0.03543, 0.04678] |
| 1902 | 0.04158 | [0.03643, 0.04702] |

Both registered directional tests pass. The frozen network is therefore not
merely expressing the uniform-prior exact posterior. Some part of the
meta-learned neural transformation commits or sharpens the evidence-based field
into a distance-sensitive global policy.

The analytic neural-minus-human slope interval narrowly includes zero for seed
1901 (`[-0.00006, 0.01440]`) and is positive for seed 1902
(`[0.00070, 0.01498]`). This does not replace the registered sampled behavioral
NO-GO. It separates the frozen expected pair field from the finite registered
choice sample and prevents the seed-1901 analytic contrast from being called an
equivalence result.

### Neural margins are almost purely additive

The Hodge gradient fraction of the cohort-mean neural logit margin is:

\[
0.998684\quad(1901),\qquad 0.999400\quad(1902).
\]

The corresponding mean individual-subject fractions are `0.99580` and
`0.99686`, so the cohort result is not created only by cancellation of unrelated
subject residuals.

The corresponding neural choice fields remain strongly gradient dominated
(`0.93680` and `0.93780`), both above the human field. Neural-minus-human
gradient-fraction intervals are `[0.01623, 0.06106]` and
`[0.01825, 0.06162]`.

This is a new positive mechanism result at the policy level: the network
converts sparse, partially retained evidence into an almost perfectly additive
global logit potential. It does not establish that the full hidden state or
fast-weight matrix is itself one-dimensional.

### Human choices retain a stronger learned-pair residual

The human learned-pair, correctness-aligned residual coefficient exceeds the
neural value by:

| Seed comparison | Human minus neural | 95% bootstrap interval |
| --- | ---: | ---: |
| 1901 | 0.04274 | [0.01607, 0.06988] |
| 1902 | 0.03760 | [0.01071, 0.06505] |

The preregistered mixed-code direction therefore replicates across both pilot
seeds. At the behavioral-field level, humans combine a global gradient with a
larger learned-relation residual, whereas the neural logit policy is almost a
pure additive potential.

This result is consistent with a retained local/conjunctive trace, but it does
not yet prove an episodic-memory mechanism. A psychometric nonlinearity,
population heterogeneity, or another structured confidence process could also
produce residual choice structure.

## Theory revision

The rejected path was:

\[
\text{stable omission + veridical magnitude}
\longrightarrow
\text{excessive distance slope}.
\]

The evidence instead supports the working path:

\[
D_s
\longrightarrow
P_T^{(s)}
\longrightarrow
\underbrace{s_i^{(s)}-s_j^{(s)}}_{\text{near-pure neural logit potential}}
\longrightarrow
\pi_s,
\]

while human choices are better described provisionally by

\[
F^{\mathrm{human}}_{ij}
=(s_i-s_j)+c_{ij},
\]

with a stronger learned-pair-aligned residual. The key missing explanation is
now the neural commitment operation: how a non-additive exact-posterior choice
field is transformed into one coherent additive potential, and why this
operation suppresses the learned-pair residual expressed in human behavior.

## Claim boundary

- The result concerns frozen pair logits and choice fields, not the entire
  hidden-state manifold.
- It does not identify whether commitment occurs in fast-weight formation,
  recurrent query dynamics, or the final readout.
- The exact comparison is conditional on the registered uniform-prior,
  reliability-weighted magnitude model and posterior temperature 0.05.
- The human residual is a behavioral signature consistent with mixed coding,
  not yet causal evidence for a local memory store.
- The two pilot seeds remain workflow-validation seeds and do not become formal
  confirmation evidence.

## Next decisive tests

The next analysis should remain read-only and use the same two checkpoints.

1. Save the fast-weight state after each support trial and freeze-query all 28
   pairs. Measure when gradient fraction, global order, and unobserved-pair
   logits emerge. A global-assembly mechanism predicts distributed changes to
   pairs that were not just presented; pair accumulation predicts local changes.
2. Remove each support relation counterfactually and measure its influence on
   every unobserved pair. Structured nonlocal influence would establish the
   missing `D_s -> P_T -> pi_s` assembly link.
3. If commitment is already visible before final readout, use ambiguous evidence
   sets to identify a reproducible meta-learned prior. If it appears only at
   readout, analyze neural over-sharpening before changing the evidence model.

Do not add a local-memory module or run formal seeds 2001-2010 before these
diagnostics distinguish the alternatives.

## Reproduction

```bash
direnv exec . python -m fsrl.assembly_diagnostics
direnv exec . python -m unittest tests.test_assembly_diagnostics -v
```

The machine-readable result is `results/assembly_diagnostics_v1.json`.
Two consecutive registered runs produced SHA-256
`a768495fe84b2d74a4b258d0d14cbfb8d7057c2fd4c3675fc006b1f010a19500`.
