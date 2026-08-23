# History state-by-factor closure

## Question and frozen scope

This diagnostic asks why the fourth-exposure write generated after removing
prior presentations of the same relation restores policy magnitude when it is
evaluated on its own altered history, but not reliably when all generated
factors are replayed on one natural fast-weight baseline.

The machine-readable contract is
[`benchmarks/history_state_factorial_v1.json`](../benchmarks/history_state_factorial_v1.json).
It was committed as `03abc72` before implementation or execution. The analysis
uses only frozen pilot checkpoints 1901 and 1902, exposure 4, and effective
support steps 2 and 3. It does not train, change an architecture or checkpoint,
access formal seeds 2001--2010, refit a nuisance parameter, or alter the older
`confirmation_v1` contract.

For each retained subject-relation, the registered 2-by-2 replay crosses:

- `B_N`: the zero-magnitude current-trial effective-connectivity baseline after
  natural accumulated history;
- `B_H`: the same baseline after replaying earlier trials while zeroing prior
  presentations of the current relation;
- `C_N`: the matched total effective factor generated under natural history;
- `C_H`: the matched total effective factor generated under no-prior-relation
  history.

The four neural-potential changes are
`Y_NN = R(B_N+C_N)-R(B_N)`, `Y_NH = R(B_N+C_H)-R(B_N)`,
`Y_HN = R(B_H+C_N)-R(B_H)`, and
`Y_HH = R(B_H+C_H)-R(B_H)`. No cell is norm matched.

## Estimands and validation

The scalar outcome `Q_ab` is the L2 norm of the centered Hodge potential
`Y_ab`. The registered contrasts are:

- factor generation: `F = [(Q_NH-Q_NN) + (Q_HH-Q_HN)] / 2`;
- baseline expression: `B = [(Q_HN-Q_NN) + (Q_HH-Q_NH)] / 2`;
- interaction: `I = Q_HH-Q_HN-Q_NH+Q_NN`;
- matched-history contrast: `M = I/2`;
- actual restoration: `Q_HH-Q_NN`.

`M` is retained for interpretation but is algebraically dependent on `I`, so it
is not a second piece of evidence. Vector factorial contrasts are computed
before taking their norms. Directional quality uses the same subject and
relation's natural neural-potential update at its first exposure as an
independent target.

All implementation gates passed. The largest replay or factor-composition
error was `1.19e-6` in seed 1901 and `1.67e-6` in seed 1902, below the registered
float32 reproduction tolerance `3.81e-6`. Stable-omitted relations had exactly
zero factors and cell updates. `Y_NN` and `Y_HH` reproduced their registered
natural and no-prior-history source branches.

## Results

The table reports subject means with paired subject-bootstrap 95% intervals.

| Seed | `Q_NN` | `Q_NH` | `Q_HN` | `Q_HH` |
| --- | ---: | ---: | ---: | ---: |
| 1901 | 1.06076 | 1.07882 | 1.09455 | 1.11794 |
| 1902 | 1.22841 | 1.24709 | 1.25450 | 1.27735 |

| Seed | Generation `F` | Baseline `B` | Interaction `I` | Restoration `Q_HH-Q_NN` |
| --- | ---: | ---: | ---: | ---: |
| 1901 | 0.02073 [-0.00039, 0.04157] | 0.03645 [0.03227, 0.04091] | 0.00533 [0.00322, 0.00750] | 0.05718 [0.03349, 0.08103] |
| 1902 | 0.02076 [-0.00010, 0.04241] | 0.02817 [0.02494, 0.03147] | 0.00418 [0.00254, 0.00600] | 0.04893 [0.02611, 0.07276] |

The scalar baseline-expression effect and interaction are positive in both
pilots. Total factor generation is positive in mean but unresolved in both; it
is not promoted to a supported causal locus. The actual matched-history
restoration is positive in both.

The vector effects are not numerical zeros. Mean norms for
`(V_F, V_B, V_I)` were `(0.14332, 0.04944, 0.01209)` in seed 1901 and
`(0.13975, 0.03952, 0.00871)` in seed 1902. At the same time, all four
registered cross-cell direction cosines were very high: their bootstrap lower
bounds ranged from `0.99923` to `0.99976`. History therefore produces detectable
vector changes while preserving the dominant update direction.

Against the independent first-exposure same-relation target, baseline
expression and interaction were again positive in both seeds. The alignment
baseline means were `0.000547` and `0.000511`; interaction means were
`0.0000774` and `0.0000786`, with all four lower intervals above zero. The
alignment generation effect was positive in seed 1901 but unresolved in seed
1902. These effects are small and describe reproducible local changes around
already-high cell alignments (`0.9883`--`0.9918`), not a wholesale redirection.

## Mechanistic revision

Together with the prior write-localization and factor-swap positives, this
closure supports a meta-learned state-dependent iterative-relaxation mechanism:

1. eligibility carries evidence-specific write content and transferable
   relation direction;
2. modeled DA changes write and policy magnitude while largely preserving that
   direction;
3. alpha places generated writes in systematically high-gain recurrent
   directions;
4. accumulated fast-weight state changes the local recurrent sensitivity
   landscape in which a write is expressed;
5. history-matched factor and baseline combinations have a small positive
   nonlinear advantage.

The negative result is informative: the closure does not support a standalone
increase in total factor generation. It rules out completing the explanation
with eligibility or a scalar learning-rate story alone. The supported route is
that accumulated state primarily changes expression of the write, with an
additional matched interaction, while the generated factor retains nearly the
same dominant direction.

This is a computational mechanism in frozen trained networks. It does not
identify modeled DA with biological dopamine, establish exact sequential
Bayesian updating, or show that correctness-aligned remote propagation is
positive. Terminal distributional expected-rank alignment and immediate global
causal reach remain separate supported links.

## Decision

The registered pilot stop rule is met: both source cells reproduce within
tolerance, and baseline expression plus the interaction resolve prospectively
in both pilots. Mechanism-discovery pilots stop here. A separate
`mechanism_confirmation_v1` is frozen before any formal-seed access; the older
`confirmation_v1` remains unchanged. Formal results must report every declared
seed and preserve unsupported development contrasts, including total factor
generation, as non-primary diagnostics rather than silently redefining them.
