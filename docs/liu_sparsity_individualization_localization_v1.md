# Liu sparsity individualization localization v1

## Registered outcome

`ORDER_CONVERGENCE_WITHOUT_REPLICATED_STABLE_ERROR_LOSS`

Across both nested graph families and all three backbones, the mean pairwise
Kendall tau among all 77 participant orders increases significantly with
evidence density. The registered stable-error-incidence slope is negative in
all six analyses, but its confidence interval excludes zero in only one.

This localizes the earlier E=10 boundary as follows:

> Denser compatible evidence reliably makes participant-specific constructed
> orders more similar, more true-order aligned, and lower in stable-error
> burden. It does not reliably eliminate the binary event that a participant
> retains at least one 80-percent-stable error.

The result does not repair the parent
`SPARSITY_DEPENDENT_OR_UNRESOLVED` outcome. It explains the background against
which one E=10 stable-error-prevalence interval missed its threshold.

## Frozen read-only scope

The analysis uses only the participant rows and exact-decision summaries
already stored in `results/liu_evidence_sparsity_transport_v1.json`. It loads no
checkpoint, performs no neural forward pass, resamples no choices, and adds no
graph, density, feature, or threshold.

Participant index is paired across `E=7,8,9,10` within every family and
backbone. All-participant estimands avoid the changing noncorrect-ranker
denominator used by the historical primary link. Correct rankers count as zero
stable-error incidence. Bootstrap draws are paired across all four densities
and remain separate by family and network.

All source-hash, cell-count, participant-alignment, finite-value,
subjective-order-permutation, and decomposition-identity gates pass. The exact
identity

`all-participant stable incidence = noncorrect incidence * stable prevalence conditional on noncorrect`

has maximum numerical error `1.11e-16`.

## Primary localization

Slopes are changes per added observed relation. Every interval is a paired
participant bootstrap within one family and one backbone.

| Seed | Family | Stable-error incidence slope | All-participant pairwise-tau slope |
| ---: | --- | ---: | ---: |
| 2101 | Liu cycle | -.02208 `[-.04675,.00130]` | +.02957 `[.01857,.04012]` |
| 2101 | balanced branched | -.01818 `[-.04156,.00390]` | +.02691 `[.01467,.03906]` |
| 2102 | Liu cycle | -.02338 `[-.05325,.00649]` | +.02911 `[.01631,.04075]` |
| 2102 | balanced branched | -.02857 `[-.05455,-.00519]` | +.02377 `[.01279,.03490]` |
| 2103 | Liu cycle | -.02078 `[-.04545,.00260]` | +.02431 `[.01344,.03444]` |
| 2103 | balanced branched | -.01688 `[-.04286,.00909]` | +.02455 `[.01201,.03668]` |

The order-convergence effect is replicated in all six analyses, with slopes
`.02377--.02957`. Stable-error incidence has a consistently negative point
direction (`-.02857---.01688`) but only one registered PASS. Repeated negative
point estimates are not treated as six significant results, and the five
intervals crossing zero are not called equivalence.

All-participant pairwise tau rises visibly in every family and network. For
example, the E=7 to E=10 change ranges from approximately `.0640` to `.0962`.
It stays far below identity, so density attenuates rather than abolishes
individualization.

## Registered diagnostics

Three convergent diagnostics clarify what changes beneath the binary
stable-error incidence:

- Stable-error *count* decreases in all six analyses, with slopes
  `-.3883---.2416`; every confidence interval is below zero.
- Subjective-order Kendall tau to true rank increases in all six, with slopes
  `+.01939--+.02449`; every interval is above zero.
- Intact exact overall error decreases in all six, with slopes
  `-.01220---.00891`; every interval is below zero.

Noncorrect-ranker incidence decreases significantly in only two analyses. The
remaining four intervals cross zero. Thus density chiefly reduces how many
stable wrong pair judgments a subject carries and aligns whole orders more
closely, while many subjects still retain at least one error and remain
classified as noncorrect.

The paired E=10 balanced-branched-minus-Liu-cycle contrasts do not localize a
family-specific effect. In every backbone, intervals cross zero for
stable-error incidence, stable-error count, truth alignment, exact overall
error, noncorrect incidence, and all-participant pairwise tau. The lone parent
failure therefore should not be promoted into a replicated
balanced-branched-specific mechanism.

## Interpretation

The sparsity evidence now supports two simultaneous claims:

1. The `P_T/a_T` causal division is stable across all tested densities, and the
   full eight-link conjunction transports cleanly through `E=9`.
2. Increasing compatible evidence progressively regularizes the individualized
   global ranking: subjects become more mutually similar and true-order
   aligned, with fewer stable erroneous pairs.

The E=10 seed-2103 lower-bound miss is scientifically consistent with this
general convergence process, but it remains only one failed registered link.
The binary “at least one stable error” endpoint saturates near one and changes
less decisively than stable-error burden. This endpoint sensitivity explains
the heterogeneity without changing the frozen pass/fail decision.

The result also fits the rejected density-allocation prediction. More evidence
strengthens dependence on `P_T` while global orders converge; `P_T` is therefore
better described as an evidence-integrating coherent constructor than as a
fallback recruited only when direct evidence is sparse. `a_T` continues to
preserve direct fidelity rather than becoming a dominant all-query policy.

## Provenance and next step

- Registration commit: `619500eaef3aa5a9bedb3ab74397d14cc7e81969`.
- Implementation/source-lock commit:
  `4d2631c50a3e028d7235245321840f04abc21c9d`.
- CPU execution: one process with BLAS/OMP thread limits set to one.
- Registered result SHA-256:
  `70d00237f3aeef0897f31c558008179bccc1ff2d5ec07993084b73aea513e06a`.
- An independent execution is byte-identical.

This resolves the required sparsity localization without rerunning a network.
Item-count transport can now be designed, but it must retain the correct
boundary: test whether the causal organization scales with problem size, while
reporting individualization as a density-sensitive continuum rather than
requiring the ten-edge parent result to become a retrospective PASS. Freeze
item counts, graph construction, cue embedding, evidence magnitudes, model
compatibility, competence gates, and size-specific interpretations before any
new training or evaluation. List linking, human experiments, MEG, Miconi
ancestry, and new global compression remain deferred.
